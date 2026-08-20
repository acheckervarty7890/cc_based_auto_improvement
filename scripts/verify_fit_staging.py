#!/usr/bin/env python
"""Regression checks for ``retrain._to_device_for_fit``.

Run:

    .venv_claude/bin/python scripts/verify_fit_staging.py

Staging activations on the GPU before ``ProbeFactory.build`` is a pure placement
change, so nothing downstream reports when it picks the wrong tensor — the fit just
gets slower, by up to 3x. These checks pin the behaviour that is easy to regress:

* the **largest** dataset is staged first, whatever order the caller passes, because
  every dataset is traversed once per epoch and the biggest one moves the most bytes;
* a dataset is staged **whole or not at all** (tuberlens' ``Activation.__post_init__``
  multiplies ``activations`` by ``attention_mask``, so a split dataset raises);
* the ``AGENTIC_REDTEAM_STAGE_ACTIVATIONS`` / ``..._FIT_STAGING_RESERVE_GIB`` knobs work.

No GPU is required: ``_fit_device`` is pointed at torch's ``meta`` device (``.to()``
succeeds and allocates nothing) and ``_allocatable_bytes`` at a synthetic budget, so
the placement decisions are exercised without needing a card of any particular size.
"""

from __future__ import annotations

import os
import sys

import torch

from agentic_redteam import retrain as R

MB = 2**20
_BYTES_PER_ROW = 4096  # activations + attention_mask, both 1024-wide fp16


class FakeDataset:
    """Minimal stand-in for a tuberlens ``LabelledDataset``."""

    def __init__(self, name: str, mb: int) -> None:
        self.name = name
        rows = max((mb * MB) // _BYTES_PER_ROW, 1)
        self.other_fields = {
            "activations": torch.zeros(rows, 1024, dtype=torch.float16),
            "attention_mask": torch.ones(rows, 1024, dtype=torch.float16),
            "labels": "not-a-tensor",  # must be ignored, not moved
        }

    @property
    def staged(self) -> bool:
        return self.other_fields["activations"].device.type == "meta"

    def field_devices(self) -> set[str]:
        return {
            v.device.type for v in self.other_fields.values() if torch.is_tensor(v)
        }


def stage(budget_mb: int, datasets, *, reserve_mb: int | None = 0, env=None):
    """Run the real ``_to_device_for_fit`` against a synthetic budget."""
    saved = (R._fit_device, R._allocatable_bytes, torch.Tensor.to)
    saved_env = {k: os.environ.get(k) for k in (R._STAGING_ENV, R._STAGING_RESERVE_ENV)}
    for k, v in (env or {}).items():
        os.environ[k] = v

    used = {"n": 0}
    R._fit_device = lambda: "meta"
    R._allocatable_bytes = lambda _device: budget_mb * MB - used["n"]

    real_to = torch.Tensor.to

    def counting_to(self, *args, **kwargs):
        out = real_to(self, *args, **kwargs)
        if args and args[0] == "meta":
            used["n"] += self.numel() * self.element_size()
        return out

    torch.Tensor.to = counting_to
    try:
        R._to_device_for_fit(
            list(datasets),
            verbose=False,
            reserve_bytes=None if reserve_mb is None else reserve_mb * MB,
        )
    finally:
        R._fit_device, R._allocatable_bytes, torch.Tensor.to = saved
        for k, v in saved_env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return [d.name for d in datasets if d.staged]


FAILURES: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got={got!r} want={want!r}")
    if not ok:
        FAILURES.append(label)


def main() -> int:
    print("1. when only one fits, the LARGER one is staged regardless of caller order")
    # A big dev set against a small training set: `--dev-data dev_samples/highstakes`.
    for order in ("train-first", "dev-first"):
        big, small = FakeDataset("dev-1908", 400), FakeDataset("train-666", 140)
        passed = [small, big] if order == "train-first" else [big, small]
        check(f"passed {order}", stage(420, passed), ["dev-1908"])

    print("2. the MIRRORED config stages the training set — same code, no flag")
    # A small dev set against a grown red-team training set: hu_ha / instructions, and
    # every run using the default split instead of --dev-data. Hardcoding either end of
    # this pair strands the other; before the sort, this case staged the 60 MB set and
    # left the 400 MB one copying every epoch.
    big_train, small_dev = FakeDataset("train-978", 400), FakeDataset("dev-290", 60)
    check("small dev / big train", stage(420, [big_train, small_dev]), ["train-978"])

    print("3. an ample budget stages everything; a tiny one stages nothing")
    a, b = FakeDataset("train", 100), FakeDataset("dev", 100)
    check("ample budget", stage(4096, [a, b]), ["train", "dev"])
    a, b = FakeDataset("train", 400), FakeDataset("dev", 400)
    check("tiny budget", stage(10, [a, b]), [])

    print("4. all-or-nothing per dataset — no dataset straddles two devices")
    dev, train = FakeDataset("dev", 400), FakeDataset("train", 140)
    stage(420, [train, dev])
    check("staged dataset is wholly on device", dev.field_devices(), {"meta"})
    check("skipped dataset is wholly on host", train.field_devices(), {"cpu"})

    print("5. environment knobs")
    a, b = FakeDataset("train", 10), FakeDataset("dev", 10)
    check(
        f"{R._STAGING_ENV}=0 disables staging",
        stage(4096, [a, b], env={R._STAGING_ENV: "0"}),
        [],
    )
    a = FakeDataset("dev", 400)
    check("a reserve larger than the headroom blocks it", stage(420, [a], reserve_mb=100), [])
    a = FakeDataset("dev", 400)
    check("a small reserve allows it", stage(420, [a], reserve_mb=1), ["dev"])
    a = FakeDataset("dev", 400)
    check(
        "reserve_bytes=None falls back to the 2 GiB env default",
        stage(420, [a], reserve_mb=None),
        [],
    )

    print("6. reserve parsing")
    os.environ[R._STAGING_RESERVE_ENV] = "0.5"
    check("0.5 -> 0.5 GiB", R._staging_reserve_bytes(), int(0.5 * 2**30))
    os.environ[R._STAGING_RESERVE_ENV] = "banana"
    check("unparseable -> default", R._staging_reserve_bytes(), int(2.0 * 2**30))
    os.environ[R._STAGING_RESERVE_ENV] = "-1"
    check("negative -> default", R._staging_reserve_bytes(), int(2.0 * 2**30))
    os.environ.pop(R._STAGING_RESERVE_ENV, None)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
