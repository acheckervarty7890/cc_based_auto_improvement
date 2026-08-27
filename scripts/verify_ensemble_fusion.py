#!/usr/bin/env python
"""Regression checks for the fused-vs-sequential ensemble switch.

Run:

    .venv_claude/bin/python scripts/verify_ensemble_fusion.py

``PROBE_FUSED_ENSEMBLE`` selects between fitting an ensemble's members together
(one vmapped pass over the activations) and fitting them one at a time, and
between scoring them together and scoring them member by member. Both settings
must produce the same *kind* of object — the difference is wall-clock only — so
nothing downstream reports when the switch stops being honoured; the fast path
would simply keep running with the knob turned off. These checks pin:

* ``ensemble.fusion_enabled()`` tracks the setting, in-process and from the
  environment;
* a fit dispatches to ``ProbeFactory.build_ensemble`` when it is on and to this
  repo's own per-seed ``ProbeFactory.build`` loop when it is off — **not** to
  ``build_ensemble``'s internal sequential fallback, which seeds ``torch`` alone
  where ``_build`` calls ``seed_everything``. Turning fusion off is how a run is
  compared against the pre-fusion ones, so it has to land on the path those took;
* each member of a sequential fit is seeded with its own pinned ``ENSEMBLE_SEEDS``
  entry, in order;
* scoring consults the switch, so an ``EnsembleProbe`` reverts too.

No GPU and no probe fit are required: ``ProbeFactory`` is stubbed out, so what is
exercised is the dispatch, not the arithmetic (which is tuberlens' to verify).
"""

from __future__ import annotations

import os
import subprocess
import sys

import torch
from tuberlens.config import global_settings
from tuberlens.probes.probe_factory import ProbeFactory

from agentic_redteam import retrain as R
from agentic_redteam.ensemble import EnsembleProbe, fusion_enabled

SEEDS = [3699, 14431, 23529]

_failures: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {got!r}" + ("" if ok else f" != {want!r}"))
    if not ok:
        _failures.append(label)


class _FakeMember:
    """Enough of a probe for `EnsembleProbe.from_members` and the dispatch checks."""

    model_name = "llama-1b"
    layer = 8
    description = "test"
    pos_class_label = "positive"
    neg_class_label = "negative"
    _classifier = None  # not a PytorchAdamClassifier, so fused scoring cannot apply

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def predict_proba(self, dataset):
        return [0.5] * len(dataset)


class _FakeDataset(list):
    def __init__(self, n: int) -> None:
        super().__init__(range(n))
        self.other_fields: dict = {}


def fit_calls(fused: bool) -> list:
    """Which ProbeFactory entry point a fit reaches, with the switch in each state."""
    calls: list = []
    saved = (
        ProbeFactory.build,
        ProbeFactory.build_ensemble,
        R._to_device_for_fit,
        R._concatenate_consuming,
        R.EnsembleProbe.from_members,
        global_settings.PROBE_FUSED_ENSEMBLE,
    )
    global_settings.PROBE_FUSED_ENSEMBLE = fused
    # `_build` reseeds through `seed_everything`, so torch's own seed records which
    # member is being fit — that is what distinguishes this loop from the fallback
    # inside `build_ensemble`.
    ProbeFactory.build = staticmethod(
        lambda **kw: calls.append(("build", torch.initial_seed())) or _FakeMember(0)
    )
    ProbeFactory.build_ensemble = staticmethod(
        lambda **kw: calls.append(("build_ensemble", tuple(kw["seeds"])))
        or [_FakeMember(s) for s in kw["seeds"]]
    )
    R._to_device_for_fit = lambda *a, **k: None
    R._concatenate_consuming = lambda parts: _FakeDataset(4)
    R.EnsembleProbe.from_members = staticmethod(lambda members, seeds: (members, seeds))
    try:
        R._train_with_cached_base_activations(
            base_train=None,
            base_val=None,
            extra_train=None,
            extra_val=None,
            dev_val=None,
            model_name="llama-1b",
            layer=8,
            probe_spec=R._coerce_probe_spec("linear_then_softmax"),
            pos_class_label="positive",
            neg_class_label="negative",
            probe_description="test",
            base_train_cache=None,
            base_val_cache=None,
            seed=42,
            ensemble_seeds=list(SEEDS),
            verbose=False,
        )
    finally:
        (
            ProbeFactory.build,
            ProbeFactory.build_ensemble,
            R._to_device_for_fit,
            R._concatenate_consuming,
            R.EnsembleProbe.from_members,
            global_settings.PROBE_FUSED_ENSEMBLE,
        ) = saved
    return calls


def scoring_consults_switch() -> bool:
    """Whether `_fused_proba` bails out when the switch is off.

    The members here are not pytorch heads, so fusion could not apply either way —
    what is being checked is that the switch is consulted *before* that, i.e. that
    the early return exists at all.
    """
    probe = EnsembleProbe.from_members([_FakeMember(s) for s in SEEDS], list(SEEDS))
    seen: list[bool] = []
    saved = global_settings.PROBE_FUSED_ENSEMBLE
    original = EnsembleProbe._fused_proba

    def spy(self, dataset):
        seen.append(fusion_enabled())
        return original(self, dataset)

    EnsembleProbe._fused_proba = spy
    try:
        for state in (True, False):
            global_settings.PROBE_FUSED_ENSEMBLE = state
            probe.predict_proba(_FakeDataset(3))
    finally:
        EnsembleProbe._fused_proba = original
        global_settings.PROBE_FUSED_ENSEMBLE = saved
    return seen == [True, False]


def main() -> int:
    saved = global_settings.PROBE_FUSED_ENSEMBLE
    try:
        global_settings.PROBE_FUSED_ENSEMBLE = True
        check("fusion_enabled() when on", fusion_enabled(), True)
        global_settings.PROBE_FUSED_ENSEMBLE = False
        check("fusion_enabled() when off", fusion_enabled(), False)
    finally:
        global_settings.PROBE_FUSED_ENSEMBLE = saved

    # The setting is read from the environment at import, so this needs a subprocess.
    env = {**os.environ, "PROBE_FUSED_ENSEMBLE": "0"}
    out = subprocess.run(
        [sys.executable, "-c",
         "from agentic_redteam.ensemble import fusion_enabled; print(fusion_enabled())"],
        capture_output=True, text=True, env=env,
    )
    check("PROBE_FUSED_ENSEMBLE=0 in the environment", out.stdout.strip(), "False")

    check("fused fit dispatch", fit_calls(True), [("build_ensemble", tuple(SEEDS))])
    sequential = fit_calls(False)
    check("sequential fit dispatch", [c[0] for c in sequential], ["build"] * len(SEEDS))
    check("sequential fit seeds each member", [c[1] for c in sequential], SEEDS)

    check("scoring consults the switch", scoring_consults_switch(), True)

    print("\n" + ("all checks passed" if not _failures else f"FAILED: {_failures}"))
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
