"""Self-contained checks for ``agentic_redteam.probe_architectures``.

Runs on synthetic activations in seconds — no gemma-3-27b, no cached blobs — so it can be
run on any box before committing a sweep to hours of fits.

    .venv_claude/bin/python scripts/test_probe_architectures.py

Three things are checked:

1. **Every architecture builds, trains and scores.** Shape and finiteness of the logits,
   plus the parameter count, which is the cheapest way to notice that a "non-linear" head
   silently fell back to a linear one.
2. **The best-epoch restore actually restores** (``docs/attribution_findings.md`` §1). The
   same fit is run twice, once with the fix and once with tuberlens' behaviour; whenever
   early stopping selects an epoch before the last, the returned weights must differ.
   This is the check that would fail silently and bias the whole sweep against the
   higher-capacity heads, for which early stopping is the main overfitting control.
3. **``ProbeType.lda`` is the same probe as ``difference_of_means``.** Asserting the
   *known defect* rather than the desired behaviour, so that if a future tuberlens
   implements ``use_lda`` this test fails and says so instead of the sweep quietly
   reporting one estimator under two names.

Two tuberlens behaviours these tests had to be written around, both worth knowing before
reading any fit's output:

- ``gradient_accumulation_steps`` (4 for most architectures) is applied as
  ``(batch_idx + 1) % steps == 0``, so a training set with **fewer than 4 batches**
  (< 64 rows at ``batch_size`` 16) never calls ``optimizer.step()`` and produces an
  untrained probe with no error. Every fit below is sized above that.
- Padded activation positions must be exactly zero for the mean-pooling heads; the cached
  blobs and the synthetic data here both satisfy that.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

EMBED_DIM = 64
SEQ_LEN = 10

# linear_then_rolling_max convolves a window of 40 tokens, so it cannot run on sequences
# shorter than that. Real conversations are hundreds of tokens; the synthetic ones here
# are 10, and lengthening them for one architecture nobody is sweeping is not worth it.
SKIP = {"sklearn", "linear_then_rolling_max"}


def make_dataset(n: int, seed: int, signal: float):
    """``n`` rows of noise with a linearly separable signal planted in dimension 0."""
    from tuberlens.interfaces.dataset import LabelledDataset, Message

    g = torch.Generator().manual_seed(seed)
    y = (torch.arange(n) % 2).float()
    acts = torch.randn(n, SEQ_LEN, EMBED_DIM, generator=g)
    acts[:, :, 0] += (y[:, None] - 0.5) * signal
    ds = LabelledDataset(
        inputs=[[Message(role="user", content=f"conversation {i}")] for i in range(n)],
        ids=[f"row-{i}" for i in range(n)],
        other_fields={
            "labels": ["positive" if v > 0.5 else "negative" for v in y]
        },
    )
    return ds.assign(
        activations=acts.to(torch.bfloat16),
        attention_mask=torch.ones(n, SEQ_LEN, dtype=torch.bool),
        input_ids=torch.ones(n, SEQ_LEN, dtype=torch.long),
    )


def test_every_architecture_trains() -> list[str]:
    from tuberlens.interfaces.activations import Activation

    from agentic_redteam.probe_architectures import all_architectures, build_probe

    train, val = make_dataset(128, 0, 4.0), make_dataset(64, 1, 4.0)
    failures = []
    print("\n--- 1. every architecture builds, trains and scores ---")
    for arch in [a for a in all_architectures() if a not in SKIP]:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                probe = build_probe(
                    arch, train, val,
                    model_name="synthetic", layer=1,
                    pos_class_label="positive", neg_class_label="negative",
                    hyperparams={"epochs": 4, "patience": 3},
                )
                logits = probe._classifier.logits(Activation.from_dataset(val))
            logits = logits.float().flatten()
            assert logits.numel() == len(val), (
                f"got {logits.numel()} logits for {len(val)} rows"
            )
            assert torch.isfinite(logits).all(), "non-finite logits"
            n_params = sum(p.numel() for p in probe._classifier.model.parameters())
            # difference_of_means / lda have no epochs at all, hence getattr.
            print(f"  {arch:22s} ok   params={n_params:>7d}  "
                  f"best_epoch={getattr(probe._classifier, 'best_epoch', None)}")
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            failures.append(f"{arch}: {exc!r}")
            print(f"  {arch:22s} FAIL {exc!r}")
    return failures


def test_best_epoch_is_restored() -> list[str]:
    """Same fit twice; the fix must return the selected epoch, not the last one."""
    from agentic_redteam.probe_architectures import build_probe

    # Train and validation carry different signal strengths, so validation AUROC peaks
    # and then decays — otherwise the best epoch *is* the last one and there is nothing
    # to restore.
    train, val = make_dataset(256, 0, 3.0), make_dataset(128, 1, 0.6)
    print("\n--- 2. early stopping restores the epoch it selected ---")

    out = {}
    for restore in (True, False):
        torch.manual_seed(7)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            probe = build_probe(
                "linear_then_softmax", train, val,
                model_name="synthetic", layer=1,
                pos_class_label="positive", neg_class_label="negative",
                hyperparams={"epochs": 60, "patience": 5,
                             "optimizer_args": {"lr": 5e-2, "weight_decay": 0.0}},
                restore_best_epoch=restore,
            )
        n_epochs = sum(1 for line in buf.getvalue().splitlines()
                       if line.startswith("Validation AUROC:"))
        weights = probe._classifier.model.linear.weight.detach().float().flatten()
        out[restore] = (probe._classifier.best_epoch, weights.clone(), n_epochs)

    best_fix, w_fix, n_epochs = out[True]
    best_leg, w_leg, n_leg = out[False]
    print(f"  fixed : best_epoch={best_fix} of {n_epochs} epochs run")
    print(f"  legacy: best_epoch={best_leg} of {n_leg} epochs run")
    print(f"  max |w_fixed - w_legacy| = {(w_fix - w_leg).abs().max():.6f}")

    if best_fix != best_leg or n_epochs != n_leg:
        return ["the two runs diverged; they should follow the identical trajectory"]
    if best_fix is None or best_fix >= n_epochs:
        print("  INCONCLUSIVE: the best epoch was the last one — nothing to restore.")
        return []
    if torch.equal(w_fix, w_leg):
        return [
            f"best-epoch restore is still a no-op: epoch {best_fix} of {n_epochs} was "
            "selected but both runs returned identical weights"
        ]
    print(f"  ok: epoch {best_fix} was selected and its weights were restored")
    return []


def test_tuberlens_lda_is_difference_of_means() -> list[str]:
    """Assert the *defect*, so a future fix surfaces as a failure here."""
    from tuberlens.interfaces.activations import Activation

    from agentic_redteam.probe_architectures import build_probe

    train, val = make_dataset(128, 0, 4.0), make_dataset(64, 1, 4.0)
    print("\n--- 3. tuberlens' `lda` is an alias for `difference_of_means` ---")
    scores = {}
    for arch in ("difference_of_means", "lda"):
        with contextlib.redirect_stdout(io.StringIO()):
            probe = build_probe(
                arch, train, val,
                model_name="synthetic", layer=1,
                pos_class_label="positive", neg_class_label="negative",
            )
            scores[arch] = probe._classifier.logits(
                Activation.from_dataset(val)
            ).float().flatten()
    identical = torch.equal(scores["difference_of_means"], scores["lda"])
    print(f"  logits identical: {identical}")
    if not identical:
        return [
            "tuberlens' `lda` now differs from `difference_of_means` — it used to declare "
            "`use_lda` and never read it. Re-check whether `lda_shrinkage` is still needed."
        ]
    print("  ok: confirmed identical, so the sweep uses the local `lda_shrinkage` instead")
    return []


def test_hyperparam_override_reaches_the_trainer() -> list[str]:
    """The ``--normalize-hyperparams`` path must actually change the training args.

    tuberlens ships different optimizer defaults per architecture (``attention`` uses
    batch_size 128 / grad_accum 1 / final_lr 5e-4; the batch-16 group uses 16 / 4 / 1e-4),
    so a comparison across those groups varies the trainer as well as the head. The sweep
    offers a normalised arm to remove that. If the override silently failed to reach
    ``training_args``, the normalised arm would be byte-identical to the un-normalised one
    — a control that looks fine and measures nothing.
    """
    from agentic_redteam.probe_architectures import build_probe, default_hyperparams

    print("\n--- 4. hyperparameter override reaches training_args ---")
    keys = ("batch_size", "gradient_accumulation_steps", "final_lr")
    reference = default_hyperparams("linear_then_softmax")
    shipped = default_hyperparams("attention")
    target = {k: reference[k] for k in keys}
    if all(shipped[k] == v for k, v in target.items()):
        return ["`attention` and `linear_then_softmax` now ship identical optimizer "
                "settings — the normalisation arm is pointless; re-check the defaults"]

    train, val = make_dataset(128, 0, 4.0), make_dataset(64, 1, 4.0)
    failures = []
    for label, override in (("shipped", {}), ("normalised", dict(target))):
        with contextlib.redirect_stdout(io.StringIO()):
            probe = build_probe(
                "attention", train, val,
                model_name="synthetic", layer=1,
                pos_class_label="positive", neg_class_label="negative",
                hyperparams={**override, "epochs": 3, "patience": 2},
            )
        got = probe._classifier.training_args
        print(f"  {label:11s} batch={got['batch_size']:>4} "
              f"accum={got['gradient_accumulation_steps']} final_lr={got['final_lr']}")
        want = target if override else {k: shipped[k] for k in keys}
        failures += [f"{label}: {k} is {got[k]}, expected {v}"
                     for k, v in want.items() if got[k] != v]
    if not failures:
        print("  ok: the override lands, and the two arms genuinely differ")
    return failures


def test_lda_pooling_chunk_is_exact() -> list[str]:
    """Chunking the LDA's float64 cast must bound memory without changing the answer.

    ``_fit_shrinkage_lda`` mean-pools in float64. Casting the whole ``[n, seq, embed]``
    tensor at once is 747 x 1024 x 5376 x 8 B = **32.9 GB** on a real training set, which
    OOM-killed a running sweep; it is chunked to ~176 MB instead. The other tests here all
    use tiny synthetic dimensions (10 tokens, 64 dims) where the whole-tensor cast is a few
    MB, so **none of them can catch that class of bug** — hence this one, which checks the
    property the chunking has to preserve rather than the memory it saves.

    Ragged masks and a sequence length coprime to the chunk sizes, so a boundary error in
    the per-row denominators would show up.
    """
    import agentic_redteam.probe_architectures as PA
    from tuberlens.interfaces.activations import Activation

    print("\n--- 5. chunked float64 pooling in the LDA is exact ---")
    torch.manual_seed(0)
    n, seq, embed = 40, 37, 96
    acts = torch.randn(n, seq, embed)
    y = (torch.arange(n) % 2).float()
    acts[:, :, 0] += (y[:, None] - 0.5) * 3
    mask = torch.zeros(n, seq, dtype=torch.bool)
    for i in range(n):
        mask[i, : 5 + (i * 7) % (seq - 5)] = True
    activation = Activation(
        activations=acts.to(torch.bfloat16),
        attention_mask=mask,
        input_ids=torch.ones(n, seq, dtype=torch.long),
    )

    original = PA._LDA_POOL_CHUNK
    results = {}
    try:
        for chunk in (1, 4, 7, 10_000):
            PA._LDA_POOL_CHUNK = chunk
            with contextlib.redirect_stdout(io.StringIO()):
                results[chunk] = PA._fit_shrinkage_lda(activation, y, "auto")
    finally:
        PA._LDA_POOL_CHUNK = original

    ref_w, ref_b = results[1]
    failures = []
    for chunk, (w, b) in results.items():
        dw = (w - ref_w).abs().max().item()
        db = abs(b - ref_b)
        print(f"  chunk={chunk:<6} max|dw|={dw:.3e}  |db|={db:.3e}")
        if dw > 0 or db > 0:
            failures.append(f"LDA pooling at chunk={chunk} differs (dw={dw:.3e})")
    if not failures:
        print("  ok: bit-identical direction and bias at every chunk size")
    return failures


def main() -> int:
    failures: list[str] = []
    failures += test_every_architecture_trains()
    failures += test_best_epoch_is_restored()
    failures += test_tuberlens_lda_is_difference_of_means()
    failures += test_hyperparam_override_reaches_the_trainer()
    failures += test_lda_pooling_chunk_is_exact()

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
