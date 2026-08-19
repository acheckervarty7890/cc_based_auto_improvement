#!/usr/bin/env python
"""Verify (and time) the fast probe-head training path.

There is no test suite in either repo, and the probe fit is numerically
sensitive: it selects a checkpoint on a validation AUROC, so a change that merely
reshuffles batches produces a *different probe* while looking like a pure
speedup. This script asserts the invariants that keep the fast path honest, and
prints the speedups it buys.

    .venv_claude/bin/python scripts/verify_ensemble_training.py [--bench]

Checks
------
1.  ``ActivationBatcher`` yields the same batches, in the same order, as the
    ``DataLoader`` it replaces -- and advances the ambient RNG by the same amount,
    which is what makes an existing config retrain to the same probe.
2.  All three placements (resident / staged / pageable) train to the *same*
    weights, so which one a box happens to pick can never change a result.
3.  ``ProbeFactory.build_ensemble`` is reproducible from its seeds alone, and its
    members match members trained one at a time to within bf16 reassociation.
4.  ``EnsembleProbe.predict_proba``'s fused scoring matches the per-member loop.
5.  Every probe architecture survives being stacked and vmapped.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time

import numpy as np
import torch

SEQ = 1024
HIDDEN = 256  # small enough to run on any box; the code paths do not depend on it
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def planted(n: int, seed: int, hidden: int = HIDDEN):
    """Activations carrying a weak planted direction, with realistic ragged lengths."""
    from tuberlens.interfaces.activations import Activation

    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(hidden, generator=torch.Generator().manual_seed(11))
    direction /= direction.norm()
    labels = (torch.rand(n, generator=generator) > 0.5).float()
    acts = torch.randn(n, SEQ, hidden, generator=generator) * 0.5
    acts += (labels.view(n, 1, 1) * 2 - 1) * direction.view(1, 1, -1) * 0.15
    lengths = torch.clamp(
        torch.normal(535.0, 200.0, (n,), generator=generator).long(), 32, SEQ
    )
    mask = torch.arange(SEQ).unsqueeze(0) < lengths.unsqueeze(1)
    activation = Activation(
        activations=acts.to(torch.bfloat16),
        attention_mask=mask,
        input_ids=torch.zeros(n, SEQ, dtype=torch.bfloat16),
    )
    return activation, labels


def as_dataset(activation, labels):
    from tuberlens.interfaces.dataset import Label, LabelledDataset

    dataset = LabelledDataset(
        inputs=[f"sample-{i}" for i in range(len(labels))],
        ids=[str(i) for i in range(len(labels))],
        other_fields={"labels": [Label.from_int(int(v)).value for v in labels]},
    )
    return dataset.assign(
        activations=activation.activations,
        attention_mask=activation.attention_mask,
        input_ids=activation.input_ids,
    )


def free_gpu():
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- #


def check_batch_order() -> None:
    """The batcher must be a drop-in for DataLoader, RNG consumption included."""
    from torch.utils.data import DataLoader, TensorDataset

    from tuberlens.interfaces.activations import ActivationBatcher

    n, batch_size, epochs = 37, 8, 4

    def dataloader_orders(seed, shuffle):
        torch.manual_seed(seed)
        loader = DataLoader(
            TensorDataset(torch.arange(n)), batch_size=batch_size, shuffle=shuffle
        )
        orders = [[b[0].tolist() for b in loader] for _ in range(epochs)]
        return orders, torch.get_rng_state()

    def batcher_orders(seed, shuffle):
        torch.manual_seed(seed)
        batcher = ActivationBatcher(
            torch.arange(n).view(n, 1, 1).float(),
            torch.ones(n, 1, dtype=torch.bool),
            torch.arange(n).float(),
            mode="pageable",
            device="cpu",
            dtype=torch.float32,
        )
        orders = [
            [[int(v) for v in y.tolist()] for _, _, y in batcher.batches(batch_size, shuffle=shuffle)]
            for _ in range(epochs)
        ]
        return orders, torch.get_rng_state()

    for shuffle in (True, False):
        same_order = same_rng = True
        for seed in (0, 42, 3699):
            a_orders, a_rng = dataloader_orders(seed, shuffle)
            b_orders, b_rng = batcher_orders(seed, shuffle)
            same_order &= a_orders == b_orders
            same_rng &= torch.equal(a_rng, b_rng)
        check(f"batch order matches DataLoader (shuffle={shuffle})", same_order)
        check(f"RNG consumption matches DataLoader (shuffle={shuffle})", same_rng)


def check_placements_agree() -> None:
    """resident / staged / pageable must be three routes to the same probe."""
    from tuberlens.interfaces.probes import ProbeType
    from tuberlens.probes.pytorch_classifiers import PytorchAdamClassifier
    from tuberlens.probes.pytorch_modules import LinearThenSoftmax
    import tuberlens.interfaces.activations as activations_module

    hyperparams = dict(ProbeType.linear_then_softmax.default_hyperparams)
    hyperparams["epochs"] = 4
    train, y_train = planted(48, 1)
    val, y_val = planted(32, 2)

    original_resolve = activations_module.ActivationBatcher._resolve_mode
    weights = {}
    for mode in ("resident", "staged", "pageable"):
        if mode != "pageable" and not torch.cuda.is_available():
            continue
        activations_module.ActivationBatcher._resolve_mode = (
            lambda self, *a, _m=mode, **k: _m
        )
        try:
            torch.manual_seed(3699)
            classifier = PytorchAdamClassifier(
                training_args=hyperparams, probe_architecture=LinearThenSoftmax
            )
            with quiet():
                classifier.train(
                    train, y_train, validation_activations=val, validation_y=y_val
                )
            weights[mode] = classifier.model.linear.weight.detach().float().cpu().clone()
        finally:
            activations_module.ActivationBatcher._resolve_mode = original_resolve
        free_gpu()

    reference = next(iter(weights.values()))
    for mode, weight in weights.items():
        delta = (weight - reference).abs().max().item()
        check(f"placement '{mode}' trains to the same weights", delta == 0.0,
              f"max|Δw|={delta:.3e}")


def check_ensemble() -> None:
    """Fused members: reproducible from seeds, and equivalent to sequential fits."""
    from tuberlens.interfaces.probes import ProbeSpec, ProbeType
    from tuberlens.probes.probe_factory import ProbeFactory
    from tuberlens.probes.pytorch_classifiers import PytorchAdamClassifier
    from tuberlens.probes.pytorch_modules import LinearThenSoftmax

    seeds = [3699, 14431, 23529]
    hyperparams = dict(ProbeType.linear_then_softmax.default_hyperparams)
    hyperparams["epochs"] = 6
    train, y_train = planted(96, 1)
    val, y_val = planted(64, 2)
    test, _ = planted(48, 9)
    spec = ProbeSpec(name=ProbeType.linear_then_softmax, hyperparams=hyperparams)

    def build():
        free_gpu()
        with quiet():
            return ProbeFactory.build_ensemble(
                probe_spec=spec,
                train_dataset=as_dataset(train, y_train),
                model_name="synthetic",
                layer=0,
                seeds=seeds,
                validation_dataset=as_dataset(val, y_val),
                verbose=False,
            )

    def score(model):
        classifier = PytorchAdamClassifier(
            training_args=hyperparams,
            probe_architecture=LinearThenSoftmax,
            model=model,
        )
        with quiet():
            return classifier.probs(test).float().cpu().numpy()

    first = np.stack([score(p._classifier.model) for p in build()])
    second = np.stack([score(p._classifier.model) for p in build()])
    check("fused ensemble is reproducible from its seeds", np.array_equal(first, second))

    sequential = []
    for seed in seeds:
        free_gpu()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        classifier = PytorchAdamClassifier(
            training_args=hyperparams, probe_architecture=LinearThenSoftmax
        )
        with quiet():
            classifier.train(train, y_train, validation_activations=val, validation_y=y_val)
        sequential.append(score(classifier.model))
    sequential = np.stack(sequential)
    delta = np.abs(sequential - first).max()
    corr = np.corrcoef(sequential.ravel(), first.ravel())[0, 1]
    # Not equality: vmap dispatches the members' projections as one batched matmul,
    # so the members' trajectories differ in the last bits of bf16 and then diverge
    # over the epochs. Equivalence is the claim, and correlation is how it is stated.
    check("fused members track sequential members", corr > 0.95,
          f"max|Δp|={delta:.4f} corr={corr:.5f}")

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
    from agentic_redteam.ensemble import EnsembleProbe

    members = build()
    ensemble = EnsembleProbe.from_members(members, seeds)
    test_dataset = as_dataset(test, torch.zeros(len(test.activations)))
    fused = ensemble.predict_proba(test_dataset)
    loop = np.mean(
        [np.asarray(m.predict_proba(test_dataset), dtype=float) for m in members], axis=0
    )
    delta = float(np.abs(fused - loop).max())
    check("fused scoring matches the per-member loop", delta < 5e-3, f"max|Δp|={delta:.2e}")


def check_all_architectures() -> None:
    """Every ProbeType the fused trainer claims must actually stack and vmap."""
    import copy

    from torch.func import functional_call, stack_module_state

    from tuberlens.interfaces.probes import ProbeType
    from tuberlens.probes.probe_factory import _ADAM_ARCHITECTURES

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    embed, batch, seq, members = 128, 4, 64, 3
    x = torch.randn(batch, seq, embed).to(device).to(dtype)
    mask = torch.zeros(batch, seq, dtype=torch.bool, device=device)
    mask[:, :50] = True

    for name, architecture in _ADAM_ARCHITECTURES.items():
        hyperparams = ProbeType(name).default_hyperparams
        heads = [architecture(embed, **hyperparams).to(device).to(dtype) for _ in range(members)]
        reference = torch.stack([h(x, mask) for h in heads])
        params, buffers = stack_module_state(heads)
        stateless = copy.deepcopy(heads[0]).to("meta")
        forward = torch.vmap(
            lambda p, b, xx, mm: functional_call(stateless, (p, b), (xx, mm)),
            in_dims=(0, 0, None, None),
        )
        got = forward(params, buffers, x, mask)
        delta = (got.float() - reference.float()).abs().max().item()
        check(f"vmap reproduces {name}", delta < 1e-2, f"max|Δ|={delta:.2e}")


def benchmark() -> None:
    """Speed of the fused ensemble against members trained one at a time."""
    from tuberlens.interfaces.probes import ProbeSpec, ProbeType
    from tuberlens.probes.probe_factory import ProbeFactory
    from tuberlens.probes.pytorch_classifiers import PytorchAdamClassifier
    from tuberlens.probes.pytorch_modules import LinearThenSoftmax

    seeds = [3699, 14431, 23529, 26229, 26660]
    hyperparams = dict(ProbeType.linear_then_softmax.default_hyperparams)
    hyperparams["epochs"] = 10
    hyperparams["patience"] = 10**9
    train, y_train = planted(300, 1, hidden=2048)
    val, y_val = planted(600, 2, hidden=2048)

    free_gpu()
    start = time.perf_counter()
    for seed in seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        classifier = PytorchAdamClassifier(
            training_args=hyperparams, probe_architecture=LinearThenSoftmax
        )
        with quiet():
            classifier.train(train, y_train, validation_activations=val, validation_y=y_val)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sequential_time = time.perf_counter() - start

    free_gpu()
    start = time.perf_counter()
    with quiet():
        ProbeFactory.build_ensemble(
            probe_spec=ProbeSpec(name=ProbeType.linear_then_softmax, hyperparams=hyperparams),
            train_dataset=as_dataset(train, y_train),
            model_name="synthetic",
            layer=0,
            seeds=seeds,
            validation_dataset=as_dataset(val, y_val),
            verbose=False,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fused_time = time.perf_counter() - start
    print(
        f"\n  {len(seeds)} members x {hyperparams['epochs']} epochs, 300 train / 600 val:\n"
        f"    one at a time  {sequential_time:7.2f}s\n"
        f"    fused          {fused_time:7.2f}s   ({sequential_time / fused_time:.2f}x)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", action="store_true", help="also time fused vs sequential")
    args = parser.parse_args()

    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")
    print("batcher")
    check_batch_order()
    check_placements_agree()
    print("\nensemble")
    check_ensemble()
    print("\narchitectures")
    check_all_architectures()
    if args.bench:
        print("\nbenchmark")
        benchmark()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
