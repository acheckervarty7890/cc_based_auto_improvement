#!/usr/bin/env python
"""Regression checks for the MultiMax probe architecture.

Run:

    .venv_claude/bin/python scripts/verify_multimax_probe.py

MultiMax (arXiv:2601.11516 Section 3.2.1, Equation 9) is

    f(S_i) = sum_h  max_j  ( v_h^T . y_{i,j} )       y_{i,j} = MLP_M(x_{i,j})

which is a short enough forward that the ways it can be silently wrong are all
structural rather than arithmetic, and none of them would fail a run:

* **Padding.** A max is dominated by its largest element, so a pad position that
  scores high is not diluted the way it would be under a mean or a softmax — it
  simply becomes the answer. A missing mask makes the probe read whatever the
  padding embeds, and the score still looks perfectly ordinary.
* **Heads.** ``sum_h max_j`` and ``max_j sum_h`` are different functions, and the
  second is the one you get by summing the heads before taking the max. It is also
  a strictly weaker probe (one token must explain every head), and it fits.
* **Train/eval divergence.** The fused ensemble paths call the module through
  ``functional_call`` on a ``copy.deepcopy(...).to("meta")`` stand-in that nothing
  ever puts in eval mode, so any forward that branched on ``self.training`` would
  score differently fused than sequentially — a discrepancy this repo's contract
  says cannot happen ("the fast path can only cost speed, never a score").
* **Round-trip.** A retrain reconstructs its architecture from the pickled probe via
  ``retrain._infer_probe_spec``; an arch missing from that map does not raise until
  someone tries to retrain a probe of it, iterations into a run.

No GPU, no network, no probe fit and no extraction model: everything below runs on
small random tensors.
"""

from __future__ import annotations

import copy
import sys

import torch
from tuberlens.interfaces.probes import ProbeSpec, ProbeType
from tuberlens.probes.probe_factory import _ADAM_ARCHITECTURES
from tuberlens.probes.pytorch_modules import MultiMax

from agentic_redteam import retrain as R

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


@torch.no_grad()
def reference_forward(module: MultiMax, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Equation 9 written out per sample and per head, with no batching tricks."""
    out = []
    for b in range(x.shape[0]):
        total = 0.0
        valid = [j for j in range(x.shape[1]) if bool(mask[b, j])]
        for h in range(module.values.out_features):
            v_h = module.values.weight[h]
            total = total + max(
                float(v_h @ module.mlp(x[b, j])) for j in valid
            )
        out.append(total)
    return torch.tensor(out)


def main() -> int:
    torch.manual_seed(0)
    embed, seq, batch, heads = 16, 12, 4, 5
    module = MultiMax(embed, n_heads=heads, mlp_layers=2, mlp_width=8).eval()
    x = torch.randn(batch, seq, embed)
    mask = torch.ones(batch, seq, dtype=torch.bool)
    mask[1, 7:] = False
    mask[2, 3:] = False

    print("Equation 9")
    got = module(x, mask)
    want = reference_forward(module, x, mask)
    check(
        "matches a per-sample, per-head reference",
        torch.allclose(got, want, atol=1e-5),
        f"max abs diff {float((got - want).abs().max()):.2e}",
    )
    check("returns one score per sample", tuple(got.shape) == (batch,), str(tuple(got.shape)))

    print("\nPadding is excluded from the max")
    # Make the padded tail enormous. Under a correct mask the score cannot move;
    # under a missing one the tail becomes every head's argmax.
    x_spiked = x.clone()
    x_spiked[1, 7:] = 50.0
    x_spiked[2, 3:] = -50.0
    check(
        "a huge spike inside padding does not change the score",
        torch.allclose(module(x_spiked, mask), got, atol=1e-5),
        f"max abs diff {float((module(x_spiked, mask) - got).abs().max()):.2e}",
    )
    check(
        "the same spike DOES move the score once unmasked (the test has teeth)",
        not torch.allclose(
            module(x_spiked, torch.ones_like(mask))[1], got[1], atol=1e-3
        ),
    )
    check(
        "an all-padding row stays finite rather than NaN",
        bool(torch.isfinite(module(x, torch.zeros_like(mask))).all()),
    )

    print("\nEach head takes its own max (sum_h max_j, not max_j sum_h)")
    y = module.mlp(x)
    per_head = module.values(y).masked_fill(~mask.unsqueeze(-1), torch.finfo(y.dtype).min)
    sum_of_maxes = per_head.max(dim=1).values.sum(dim=-1)
    max_of_sums = per_head.sum(dim=-1).max(dim=1).values
    check("the module computes sum-of-maxes", torch.allclose(got, sum_of_maxes, atol=1e-5))
    check(
        "sum-of-maxes and max-of-sums actually differ here (the test has teeth)",
        not torch.allclose(sum_of_maxes, max_of_sums, atol=1e-3),
        f"max abs diff {float((sum_of_maxes - max_of_sums).abs().max()):.2e}",
    )
    check(
        "n_heads heads are allocated as one matrix",
        tuple(module.values.weight.shape) == (heads, 8) and module.values.bias is None,
        str(tuple(module.values.weight.shape)),
    )

    print("\nThe forward does not depend on train/eval mode")
    train_mode = copy.deepcopy(module).train()
    check(
        "module.train() and module.eval() score identically",
        torch.allclose(train_mode(x, mask), got, atol=1e-6),
    )
    softmax_module = MultiMax(embed, n_heads=heads, mlp_layers=2, mlp_width=8, agg="softmax")
    softmax_module.load_state_dict(module.state_dict())
    check(
        "agg='softmax' is reachable and differs from the max",
        not torch.allclose(softmax_module.eval()(x, mask), got, atol=1e-3),
    )
    check(
        "agg is a plain attribute, so stack_module_state ignores it",
        "agg" not in module.state_dict(),
    )

    print("\nFused (vmapped) and sequential scoring agree")
    members = [MultiMax(embed, n_heads=heads, mlp_layers=2, mlp_width=8).eval() for _ in range(3)]
    from torch.func import functional_call, stack_module_state

    params, buffers = stack_module_state(members)
    stateless = copy.deepcopy(members[0]).to("meta")
    fused = torch.vmap(
        lambda p, b, a, m: functional_call(stateless, (p, b), (a, m)),
        in_dims=(0, 0, None, None),
    )(params, buffers, x, mask)
    sequential = torch.stack([m(x, mask) for m in members])
    check(
        "vmap over stacked members matches member-by-member",
        torch.allclose(fused, sequential, atol=1e-5),
        f"max abs diff {float((fused - sequential).abs().max()):.2e}",
    )
    from tuberlens.probes.fused_ensemble import can_fuse

    check("can_fuse accepts a stack of MultiMax heads", can_fuse(members))

    print("\nPer-token trace")
    logits, token_scores, selected = module(x, mask, return_per_token=True)
    check("sequence logits are unchanged by the trace", torch.allclose(logits, got, atol=1e-6))
    check(
        "trace tensors are (batch, seq)",
        tuple(token_scores.shape) == (batch, seq) and tuple(selected.shape) == (batch, seq),
    )
    check(
        "selection counts sum to n_heads per sample",
        torch.allclose(selected.sum(dim=1), torch.full((batch,), float(heads)), atol=1e-5),
    )
    check(
        "no head selects a padded position",
        float(selected[~mask].abs().sum()) == 0.0,
    )

    print("\nWiring")
    check("ProbeType exposes 'multimax'", ProbeType("multimax") is ProbeType.multimax)
    check(
        "the factory routes it to an Adam-trained MultiMax",
        _ADAM_ARCHITECTURES.get(ProbeType.multimax) is MultiMax,
    )
    defaults = ProbeType.multimax.default_hyperparams
    check(
        "defaults follow the paper (10 heads, 2x100 MLP, hard max)",
        (defaults["n_heads"], defaults["mlp_layers"], defaults["mlp_width"], defaults["agg"])
        == (10, 2, 100, "max"),
        str({k: defaults[k] for k in ("n_heads", "mlp_layers", "mlp_width", "agg")}),
    )
    spec = R._coerce_probe_spec("multimax")
    check(
        "retrain._coerce_probe_spec accepts the name",
        isinstance(spec, ProbeSpec) and spec.name is ProbeType.multimax,
    )

    print("\nretrain._infer_probe_spec round-trip")

    class _Classifier:
        probe_architecture = MultiMax

    class _Probe:
        _classifier = _Classifier()
        hyper_params = {"n_heads": 4, "mlp_width": 32, "agg": "max"}

    inferred = R._infer_probe_spec(_Probe())
    check(
        "a pickled MultiMax probe infers back to ProbeType.multimax",
        inferred.name is ProbeType.multimax,
        str(inferred.name),
    )
    # ProbeSpec.model_post_init merges the arch's defaults under whatever was set,
    # so the round-trip preserves the explicit values and fills in the rest.
    check(
        "its explicitly-set hyperparameters survive the round-trip",
        all(inferred.hyperparams[k] == v for k, v in _Probe.hyper_params.items()),
        str({k: inferred.hyperparams[k] for k in _Probe.hyper_params}),
    )
    check(
        "unset hyperparameters fall back to the arch defaults",
        inferred.hyperparams["mlp_layers"] == 2 and inferred.hyperparams["patience"] == 50,
    )
    check(
        "it is not in DETERMINISTIC_ARCHS (seeds do diversify it)",
        "multimax" not in R.DETERMINISTIC_ARCHS,
    )
    check(
        "the repo default architecture is unchanged",
        R.DEFAULT_FRESH_PROBE_ARCH == "linear_then_softmax",
        R.DEFAULT_FRESH_PROBE_ARCH,
    )

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
