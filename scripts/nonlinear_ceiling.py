"""Is there anything a *non-linear* readout of layer 32 can extract that a linear one cannot?

This bounds the architecture sweep before it is read. ``why_last_iteration_adds_nothing.md``
§2 fits a linear probe on **each eval split itself** (5-fold CV, mean-pooled layer-32
activations) to get what a linear readout of these activations can do on that split:

    eval_ai_dilemmas 0.7997 | eval_ant_hh 0.9599 | eval_balanced_refusal 0.9977 |
    eval_daily_dilemmas 0.9050

against 0.64-0.91 achieved by probes trained on the actual training distribution. The
information is already there and linearly available; what fails is transfer from the
training data to the eval distribution. If that reading is right, a non-linear head has
little *extra* to find, and the same measurement run with non-linear models should show
a ceiling that barely moves.

That is the point of this script: it is the cheap falsification test for the whole Tier-1
half of the sweep. If the non-linear ceiling *does* clear the linear one — especially on
``eval_ant_hh``, where two thirds of the 31-row hard core lives — then capacity is worth
pursuing and the MLP heads deserve tuning. If it does not, an MLP head that fails to help
is confirming a bound rather than being under-tuned, and no amount of hyperparameter
search will change it.

Two measurements, and the second exists because the first cannot answer the real question:

**1. Ceiling (in-domain).** 5-fold CV on the split itself, so train and test come from
   the same distribution. Upper bound on what is extractable, not a claim about transfer.
   Protocol matched to ``why_iter3_null.section_ceiling`` exactly — same pooling, same
   folds, same best-over-C-grid selection — so the linear column here reproduces §2's
   numbers and the non-linear columns are read against them on the same scale. (Selecting
   the grid point on the CV score is optimistically biased; that is deliberate for a
   *ceiling*, and applies equally to every model family, but it means these numbers are
   upper bounds rather than honest estimates.)

**2. Transfer.** Fit on the actual training set — base plus the arm's iteration-3
   red-team dump — and score the eval splits, which is what a deployed probe does.
   A model family can raise the in-domain ceiling and still transfer worse, and under
   distribution shift that is the common case rather than the exception; only this
   half speaks to what the pipeline would gain.

Both use mean-pooled activations, so the comparison isolates the *readout* with pooling
held fixed. Pooling is the other axis, and ``arch_sweep.py`` is where it is varied.

Usage:
    .venv_claude/bin/python scripts/nonlinear_ceiling.py
    .venv_claude/bin/python scripts/nonlinear_ceiling.py --arm gptoss120b --quick
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A
import why_iter3_null as W

#: Rows pooled per chunk. ``why_iter3_null._pool`` casts a whole blob to fp32 at once,
#: which for the 400-row eval_balanced_refusal split is 400 x 1024 x 5376 x 4 B = 8.8 GB
#: — enough, on this box, to OOM-kill a concurrently running sweep (it did). Pooling is a
#: per-row reduction, so chunking is exact and caps the spike at ~0.7 GB.
POOL_CHUNK = 32

OUT_DIR = A.REPO / "results_hu_harm_gemma27b_batch_ablation/arch_sweep"
HELDOUT_DIR = A.REPO / "results_hu_harm_gemma27b_batch_ablation/heldout_v3_vs_v2"
NOBASE_DIR = A.REPO / "results_hu_harm_gemma27b_batch_ablation/heldout_v3_vs_v2_nobase"

#: Matched to why_iter3_null.C_GRID so the linear column reproduces §2.
C_GRID = W.C_GRID
#: gamma='scale' is 1/(d * Var(X)); at d=5376 the multipliers bracket it by ~an order of
#: magnitude either way, which is where an RBF on standardised high-dimensional data
#: either finds curvature or degenerates (too small -> linear, too large -> memorises).
SVM_GRID = [
    {"C": c, "gamma": g}
    for c in (1.0, 10.0, 100.0)
    for g in ("scale", 1e-5, 1e-4)
]
GB_GRID = [
    {"max_iter": 200, "learning_rate": 0.1, "max_depth": 3},
    {"max_iter": 200, "learning_rate": 0.1, "max_depth": None},
    {"max_iter": 400, "learning_rate": 0.05, "max_depth": 3},
]
QUICK_SVM_GRID = SVM_GRID[:3]
QUICK_GB_GRID = GB_GRID[:1]


# --- features -----------------------------------------------------------------------


def pool_chunked(acts: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Mean over unpadded tokens, in fp32, ``POOL_CHUNK`` rows at a time.

    Numerically identical to ``why_iter3_null._pool`` (a mean over the token axis is a
    per-row reduction, so the chunking cannot change a value) but with a bounded peak —
    see :data:`POOL_CHUNK`.
    """
    out = np.empty((acts.shape[0], acts.shape[-1]), dtype=np.float32)
    for i in range(0, acts.shape[0], POOL_CHUNK):
        m = mask[i : i + POOL_CHUNK].bool()
        h = acts[i : i + POOL_CHUNK].float() * m[..., None]
        out[i : i + POOL_CHUNK] = (
            h.sum(-2) / m.sum(-1, keepdim=True).clamp(min=1)
        ).numpy()
        del h, m
    return out


def eval_features(split: str) -> tuple[np.ndarray, np.ndarray]:
    """``(X, y)`` for one eval split — ``why_iter3_null.eval_features``, chunked."""
    from tuberlens.interfaces.dataset import LabelledDataset

    ds = LabelledDataset.load_from(
        A.EVAL_DATASET_DIR / f"{split}.jsonl",
        pos_class_label="harmful_to_human",
        neg_class_label="not_harmful_to_human",
        combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
        convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
    )
    y = ds.labels_torch().float().cpu().numpy()
    blob = torch.load(
        A.EVAL_ACTIVATIONS_DIR / f"{split}-acts_full.pt", weights_only=False, mmap=True
    )
    x = pool_chunked(blob["activations"], blob["attention_mask"])
    del blob
    gc.collect()
    return x, y


def redteam_features(arm: str, iteration: int = 3):
    """``(X, y, dataset)`` for one arm's postprocessed dump, one blob at a time."""
    ds = A.load_redteam_dataset(arm, iteration)
    xs = []
    for msgs in ds.inputs:
        blob = torch.load(A.redteam_blob_path(msgs), weights_only=False)
        xs.append(pool_chunked(blob["activations"], blob["attention_mask"])[0])
        del blob
    y = np.array(
        [1.0 if v == "positive" else 0.0 for v in ds.other_fields["labels"]],
        dtype=np.float32,
    )
    return np.stack(xs), y, ds


def base_features():
    """``((X_train, y_train), (X_val, y_val))`` for the base training data."""
    from tuberlens.interfaces.dataset import LabelledDataset

    ds = LabelledDataset.load_from(
        A.BASE_TRAINING_DATA,
        pos_class_label="harmful_to_human",
        neg_class_label="not_harmful_to_human",
        combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
        convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
    )
    train, val = A.split_sides(ds)
    out = []
    for part, path in zip((train, val), A.base_blob_paths()):
        blob = torch.load(path, weights_only=False, mmap=True)
        out.append((
            pool_chunked(blob["activations"], blob["attention_mask"]),
            part.labels_torch().float().cpu().numpy(),
        ))
        del blob
        gc.collect()
    return out[0], out[1]


# --- model families -----------------------------------------------------------------


def _fit_score(kind: str, params: dict, x_tr, y_tr, x_te):
    """Fit one model and return decision scores on ``x_te``.

    Standardisation is applied for the two distance/margin-based families and skipped for
    the trees, which are invariant to monotone per-feature transforms — scaling them would
    only cost time.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if kind == "trees":
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(random_state=0, **params).fit(x_tr, y_tr)
        return clf.decision_function(x_te)

    scaler = StandardScaler().fit(x_tr)
    x_tr_s, x_te_s = scaler.transform(x_tr), scaler.transform(x_te)
    if kind == "linear":
        clf = LogisticRegression(max_iter=4000, **params).fit(x_tr_s, y_tr)
    elif kind == "rbf_svm":
        from sklearn.svm import SVC

        clf = SVC(kernel="rbf", **params).fit(x_tr_s, y_tr)
    else:
        raise ValueError(kind)
    return clf.decision_function(x_te_s)


def _grids(quick: bool) -> dict[str, list[dict]]:
    return {
        "linear": [{"C": c} for c in C_GRID],
        "rbf_svm": QUICK_SVM_GRID if quick else SVM_GRID,
        "trees": QUICK_GB_GRID if quick else GB_GRID,
    }


# --- 1. in-domain ceiling -----------------------------------------------------------


def prompt_groups(split: str) -> np.ndarray:
    """Group id per row, keyed on the conversation's **user turns**.

    Three of the four hu_ha splits are prompt-paired: the same user prompt appears once
    with a harmful assistant reply and once with a non-harmful one, so the label lives
    entirely in the assistant turn. Measured over the splits as loaded, 100% of
    `ai_dilemmas`, `balanced_refusal` and `daily_dilemmas` rows sit in a group carrying
    **both** labels; `eval_ant_hh` is unpaired (132 of 133 groups are singletons).
    """
    import hashlib
    from collections import defaultdict

    from tuberlens.interfaces.dataset import LabelledDataset

    ds = LabelledDataset.load_from(
        A.EVAL_DATASET_DIR / f"{split}.jsonl",
        pos_class_label="harmful_to_human",
        neg_class_label="not_harmful_to_human",
        combine_consecutive_messages=A.COMBINE_CONSECUTIVE_MESSAGES,
        convert_tool_to_assistant=A.CONVERT_TOOL_TO_ASSISTANT,
    )
    by_prompt: dict[str, list[int]] = defaultdict(list)
    for i, messages in enumerate(ds.inputs):
        user = "".join(m.content for m in messages if m.role == "user")
        by_prompt[hashlib.sha256(user.encode("utf-8")).hexdigest()].append(i)
    gid = np.empty(len(ds), dtype=int)
    for k, (_h, idxs) in enumerate(sorted(by_prompt.items())):
        for i in idxs:
            gid[i] = k
    return gid


def ceiling(eval_x: dict, eval_y: dict, quick: bool, group_by_prompt: bool = False) -> dict:
    """Best 5-fold-CV AUROC per split per model family, fitting on the split itself.

    Default folds are ``StratifiedKFold(5, shuffle=True, random_state=0)``, identical to
    ``why_iter3_null.section_ceiling``, so every family sees the same partition and the
    linear column is directly comparable to the published §2 table.

    ``group_by_prompt`` switches to ``GroupKFold`` keyed on :func:`prompt_groups`, which
    is what a high-capacity family needs. Ungrouped, a row's opposite-label partner is in
    the training fold ~80% of the time, and since the user prompt dominates a mean-pooled
    activation, a model able to key on it predicts the partner's label and is inverted
    rather than merely wrong: gradient-boosted trees measured **0.148** on
    `eval_ai_dilemmas` this way (orientation verified separately — trees score 1.0000 on
    synthetic separable data, so this is the data, not a sign bug). The linear column is
    largely unaffected, since L2 logistic regression cannot memorise a prompt that
    sharply, which is why the ungrouped numbers still reproduce §2 exactly.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    grids = _grids(quick)
    out: dict = {}
    for split in A.EVAL_SPLITS:
        x, y = eval_x[split], eval_y[split]
        gid = prompt_groups(split) if group_by_prompt else None
        out[split] = {}
        for kind, grid in grids.items():
            best, best_params = -1.0, None
            t0 = time.time()
            for params in grid:
                s = np.zeros(len(y))
                folds = (
                    GroupKFold(5).split(x, y, groups=gid)
                    if group_by_prompt
                    else StratifiedKFold(5, shuffle=True, random_state=0).split(x, y)
                )
                for tr, te in folds:
                    s[te] = _fit_score(kind, params, x[tr], y[tr], x[te])
                auc = roc_auc_score(y, s)
                if auc > best:
                    best, best_params = auc, params
            out[split][kind] = {
                "auroc": float(best),
                "params": best_params,
                "seconds": time.time() - t0,
            }
            print(f"  {split:24s} {kind:9s} {best:.4f}  ({best_params}, "
                  f"{time.time() - t0:.0f}s)", flush=True)
    for kind in grids:
        out.setdefault("mean", {})[kind] = {
            "auroc": float(np.mean([out[s][kind]["auroc"] for s in A.EVAL_SPLITS]))
        }
    return out


# --- 2. transfer --------------------------------------------------------------------


def _core_positions() -> dict[str, set[tuple[str, int]]]:
    """The eight-family hard core from the heldout sweep, as ``(split, idx)`` pairs.

    Positional rather than content-keyed: 41 of the 866 eval rows are duplicates of
    another row, nearly all across splits, so a content join over-counts the core (see
    ``arch_sweep.load_hard_core``).
    """
    def four_way(d: Path, rule: str) -> set[tuple[str, int]]:
        cores = []
        for arm in sorted(A.ARMS):
            path = d / f"{arm}_eval_tags.jsonl"
            if not path.exists():
                return set()
            rows = [json.loads(line) for line in path.open(encoding="utf-8")]
            cores.append({
                (r["split"], r["idx_in_split"]) for r in rows
                if r[f"{rule}_v2_tag"] == "incorrect"
                and r[f"{rule}_v3new_tag"] == "incorrect"
            })
        return cores[0] & cores[1] if len(cores) == 2 else set()

    return {
        rule: four_way(HELDOUT_DIR, rule) & four_way(NOBASE_DIR, rule)
        for rule in ("raw", "balanced")
    }


def transfer(arm: str, eval_x: dict, eval_y: dict, quick: bool) -> dict:
    """Fit on the arm's real training set, score the eval splits.

    Model selection uses the pipeline's own held-out validation side (the
    content-deterministic split), never the eval splits — otherwise the "transfer"
    number would be fitted to the thing it claims to transfer to.
    """
    from sklearn.metrics import roc_auc_score

    print(f"\n--- transfer: {arm} ---", flush=True)
    t0 = time.time()
    rt_x, rt_y, rt_ds = redteam_features(arm, 3)
    is_val = np.array([A.is_val(m) for m in rt_ds.inputs], dtype=bool)
    (base_tr_x, base_tr_y), (base_val_x, base_val_y) = base_features()
    x_tr = np.concatenate([base_tr_x, rt_x[~is_val]])
    y_tr = np.concatenate([base_tr_y, rt_y[~is_val]])
    x_val = np.concatenate([base_val_x, rt_x[is_val]])
    y_val = np.concatenate([base_val_y, rt_y[is_val]])
    print(f"  train {len(y_tr)} rows, val {len(y_val)} rows "
          f"({time.time() - t0:.0f}s to load)", flush=True)

    cores = _core_positions()
    grids = _grids(quick)
    out: dict = {}
    for kind, grid in grids.items():
        best_auc, best_params, best_scores = -1.0, None, None
        t1 = time.time()
        for params in grid:
            s_val = _fit_score(kind, params, x_tr, y_tr, x_val)
            try:
                auc = roc_auc_score(y_val, s_val)
            except ValueError:
                continue
            if auc > best_auc:
                best_auc, best_params = auc, params
        # Refit the selected configuration once and score every eval split with it.
        best_scores = {
            sp: _fit_score(kind, best_params, x_tr, y_tr, eval_x[sp])
            for sp in A.EVAL_SPLITS
        }
        per_split = {
            sp: float(roc_auc_score(eval_y[sp], best_scores[sp]))
            for sp in A.EVAL_SPLITS
        }
        entry = {
            "params": best_params,
            "val_auroc": float(best_auc),
            "per_split_auroc": per_split,
            "mean_auroc": float(np.mean(list(per_split.values()))),
            "seconds": time.time() - t1,
        }
        # Hard-core recovery under the balanced rule, matching arch_sweep's readout:
        # predict the top half of each split, since every hu_ha split is exactly 50/50.
        for rule in ("raw", "balanced"):
            n_core = n_ok = 0
            for sp in A.EVAL_SPLITS:
                s, y = best_scores[sp], eval_y[sp]
                if rule == "raw":
                    pred = s >= 0
                else:
                    pred = np.zeros(len(s), dtype=bool)
                    pred[np.argsort(-s, kind="stable")[: int((y > 0.5).sum())]] = True
                correct = pred == (y > 0.5)
                for split_name, idx in cores[rule]:
                    if split_name == sp:
                        n_core += 1
                        n_ok += bool(correct[idx])
            entry[f"core_{rule}"] = {"n": n_core, "recovered": n_ok}
        out[kind] = entry
        print(f"  {kind:9s} val={best_auc:.4f} eval mean={entry['mean_auroc']:.4f} "
              f"core(bal) {entry['core_balanced']['recovered']}/"
              f"{entry['core_balanced']['n']}  ({best_params}, {time.time() - t1:.0f}s)",
              flush=True)
    del rt_x, rt_ds
    gc.collect()
    return out


# --- report -------------------------------------------------------------------------


def print_report(res: dict) -> None:
    print("\n\n########## 1. in-domain ceiling (5-fold CV on each split itself) ##########")
    print("  what ANY readout of mean-pooled layer 32 can extract, per split")
    c = res["ceiling"]
    kinds = ["linear", "rbf_svm", "trees"]
    print(f"\n{'split':24s} " + " ".join(f"{k:>10s}" for k in kinds)
          + f" {'best - linear':>15s}")
    for sp in A.EVAL_SPLITS + ["mean"]:
        vals = [c[sp][k]["auroc"] for k in kinds]
        delta = max(vals[1:]) - vals[0]
        print(f"{sp:24s} " + " ".join(f"{v:>10.4f}" for v in vals)
              + f" {delta:>+15.4f}")

    print("\n\n########## 2. transfer (fit on the real training set) ##########")
    print("  what each family achieves on the eval splits from the training distribution")
    for arm, t in res["transfer"].items():
        print(f"\n--- {arm} ---")
        print(f"{'family':10s} {'val':>8s} {'eval mean':>10s} "
              + " ".join(f"{sp.replace('eval_', ''):>18s}" for sp in A.EVAL_SPLITS)
              + f" {'core recovered':>15s}")
        for kind in kinds:
            e = t[kind]
            core = f"{e['core_balanced']['recovered']}/{e['core_balanced']['n']}"
            print(f"{kind:10s} {e['val_auroc']:>8.4f} {e['mean_auroc']:>10.4f} "
                  + " ".join(f"{e['per_split_auroc'][sp]:>18.4f}"
                             for sp in A.EVAL_SPLITS)
                  + f" {core:>15s}")

    print("\n\n########## reading this ##########")
    ant = c["eval_ant_hh"]
    lift = max(ant["rbf_svm"]["auroc"], ant["trees"]["auroc"]) - ant["linear"]["auroc"]
    print(f"  eval_ant_hh holds two thirds of the 31-row hard core. Its LINEAR ceiling is")
    print(f"  {ant['linear']['auroc']:.4f}; the best non-linear ceiling is "
          f"{max(ant['rbf_svm']['auroc'], ant['trees']['auroc']):.4f} ({lift:+.4f}).")
    if lift < 0.01:
        print("  -> capacity is NOT the binding constraint on that split. An MLP head that")
        print("     fails to help in arch_sweep is confirming this bound, not under-tuned.")
    else:
        print("  -> there IS non-linear headroom here; the MLP heads deserve tuning, and")
        print("     arch_sweep's --sensitivity grid is the place to look.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", choices=sorted(A.ARMS), default=sorted(A.ARMS))
    ap.add_argument("--quick", action="store_true",
                    help="smaller hyperparameter grids (smoke test, not a result)")
    ap.add_argument("--skip-transfer", action="store_true")
    ap.add_argument("--group-by-prompt", action="store_true",
                    help="GroupKFold on the user-turn hash instead of StratifiedKFold. "
                         "Required for any high-capacity family: three of the four splits "
                         "are prompt-paired with opposite labels, which inverts an "
                         "ungrouped CV. Ungrouped is the default so the linear column "
                         "stays comparable to why_iter3_null section 2.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        suffix = "_grouped" if args.group_by_prompt else ""
        args.out = OUT_DIR / f"nonlinear_ceiling{suffix}.json"

    print("loading eval features (mean-pooled layer 32) ...", flush=True)
    eval_x, eval_y = {}, {}
    for split in A.EVAL_SPLITS:
        eval_x[split], eval_y[split] = eval_features(split)
        print(f"  {split:24s} {eval_x[split].shape}", flush=True)

    print("\n--- in-domain ceiling ---", flush=True)
    res = {"quick": args.quick, "group_by_prompt": args.group_by_prompt,
           "ceiling": ceiling(eval_x, eval_y, args.quick, args.group_by_prompt),
           "transfer": {}}

    if not args.skip_transfer:
        for arm in args.arm:
            res["transfer"][arm] = transfer(arm, eval_x, eval_y, args.quick)

    print_report(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
