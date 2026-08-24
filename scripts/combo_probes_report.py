#!/usr/bin/env python
"""Regenerate reports/combo_probes/ from whatever scripts/combo_probes.py has produced.

Safe to call repeatedly while the run is still going: a concept without an eval CSV is
reported as pending rather than blocking the report. Everything the report claims is
computed here from the eval CSVs — nothing is typed in — so it stays true as concepts
land. A hand-written narrative in reports/combo_probes/FINDINGS.md is appended verbatim
when that file exists, so it survives a regeneration.
"""

from __future__ import annotations

import itertools
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

RESULTS = REPO_ROOT / "results_combos"
OUT = REPO_ROOT / "reports" / "combo_probes"
CONCEPTS = ["hu_ha", "instructions", "highstakes"]
ORDER = ["llama8b", "llama70b", "dsv4pro", "nemotron550b"]
SHORT = {"llama8b": "l8b", "llama70b": "l70b", "dsv4pro": "dsv", "nemotron550b": "nem"}
VAL_MODES = ["dev", "split"]
N_PROBES = 30  # 15 combos x 2 validation modes


def _combos() -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for k in range(1, len(ORDER) + 1):
        out.extend(itertools.combinations(ORDER, k))
    return out


def _name(combo) -> str:
    return "+".join(SHORT[g] for g in combo)


def _load():
    """{concept: DataFrame of per-split rows}, mean rows dropped."""
    import pandas as pd

    out = {}
    for c in CONCEPTS:
        csv = RESULTS / c / "eval_results.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        out[c] = df[df.dataset != "mean"].copy()
    return out


def _v(df):
    """{(combo name, val_mode): mean AUROC over that concept's eval splits}."""
    g = df.groupby(["combo", "val_mode"])["auroc"].mean()
    return {k: float(v) for k, v in g.items()}


def _ceiling() -> dict:
    import pandas as pd

    csv = REPO_ROOT / "reports" / "cross_concept_ceiling" / "cross_concept_ceiling.csv"
    if not csv.exists():
        return {}
    df = pd.read_csv(csv)
    df = df[(df.scope == "concept") & df.arm.str.startswith("within/")]
    return {r["name"]: float(r["auroc"]) for _i, r in df.iterrows()}


def _marginals(v: dict) -> dict:
    """{(generator, val_mode): mean AUROC change from adding g to a non-empty pool}.

    Averaged over the seven non-empty subsets of the other three generators, so every
    generator is scored on the same seven baselines. The empty pool is excluded because
    there is no probe for it — a v(empty) would have to be asserted (chance? the noise
    floor?) rather than measured, and the choice would drive the singleton term.
    """
    out = {}
    for g in ORDER:
        others = [x for x in ORDER if x != g]
        subsets = [s for k in range(1, 4) for s in itertools.combinations(others, k)]
        for vm in VAL_MODES:
            deltas = []
            for s in subsets:
                base, plus = _name(s), _name(tuple(x for x in ORDER if x in set(s) | {g}))
                if (base, vm) in v and (plus, vm) in v:
                    deltas.append(v[(plus, vm)] - v[(base, vm)])
            if deltas:
                out[(g, vm)] = sum(deltas) / len(deltas)
    return out


def _fit_times() -> dict:
    """{(concept, val_mode): [seconds]} parsed from the run log.

    The two validation sources cost very differently — the dev set is scored every
    epoch — and that gap is the only real price difference in this experiment, since
    every probe here is a single head over cached activations.
    """
    import re

    log = REPO_ROOT / "logs" / "combo_probes.log"
    if not log.is_file():
        return {}
    head = re.compile(r"^=== (\w+) \| (\S+) \| val=(\w+) ===")
    fit = re.compile(r"^\s+fit in (\d+)s")
    out: dict = {}
    cur = None
    for line in log.read_text(errors="ignore").splitlines():
        m = head.match(line)
        if m:
            cur = (m.group(1), m.group(3))
            continue
        m = fit.match(line)
        if m and cur:
            out.setdefault(cur, []).append(int(m.group(1)))
            cur = None
    return out


def _singleton_crosscheck() -> list[str]:
    """Compare the singleton combos against the original per-generator runs.

    A singleton combo's training file is byte-identical to that generator's own cut, so
    its `single` probe should reproduce `scripts/concept_probes.py`'s exactly. Any drift
    here means the content-addressed cache slicing put different rows in the blob.
    """
    import pandas as pd

    rows = []
    for c in CONCEPTS:
        csv = RESULTS / c / "eval_results.csv"
        if not csv.exists():
            continue
        mine = pd.read_csv(csv)
        mine = mine[mine.dataset != "mean"]
        for g in ORDER:
            ref_csv = REPO_ROOT / f"results_{g}" / c / "eval_results.csv"
            if not ref_csv.exists():
                continue
            ref = pd.read_csv(ref_csv)
            ref = ref[(ref.dataset != "mean") & (ref.config == "single")]
            for vm in VAL_MODES:
                a = mine[(mine.combo == SHORT[g]) & (mine.val_mode == vm)]
                b = ref[ref.val_mode == vm]
                if a.empty or b.empty:
                    continue
                j = a.merge(b, on="dataset", suffixes=("_combo", "_ref"))
                if j.empty:
                    continue
                rows.append((c, g, vm, float((j.auroc_combo - j.auroc_ref).abs().max())))
    return rows


def main() -> int:
    import pandas as pd

    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    data = _load()
    ceiling = _ceiling()

    lines = [
        "# Pooled generators — one single probe per subset of the four synthetic cuts",
        "",
        f"_Generated {now}._",
        "",
        "## What is being measured",
        "",
        "Four attacker models each wrote ~50 balanced two-turn conversations per concept.",
        "[`concept_probes_summary.md`](../concept_probes_summary.md) asked what each is worth",
        "**alone**. This asks what they are worth **pooled**: every non-empty subset of",
        "",
        "&nbsp;&nbsp;&nbsp;&nbsp;`llama8b` (`l8b`) · `llama70b` (`l70b`) · `dsv4pro` (`dsv`) · `nemotron550b` (`nem`)",
        "",
        "is concatenated into one training set, giving 4 singles (~50 rows), 6 pairs (~100),",
        "4 triples (~150) and the all-four pool (~200). Only **single** probes are fitted — no",
        "ensembles — crossed with the two validation sources:",
        "",
        "| axis | values |",
        "| --- | --- |",
        "| training data | the 15 non-empty subsets of the four generators |",
        "| validation | `dev` (the concept's `dev_samples/` set is the whole validation set; every training row trains) · `split` (a 0.2 content-deterministic slice is held out instead) |",
        "",
        f"{N_PROBES} probes per concept, {N_PROBES * len(CONCEPTS)} in total. `google/gemma-3-27b-it`",
        "layer 32, `linear_then_softmax`, seed 42, scored on full eval splits off precomputed",
        "activations. Every cell below is mean AUROC over that concept's eval splits.",
        "",
        "A standalone reading version of these results — same numbers, laid out for the eye —",
        "is [`pooling-generators.html`](pooling-generators.html). It is a hand-written",
        "**snapshot**, not a generated file: if a rerun changes the numbers below, that page",
        "does not follow.",
        "",
    ]

    pending = [c for c in CONCEPTS if c not in data]
    if pending:
        lines += [f"_In progress — no eval results yet for: {', '.join(pending)}._", ""]

    # ---- headline table: combo x (concept, val_mode) -----------------------------
    if data:
        frames = []
        for c, df in data.items():
            t = df.groupby(["combo", "val_mode"])["auroc"].mean().rename(c)
            frames.append(t)
        wide = pd.concat(frames, axis=1)
        order = [(_name(k), vm) for k in _combos() for vm in VAL_MODES]
        wide = wide.reindex([o for o in order if o in wide.index])
        lines += ["## Mean AUROC by pool and validation source", "", "```",
                  wide.round(3).to_string(), "```", ""]

        # ---- pooling curve --------------------------------------------------------
        curve = []
        for c, df in data.items():
            t = df.groupby(["n_generators", "val_mode"])["auroc"].mean().rename(c)
            curve.append(t)
        cur = pd.concat(curve, axis=1)
        lines += ["## Does pooling help? Mean AUROC by number of generators", "", "```",
                  cur.round(3).to_string(), "```", ""]

        best_rows = []
        for c, df in data.items():
            v = _v(df)
            singles = {k: x for k, x in v.items() if "+" not in k[0]}
            best_single = max(singles.items(), key=lambda kv: kv[1])
            best_any = max(v.items(), key=lambda kv: kv[1])
            allfour = v.get((_name(tuple(ORDER)), "dev")), v.get((_name(tuple(ORDER)), "split"))
            allfour_best = max([x for x in allfour if x is not None], default=float("nan"))
            best_rows.append(
                (c, f"{best_single[0][0]} / {best_single[0][1]}", best_single[1],
                 f"{best_any[0][0]} / {best_any[0][1]}", best_any[1], allfour_best,
                 ceiling.get(c, float("nan")))
            )
        lines += ["## Best pool vs best single generator", "",
                  "| concept | best single | AUROC | best pool of any size | AUROC | all four | ceiling |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for c, bs, bsv, ba, bav, af, ceil in best_rows:
            lines.append(f"| {c} | `{bs}` | {bsv:.3f} | `{ba}` | {bav:.3f} | "
                         f"{af:.3f} | {ceil:.3f} |")
        lines += ["",
                  "`ceiling` is the same concept's within-concept ceiling from",
                  "[`cross_concept_ceiling/`](../cross_concept_ceiling/REPORT.md) — a probe of this",
                  "family trained on eval-distribution data. It is a pooled-across-splits AUROC on a",
                  "balanced 100/class subsample, not a mean of per-split AUROCs, so read it as a",
                  "reference point rather than a directly comparable cell.",
                  ""]

        # ---- marginal contribution ------------------------------------------------
        lines += ["## What each generator adds to a pool that lacks it", "",
                  "Mean AUROC change from adding one generator to a pool, averaged over the seven",
                  "non-empty subsets of the other three — so every generator is scored against the",
                  "same seven baselines.", "", "```"]
        marg = {}
        for c, df in data.items():
            m = _marginals(_v(df))
            for (g, vm), x in m.items():
                marg[(g, vm, c)] = x
        if marg:
            mdf = pd.Series(marg).unstack()
            mdf.index.names = ["generator", "val_mode"]
            mdf = mdf.reindex([(g, vm) for g in ORDER for vm in VAL_MODES])
            lines += [mdf.round(3).to_string()]
        lines += ["```", ""]

        # ---- pool vs its own members ---------------------------------------------
        lines += ["## Is a pool more than the average of its parts?", "",
                  "For every pool of two or more, its AUROC minus the **mean** of its members'",
                  "own singleton AUROCs, and minus the **best** of them. If pooling only averaged",
                  "its inputs the first column would sit at zero; if it added coverage the second",
                  "would be positive.", "",
                  "| concept | val_mode | pool − mean(members) | pool > mean | pool − best(member) | pool > best |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for c, df in data.items():
            v = _v(df)
            for vm in VAL_MODES:
                vs_mean, vs_best = [], []
                for combo in _combos():
                    if len(combo) < 2 or (_name(combo), vm) not in v:
                        continue
                    parts = [v[(SHORT[g], vm)] for g in combo if (SHORT[g], vm) in v]
                    if len(parts) != len(combo):
                        continue
                    vs_mean.append(v[(_name(combo), vm)] - sum(parts) / len(parts))
                    vs_best.append(v[(_name(combo), vm)] - max(parts))
                if not vs_mean:
                    continue
                lines.append(
                    f"| {c} | {vm} | {sum(vs_mean)/len(vs_mean):+.3f} | "
                    f"{sum(1 for x in vs_mean if x > 0)}/{len(vs_mean)} | "
                    f"{sum(vs_best)/len(vs_best):+.3f} | "
                    f"{sum(1 for x in vs_best if x > 0)}/{len(vs_best)} |")
        lines.append("")

        # ---- does a solo score predict a marginal contribution? -------------------
        corr_rows = []
        for c, df in data.items():
            v, m = _v(df), _marginals(_v(df))
            solo = [v[(SHORT[g], vm)] for g in ORDER for vm in VAL_MODES]
            marg = [m[(g, vm)] for g in ORDER for vm in VAL_MODES]
            corr_rows.append((c, pd.Series(solo).corr(pd.Series(marg)),
                              pd.Series(solo).corr(pd.Series(marg), method="spearman")))
        if corr_rows:
            allsolo, allmarg = [], []
            for c, df in data.items():
                v, m = _v(df), _marginals(_v(df))
                allsolo += [v[(SHORT[g], vm)] for g in ORDER for vm in VAL_MODES]
                allmarg += [m[(g, vm)] for g in ORDER for vm in VAL_MODES]
            lines += ["## Does a generator's solo score predict what it adds to a pool?", "",
                      "Correlation between the eight solo AUROCs above (4 generators x 2 validation",
                      "modes) and the eight marginal contributions.", "",
                      "| scope | pearson | spearman |", "| --- | --- | --- |"]
            for c, pe, sp in corr_rows:
                lines.append(f"| {c} | {pe:+.3f} | {sp:+.3f} |")
            lines.append(f"| **all concepts pooled** | "
                         f"{pd.Series(allsolo).corr(pd.Series(allmarg)):+.3f} | "
                         f"{pd.Series(allsolo).corr(pd.Series(allmarg), method='spearman'):+.3f} |")
            lines.append("")

        # ---- all four vs the best single, split by split --------------------------
        lines += ["## All four vs the best single generator, split by split", "",
                  "The pooled means above average over splits, which can hide a pool that wins on",
                  "one split and loses on another. `val=split` shown; the best single is the one",
                  "with the highest mean AUROC in that concept.", ""]
        for c, df in data.items():
            piv = df[df.val_mode == "split"].pivot_table(index="combo", columns="dataset",
                                                         values="auroc")
            singles = [SHORT[g] for g in ORDER if SHORT[g] in piv.index]
            if not singles:
                continue
            best = piv.loc[singles].mean(axis=1).idxmax()
            allf = _name(tuple(ORDER))
            if allf not in piv.index:
                continue
            d = piv.loc[allf] - piv.loc[best]
            lines += [f"**{c}** — best single `{best}`, all four `{allf}`; "
                      f"all four wins on {int((d > 0).sum())}/{len(d)} splits.", "", "```",
                      piv.loc[[best, allf]].round(3).to_string(), "```", ""]

        # ---- dev vs split ---------------------------------------------------------
        dv = []
        for c, df in data.items():
            p = df.groupby(["combo", "val_mode"])["auroc"].mean().unstack()
            if set(VAL_MODES) <= set(p.columns):
                dv.append((c, float((p["dev"] - p["split"]).mean()),
                           float((p["dev"] - p["split"]).abs().mean()),
                           int((p["dev"] > p["split"]).sum()), len(p)))
        if dv:
            lines += ["## dev-set validation vs a 0.2 split", "",
                      "| concept | mean dev − split | mean abs diff | dev wins |",
                      "| --- | --- | --- | --- |"]
            for c, mean_d, abs_d, wins, n in dv:
                lines.append(f"| {c} | {mean_d:+.3f} | {abs_d:.3f} | {wins}/{n} |")
            lines.append("")

        # ---- cost -----------------------------------------------------------------
        ft = _fit_times()
        if ft:
            lines += ["## What the two validation sources cost", "",
                      "Every probe here is one `linear_then_softmax` head over cached activations,",
                      "so the only meaningful cost difference is that the `dev` arm scores the whole",
                      "`dev_samples/` set every epoch while `split` scores a ~20-row slice.", "",
                      "| concept | dev set | dev fit (median) | split fit (median) |",
                      "| --- | --- | --- | --- |"]
            dev_rows = {c: sum(1 for f in sorted((REPO_ROOT / "dev_samples" / c).glob("*.jsonl"))
                               for l in f.open() if l.strip()) for c in CONCEPTS}
            for c in CONCEPTS:
                d, sp = sorted(ft.get((c, "dev"), [])), sorted(ft.get((c, "split"), []))
                if not d or not sp:
                    continue
                lines.append(f"| {c} | {dev_rows[c]} rows | {d[len(d)//2]}s | {sp[len(sp)//2]}s |")
            lines.append("")

        # ---- per-concept per-split detail ----------------------------------------
        for c in CONCEPTS:
            if c not in data:
                continue
            shutil.copy2(RESULTS / c / "eval_results.csv", OUT / f"{c}_eval_results.csv")
            piv = data[c].pivot_table(index=["combo", "val_mode"], columns="dataset",
                                      values="auroc")
            piv["MEAN"] = piv.mean(axis=1)
            piv = piv.reindex([o for o in order if o in piv.index])
            lines += [f"## {c} — AUROC per eval split", "", "```", piv.round(3).to_string(),
                      "```", ""]

        # ---- crosscheck -----------------------------------------------------------
        cross = _singleton_crosscheck()
        if cross:
            worst = max(x[3] for x in cross)
            lines += ["## Cross-check against the per-generator runs", "",
                      "A singleton pool's training file is byte-identical to that generator's own",
                      "cut, so its probe must reproduce the `single` arm of",
                      "[`concept_probes_summary.md`](../concept_probes_summary.md) exactly. Largest",
                      f"absolute per-split AUROC difference over the {len(cross)} comparable arms: "
                      f"**{worst:.2e}**.", ""]

    findings = OUT / "FINDINGS.md"
    if findings.exists():
        lines += [findings.read_text().rstrip(), ""]

    lines += [
        "## Reproducing",
        "",
        "```bash",
        ".venv_claude/bin/python scripts/combo_probes.py --phase all",
        ".venv_claude/bin/python scripts/combo_probes_report.py",
        "```",
        "",
        "No model is loaded at any point. `prepare` assembles each pool's train/val activation",
        "cache by addressing rows in the per-generator master blobs *by conversation content*,",
        "which is sound because `stable_train_test_split` is content-deterministic — a",
        "conversation falls on the same side of the train/val line in every pool it appears in.",
        "The eval reads each split's activations once and scores all of that concept's probes",
        "against it, rather than reloading 46 GB of high-stakes activations per probe.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT/'REPORT.md'} ({', '.join(data) or 'no concepts complete yet'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
