"""Summarize the 90% / 80% resampling grids over all 22 arms.

Reads every analysis/refit_studies/subsample/*.json (the first pass's 16 arms) plus
the six +attacker arms' grids from the scratchpad, and each arm's own full-run
endpoint from its comparison CSV, and prints: the per-arm draw spread at each
fraction, the eight-draw mean against the arm's reported endpoint, and whether each
within-attacker pair's ordering survives resampling.
"""
import csv, glob, io, json, pathlib, statistics as st, subprocess, sys

REPO = pathlib.Path("/workspace/probe_auto_improvement")
S = pathlib.Path("/tmp/claude-1000/-workspace-probe-auto-improvement/a15cb740-dcf6-4f45-bb48-92b7ed548985/scratchpad")
SUB = [REPO / "analysis/refit_studies/subsample", S / "deltaexp/sub"]

HU = ["eval_ai_dilemmas", "eval_ant_hh", "eval_balanced_refusal", "eval_daily_dilemmas"]
INS = ["anthropic_harmless_refusal", "bbq_substitution", "hc_context_drift",
       "hc_contradiction", "mm_substitution", "oig_context_drift"]

IBR = "origin/experiment_instruction_last"
# key -> (branch, comparison csv path)
CSV = {
 "e25_memo": ("origin/experiment25_gptoss_base_cloud", "results_hu_harm_gemma27b_gptoss120b_gptossbase_itermemo150/gptossbase_itermemo150_comparison.csv"),
 "e25_evaldesc": ("origin/experiment25_gptoss_base_cloud", "results_hu_harm_gemma27b_gptoss120b_gptossbase_evaldesc/gptossbase_evaldesc_comparison.csv"),
 "e26_memo": ("origin/experiment26_deepseek_cloud", "results_hu_harm_gemma27b_deepseekv4pro_dsbase_itermemo150/dsbase_itermemo150_comparison.csv"),
 "e26_evaldesc": ("origin/experiment26_deepseek_cloud", "results_hu_harm_gemma27b_deepseekv4pro_dsbase_evaldesc/dsbase_evaldesc_comparison.csv"),
 "hh_l70_memo": ("local", "results_hu_harm_gemma27b_llama70b_l70base_itermemo150/l70base_itermemo150_comparison.csv"),
 "hh_l70_edesc": ("local", "results_hu_harm_gemma27b_llama70b_l70base_evaldesc/l70base_evaldesc_comparison.csv"),
 "hh_l70_edatt": ("local", "results_hu_harm_gemma27b_llama70b_l70base_evaldesc_attacker/l70base_evaldesc_attacker_comparison.csv"),
 "hh_nm_memo": ("local", "results_hu_harm_gemma27b_nemotron_nmbase_itermemo150/nmbase_itermemo150_comparison.csv"),
 "hh_nm_edesc": ("local", "results_hu_harm_gemma27b_nemotron_nmbase_evaldesc/nmbase_evaldesc_comparison.csv"),
 "hh_nm_edatt": ("local", "results_hu_harm_gemma27b_nemotron_nmbase_evaldesc_attacker/nmbase_evaldesc_attacker_comparison.csv"),
 "i_ll_memo": (IBR, "results_instructions_gemma27b_llama70b_l70base_itermemo150/llama70b_l70base_itermemo150_comparison.csv"),
 "i_ll_edesc": (IBR, "results_instructions_gemma27b_llama70b_l70base_evaldesc/llama70b_l70base_evaldesc_comparison.csv"),
 "i_ll_edatt": (IBR, "results_instructions_gemma27b_llama70b_l70base_evaldesc_attacker/llama70b_l70base_evaldesc_attacker_comparison.csv"),
 "i_nm_memo": (IBR, "results_instructions_gemma27b_nemotron_nmbase_itermemo150/nemotron_nmbase_itermemo150_comparison.csv"),
 "i_nm_edesc": (IBR, "results_instructions_gemma27b_nemotron_nmbase_evaldesc/nemotron_nmbase_evaldesc_comparison.csv"),
 "i_nm_edatt": (IBR, "results_instructions_gemma27b_nemotron_nmbase_evaldesc_attacker/nemotron_nmbase_evaldesc_attacker_comparison.csv"),
 "i_go_memo": (IBR, "results_instructions_gemma27b_gptoss_gobase_itermemo150/gptoss_gobase_itermemo150_comparison.csv"),
 "i_go_edesc": (IBR, "results_instructions_gemma27b_gptoss_gobase_evaldesc/gptoss_gobase_evaldesc_comparison.csv"),
 "i_go_edatt": (IBR, "results_instructions_gemma27b_gptoss_gobase_evaldesc_attacker/gptoss_gobase_evaldesc_attacker_comparison.csv"),
 "i_ds_memo": (IBR, "results_instructions_gemma27b_deepseekv4pro_dsbase_itermemo150/deepseekv4pro_dsbase_itermemo150_comparison.csv"),
 "i_ds_edesc": (IBR, "results_instructions_gemma27b_deepseekv4pro_dsbase_evaldesc/deepseekv4pro_dsbase_evaldesc_comparison.csv"),
 "i_ds_edatt": (IBR, "results_instructions_gemma27b_deepseekv4pro_dsbase_evaldesc_attacker/deepseekv4pro_dsbase_evaldesc_attacker_comparison.csv"),
}
TRIPLES = [("hu_harm", "llama70b", "hh_l70_memo", "hh_l70_edesc", "hh_l70_edatt"),
           ("hu_harm", "nemotron", "hh_nm_memo", "hh_nm_edesc", "hh_nm_edatt"),
           ("hu_harm", "gpt-oss (E25)", "e25_memo", "e25_evaldesc", None),
           ("hu_harm", "deepseek (E26)", "e26_memo", "e26_evaldesc", None),
           ("instr", "llama70b", "i_ll_memo", "i_ll_edesc", "i_ll_edatt"),
           ("instr", "nemotron", "i_nm_memo", "i_nm_edesc", "i_nm_edatt"),
           ("instr", "gpt-oss", "i_go_memo", "i_go_edesc", "i_go_edatt"),
           ("instr", "deepseek", "i_ds_memo", "i_ds_edesc", "i_ds_edatt")]

def endpoint(key):
    branch, path = CSV[key]
    if branch == "local":
        txt = (REPO / path).read_text()
    else:
        txt = subprocess.run(["git", "show", f"{branch}:{path}"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout
    splits = HU if key.startswith(("e2", "hh_")) else INS
    by = {}
    for r in csv.DictReader(io.StringIO(txt)):
        if r["dataset"] in splits:
            by.setdefault(r["round"], {})[r["dataset"]] = float(r["auroc"])
    full = {k: v for k, v in by.items() if len(v) == len(splits)}
    last = max(full, key=lambda s: int(s.replace("iter", "")))
    return last, sum(full[last].values()) / len(splits)

def load():
    g = {}
    for d in SUB:
        for f in glob.glob(str(d / "*_f[89]0_d[0-7].json")):
            r = json.load(open(f))
            g.setdefault((r["arm"], r["frac"]), []).append(r)
    return g

if __name__ == "__main__":
    g = load()
    ends = {k: endpoint(k) for k in CSV}
    print(f'{"arm":14s} {"end@":>6s} {"endpt":>7s} | '
          + " | ".join(f'{"f"+str(f):>5s} {"mean":>7s} {"sd":>7s} {"min":>7s} {"max":>7s} {"vs end":>7s}'
                       for f in (90, 80)))
    rows = {}
    for key in CSV:
        it, e = ends[key]
        line = f'{key:14s} {it:>6s} {e:7.4f} |'
        for frac in (0.9, 0.8):
            v = [r["mean"] for r in g.get((key, frac), [])]
            if len(v) < 2:
                line += f'  n={len(v):<2d} {"":>7s} {"":>7s} {"":>7s} {"":>7s} {"":>7s} |'; continue
            m, sd = st.mean(v), st.stdev(v)
            rows[(key, frac)] = (m, sd, len(v))
            line += f'  n={len(v):<2d} {m:7.4f} {sd:7.4f} {min(v):7.4f} {max(v):7.4f} {m-e:+7.4f} |'
        print(line)

    print("\n=== does each contrast's sign survive resampling? "
          "(full-run gap, then the eight-draw gap at each fraction) ===")
    for concept, att, memo, edesc, edatt in TRIPLES:
        print(f'\n{concept}/{att}')
        pairs = [("memo -> +desc(judge)", memo, edesc)]
        if edatt:
            pairs += [("+desc(judge) -> +attacker", edesc, edatt),
                      ("memo -> +attacker", memo, edatt)]
        for name, a, b in pairs:
            fa, fb = ends[a][1], ends[b][1]
            out = f'  {name:26s} full {fb-fa:+7.4f}'
            for frac in (0.9, 0.8):
                ra, rb = rows.get((a, frac)), rows.get((b, frac))
                if not ra or not rb:
                    out += f'   f{int(frac*100)} {"n/a":>8s}'; continue
                d = rb[0] - ra[0]
                pooled = (ra[1] ** 2 / ra[2] + rb[1] ** 2 / rb[2]) ** 0.5
                flag = "same" if (d > 0) == (fb - fa > 0) else "FLIP"
                out += f'   f{int(frac*100)} {d:+7.4f} ({d/pooled:+5.1f} se, {flag})'
            print(out)
