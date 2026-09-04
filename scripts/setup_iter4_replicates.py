"""Build four independent replicates of ARM 7 (nemotron / high-stakes / memo-only)
ITERATION 4 — the cycle whose retrain moved the probe 0.76858 -> 0.81088.

Each replicate reproduces the exact state the ORIGINAL iteration 4 started from and
then re-runs that one cycle. Everything that can be pinned is pinned; the only thing
that varies between replicates is LLM sampling (nemotron's batches, gpt-5.1's
judgements, gpt-5.1's contrastive generations for newly-found successes).

WHY EACH PIECE IS COPIED / TRUNCATED
  probe_iter4.pkl        the input probe. Copied ALONE, so _latest_probe_iteration
                         returns 4 and the CLI starts at iteration 4 (probe_iter{i}
                         feeds iteration i).
  *_probing_{fp,fn}.jsonl truncated to iteration <= 3. The retrain at the end of
                         iteration 4 trains on base ∪ ALL successes in the store, so
                         a fresh store would train on ~43 rows instead of ~240 and
                         would not be the same experiment. Truncating also restores
                         the near-dup guard's and the dedup set's exact contents.
                         PRIVATE PER REPLICATE — sharing one store would let one
                         replicate's submissions dedup away another's.
  *.iteration_memos.jsonl truncated to iteration <= 3, so IterationMemoStore
                         .prior_text(4) returns the same hand-off memo the original
                         iteration 4 was given, byte for byte. This is the "same
                         cross-iteration memo" requirement.
  *.rounds_done.jsonl    truncated to iteration <= 3. The CLI passes
  *.summaries.jsonl      resume=(i == start_iter), so at iteration 4 both stores are
                         consulted; an iteration-4 row would make the replicate SKIP
                         rounds and restore a rolling memo it never wrote.
  contrastive_cache.jsonl copied whole. Content-keyed on the source conversation, so
                         the iterations 0-3 successes (re-preprocessed at every
                         retrain) reuse the original's pairs instead of paying for
                         ~200 fresh generations per replicate.
  NO redteam_done_iter4_*.marker — those would skip the rotation entirely.

The rolling ROUND memo is deliberately NOT pinned: it is built inside the iteration
from that iteration's own attempts, so it cannot be fixed without fixing the
attempts. It starts empty in every replicate exactly as it did in the original.
"""
import json, shutil, sys
from pathlib import Path

ROOT = Path("/workspace/probe_auto_improvement")
SRC_PROBES = ROOT / "probes/hs_gemma27b_nemotron_nemobase_itermemo150"
SRC_RES = ROOT / "results_hs_gemma27b_nemotron_nemobase_itermemo150"
SRC_CFG = ROOT / "configs/nemotron_hs_gemma27b_nemobase_itermemo150.md"
STEM = "nemobase_itermemo150_probing"
ITER = 4                      # the cycle being replayed
KEEP_UPTO = ITER - 1          # state as of the START of iteration 4
N_REPS = 4

def truncate(src: Path, dst: Path) -> tuple[int, int]:
    kept = total = 0
    with src.open() as fh, dst.open("w") as out:
        for line in fh:
            if not line.strip():
                continue
            total += 1
            if json.loads(line).get("iteration", -1) <= KEEP_UPTO:
                out.write(line)
                kept += 1
    return kept, total

cfg_text = SRC_CFG.read_text()
assert "jsonl_path: ../results_hs_gemma27b_nemotron_nemobase_itermemo150/nemobase_itermemo150_probing.jsonl" in cfg_text
assert not [l for l in cfg_text.splitlines() if l.strip().startswith("data_description:")], "arm 7 must have no eval.data_description"

for n in range(1, N_REPS + 1):
    tag = f"hs_nm_iter4_rep{n}"
    pdir = ROOT / "probes" / tag
    rdir = ROOT / "results" / tag if False else ROOT / f"results_{tag}"
    pdir.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(SRC_PROBES / f"probe_iter{ITER}.pkl", pdir / f"probe_iter{ITER}.pkl")
    shutil.copy2(SRC_PROBES / "contrastive_cache.jsonl", pdir / "contrastive_cache.jsonl")

    report = []
    for et in ("fp", "fn"):
        for suffix in ("", ".iteration_memos", ".rounds_done", ".summaries"):
            src = SRC_RES / f"{STEM}_{et}{suffix}.jsonl"
            dst = rdir / f"rep{n}_probing_{et}{suffix}.jsonl"
            k, t = truncate(src, dst)
            report.append(f"{et}{suffix or '.jsonl'}: {k}/{t}")

    cfg = cfg_text.replace(
        "jsonl_path: ../results_hs_gemma27b_nemotron_nemobase_itermemo150/nemobase_itermemo150_probing.jsonl",
        f"jsonl_path: ../results_{tag}/rep{n}_probing.jsonl",
    ).replace(
        "run_id: nemotron_hs_gemma27b_nemobase_itermemo150",
        f"run_id: {tag}",
    ).replace(
        "comparison_csv: ../results_hs_gemma27b_nemotron_nemobase_itermemo150/nemobase_itermemo150_comparison.csv",
        f"comparison_csv: ../results_{tag}/rep{n}_comparison.csv",
    )
    assert cfg.count(f"results_{tag}") == 2, cfg.count(f"results_{tag}")
    header = (
        f"---\n"
        f"# REPLICATE {n} of 4 — ARM 7 (nemotron, memo-only, high-stakes) ITERATION {ITER} ONLY.\n"
        f"#\n"
        f"# Identical to configs/{SRC_CFG.name} except the three output paths below\n"
        f"# (jsonl_path, run_id, comparison_csv), which are per-replicate so the four runs do not\n"
        f"# share a dedup store, a memo sidecar or a comparison CSV. Every knob that affects what\n"
        f"# the attacker, judge, preprocessor or fit does is byte-identical to arm 7's.\n"
        f"#\n"
        f"# Run it as:  --iterations {ITER+1}  with probes/{tag}/probe_iter{ITER}.pkl in place, so\n"
        f"# _latest_probe_iteration returns {ITER}, the loop is range({ITER}, {ITER+1}) = one cycle,\n"
        f"# and the run red-teams probe_iter{ITER} (eval mean 0.76858) and writes probe_iter{ITER+1}.\n"
        f"# The original wrote 0.81088 there; the four replicates measure how reproducible that is.\n"
    )
    cfg = header + cfg.split("---\n", 1)[1]
    (ROOT / "configs" / f"nemotron_hs_iter4_rep{n}.md").write_text(cfg)
    print(f"rep{n}: {pdir.name} + {rdir.name}  |  " + "  ".join(report))
