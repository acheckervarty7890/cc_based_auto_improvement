"""The 90% / 80% resampling study, extended to the six +attacker-shown arms.

Same design and same code path as run_subsample.py — for every arm, 8 independent
random subsets holding 90% of that arm's successes, refit and re-evaluated, then the
same at 80%; everything else is the arm's own recipe (its 50-row base, its probe_iter0,
its preprocessing + contrastive cache, judge-confidence 7, its dev set, fit seed 42,
ensemble size inherited), so the only thing varying within an arm is which successes
are kept. The draw seed is the same string, `{arm}|{frac}|{draw}`, so a draw here is
built exactly as its memo/evaldesc siblings' draws were.

The six arms are the ones the first pass could not cover because they had not been run
yet: `attacker.show_eval_data_description: true` on both human-harm attackers and all
four instruction attackers. experiment25 / experiment26 gained no new arm, so their
four entries stay as the first pass left them.
"""
import json, sys, time, random, shutil, pathlib, traceback
sys.path.insert(0, "/workspace/probe_auto_improvement/src")
from agentic_redteam.retrain import retrain_probe
from agentic_redteam.evaluation import evaluate_probe
from agentic_redteam.config import PreprocessingConfig
from agentic_redteam.circuit_breaker import OpenRouterOutageError

REPO = pathlib.Path("/workspace/probe_auto_improvement")
S = pathlib.Path("/tmp/claude-1000/-workspace-probe-auto-improvement/a15cb740-dcf6-4f45-bb48-92b7ed548985/scratchpad")
D = S / "deltaexp"; OUT = D / "sub"; OUT.mkdir(exist_ok=True)

CONCEPT = {
    "hu_harm": dict(dev=REPO/"dev_samples/hu_ha", eval=REPO/"eval_sets/hu_ha",
                    cache=REPO/"results_hu_harm_gemma27b_batch_ablation/base_activations",
                    ecache=REPO/"results_hu_harm_gemma27b_batch_ablation/eval_activations",
                    splits=["eval_ai_dilemmas","eval_ant_hh","eval_balanced_refusal","eval_daily_dilemmas"]),
    "instructions": dict(dev=S/"six/dev", eval=S/"six/eval",
                    cache=REPO/"results_instructions_gemma27b_shared/base_activations",
                    ecache=REPO/"results_instructions_gemma27b_shared/eval_activations",
                    splits=["anthropic_harmless_refusal","bbq_substitution","hc_context_drift",
                            "hc_contradiction","mm_substitution","oig_context_drift"]),
}
PREP = PreprocessingConfig(provider="openrouter", model="openai/gpt-5.1", max_concurrent=50,
                           max_tokens=2048, filter_percentile=0.8, assistant_centric=True)

def successes(arm_dir):
    rows = []
    for et in ("fp", "fn"):
        for line in open(arm_dir / f"{et}.jsonl"):
            if json.loads(line).get("success"):
                rows.append(line if line.endswith("\n") else line + "\n")
    return rows

def run(arm, frac, draw, cfg):
    tag = f'{arm["key"]}_f{int(frac*100)}_d{draw}'
    res_path = OUT / f"{tag}.json"
    if res_path.exists():
        return json.load(open(res_path))
    d = D / arm["key"]
    rows = successes(d)
    k = max(1, round(len(rows) * frac))
    rnd = random.Random(f'{arm["key"]}|{frac}|{draw}')      # reproducible, per (arm, fraction, draw)
    keep = rnd.sample(rows, k)
    jl = OUT / f"{tag}.jsonl"; jl.write_text("".join(keep))
    cc = OUT / f"{tag}_contrastive.jsonl"
    if not cc.exists() and (d / "contrastive_cache.jsonl").exists():
        shutil.copy(d / "contrastive_cache.jsonl", cc)      # copy: never mutate the arm's own cache
    probe_out = OUT / f"{tag}.pkl"
    t0 = time.time()
    print(f"\n===== {tag}: {k}/{len(rows)} successes =====", flush=True)
    retrain_probe(jsonl_path=[jl], base_probe_path=d / "probe_iter0.pkl",
                  base_training_data_path=arm["base"], new_probe_path=probe_out,
                  preprocessing=PREP, contrastive_cache_path=cc, min_judge_confidence=7,
                  dev_data_path=cfg["dev"], seed=42, ensemble_size=None,
        base_activation_cache_dir=cfg["cache"],
        combine_consecutive_messages=True, convert_tool_to_assistant=True,
        # The +eval-desc arms generated their contrastive pairs under a prompt that carries
        # eval.data_description, and that text is folded into the contrastive cache key. Omit it
        # and every lookup misses: 80% of an arm's records come back unpaired and the pairs get
        # REGENERATED under a different prompt than the arm used. Pass the arm's own text and the
        # hit rate is 98-100%. NOTE the two human-harm arms here carry the REWRITTEN description
        # (d793fe5d), not the one hh_l70_edesc / hh_nm_edesc ran under — which is what their own
        # retrains used, so it is what their cache is keyed on.
        eval_data_description=arm.get("desc", ""), verbose=True)
    df = evaluate_probe(str(probe_out), str(cfg["eval"]), str(cfg["ecache"]), splits=None,
                        max_samples=None, seed=42,
                        combine_consecutive_messages=True, convert_tool_to_assistant=True)
    p = df.set_index("dataset")["auroc"]; sp = p[cfg["splits"]]
    res = dict(arm=arm["key"], concept=arm["concept"], frac=frac, draw=draw, n=k, n_all=len(rows),
               mean=round(float(sp.mean()), 4),
               splits={kk: round(float(v), 4) for kk, v in sp.items()},
               minutes=round((time.time() - t0) / 60, 1))
    if arm["concept"] == "instructions":
        res["exref"] = round(float(sp.drop("anthropic_harmless_refusal").mean()), 4)
    json.dump(res, open(res_path, "w"), indent=1)
    print("RESULT " + json.dumps(res), flush=True)
    probe_out.unlink(missing_ok=True)      # 96 ensemble pickles is a lot of disk for no reuse
    return res

if __name__ == "__main__":
    arms = json.load(open(D / "arms2.json"))
    only = sys.argv[1:] or None
    for frac in (0.9, 0.8):                # part (i) first, then part (ii)
        for arm in arms:
            if only and arm["key"] not in only: continue
            cfg = CONCEPT[arm["concept"]]
            for draw in range(8):
                try:
                    run(arm, frac, draw, cfg)
                except OpenRouterOutageError as exc:
                    # Never swallow this one (see CLAUDE.md): the contrastive step needs
                    # OpenRouter, so once the breaker trips every remaining fit would be a
                    # no-op failure. Stop and leave the finished draws on disk to resume from.
                    print(f"\n[ABORT] OpenRouter unusable: {exc}", flush=True)
                    print(f"[ABORT] stopped at {arm['key']} {frac} d{draw}; "
                          f"finished draws are in {OUT} and are skipped on re-run.", flush=True)
                    sys.exit(3)
                except Exception:
                    traceback.print_exc(); print(f"[FAILED] {arm['key']} {frac} d{draw}", flush=True)
