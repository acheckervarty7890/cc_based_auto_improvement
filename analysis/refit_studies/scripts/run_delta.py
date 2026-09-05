"""Retrain each finished arm twice: once on the red-team samples from the iterations
whose retrain RAISED mean eval AUROC, once on those whose retrain LOWERED it.

Same recipe as the arm's own retrains — same base data, same preprocessing
(filter + contrastive, arm's own cache), same dev set, same judge-confidence gate,
same seed, ensemble size inherited from the arm's iteration-0 probe — so the only
thing that differs from the arm's real trajectory is WHICH successes are trained on.
"""
import json, sys, time, shutil, pathlib, traceback
sys.path.insert(0, "/workspace/probe_auto_improvement/src")
from agentic_redteam.retrain import retrain_probe
from agentic_redteam.evaluation import evaluate_probe
from agentic_redteam.config import PreprocessingConfig

REPO = pathlib.Path("/workspace/probe_auto_improvement")
S = pathlib.Path("/tmp/claude-1000/-workspace-probe-auto-improvement/642f7272-1606-4ff4-99f1-d98f1bd3bfc4/scratchpad")
D = S / "deltaexp"
OUT = D / "out"; OUT.mkdir(exist_ok=True)

CONCEPT = {
    "hu_harm": dict(
        dev=REPO / "dev_samples/hu_ha",
        eval=REPO / "eval_sets/hu_ha",
        cache=REPO / "results_hu_harm_gemma27b_batch_ablation/base_activations",
        ecache=REPO / "results_hu_harm_gemma27b_batch_ablation/eval_activations",
        splits=["eval_ai_dilemmas", "eval_ant_hh", "eval_balanced_refusal", "eval_daily_dilemmas"]),
    "instructions": dict(
        dev=S / "six/dev",           # six splits — this branch's dirs still carry oig_omission
        eval=S / "six/eval",
        cache=REPO / "results_instructions_gemma27b_shared/base_activations",
        ecache=REPO / "results_instructions_gemma27b_shared/eval_activations",
        splits=["anthropic_harmless_refusal", "bbq_substitution", "hc_context_drift",
                "hc_contradiction", "mm_substitution", "oig_context_drift"]),
}
PREP = PreprocessingConfig(provider="openrouter", model="openai/gpt-5.1", max_concurrent=50,
                           max_tokens=2048, filter_percentile=0.8, assistant_centric=True)

def write_group(arm_dir, iters, dest):
    n = 0
    with open(dest, "w") as f:
        for et in ("fp", "fn"):
            for line in open(arm_dir / f"{et}.jsonl"):
                r = json.loads(line)
                if r.get("success") and r.get("iteration") in iters:
                    f.write(line if line.endswith("\n") else line + "\n"); n += 1
    return n

def run(arm, grp, iters, cfg):
    tag = f'{arm["key"]}_{grp}'
    res_path = OUT / f"{tag}.json"
    if res_path.exists():
        print(f"[skip] {tag} already done", flush=True); return json.load(open(res_path))
    d = D / arm["key"]
    jl = OUT / f"{tag}_successes.jsonl"
    n = write_group(d, set(iters), jl)
    cc = OUT / f"{tag}_contrastive.jsonl"
    if not cc.exists() and (d / "contrastive_cache.jsonl").exists():
        shutil.copy(d / "contrastive_cache.jsonl", cc)   # copy: never mutate the arm's own cache
    probe_out = OUT / f"{tag}.pkl"
    t0 = time.time()
    print(f"\n===== {tag}: {n} successes from iterations {iters} =====", flush=True)
    retrain_probe(
        jsonl_path=[jl], base_probe_path=d / "probe_iter0.pkl",
        base_training_data_path=REPO / arm["base"], new_probe_path=probe_out,
        preprocessing=PREP, contrastive_cache_path=cc, min_judge_confidence=7,
        dev_data_path=cfg["dev"], seed=42, ensemble_size=None,
        base_activation_cache_dir=cfg["cache"],
        combine_consecutive_messages=True, convert_tool_to_assistant=True, verbose=True)
    df = evaluate_probe(str(probe_out), str(cfg["eval"]), str(cfg["ecache"]), splits=None,
                        max_samples=None, seed=42,
                        combine_consecutive_messages=True, convert_tool_to_assistant=True)
    p = df.set_index("dataset")["auroc"]
    sp = p[cfg["splits"]]
    res = dict(arm=arm["key"], group=grp, iters=iters, n_successes=n,
               mean=round(float(sp.mean()), 4),
               splits={k: round(float(v), 4) for k, v in sp.items()},
               minutes=round((time.time() - t0) / 60, 1))
    if arm["concept"] == "instructions":
        res["exref"] = round(float(sp.drop("anthropic_harmless_refusal").mean()), 4)
    json.dump(res, open(res_path, "w"), indent=1)
    print("RESULT " + json.dumps(res), flush=True)
    return res

if __name__ == "__main__":
    arms = json.load(open(D / "arms.json"))
    groups = {g["key"]: g for g in json.load(open(D / "groups.json"))}
    only = sys.argv[1:] or None
    for arm in arms:
        if only and arm["key"] not in only: continue
        g = groups[arm["key"]]; cfg = CONCEPT[arm["concept"]]
        for grp, iters in (("up", g["inc"]), ("down", g["dec"])):
            try:
                run(arm, grp, iters, cfg)
            except Exception:
                traceback.print_exc()
                print(f"[FAILED] {arm['key']}_{grp}", flush=True)
