"""Which of the 256 subsample fits would change under cache-only contrastive pairs?

A fit whose kept records ALL have a cached pair is byte-identical in the two modes, so the
result already on disk can be reused. This pass reconstructs each draw (same seed), runs the
real filter + contrastive path with generation disabled, and records how many records came
back unpaired. CPU only: no GPU, no network.
"""
import io, json, os, random, sys, contextlib, pathlib
os.environ["AGENTIC_REDTEAM_CONTRASTIVE_CACHE_ONLY"] = "1"
sys.path.insert(0, "/workspace/probe_auto_improvement/src")
from agentic_redteam.retrain import _build_redteam_dataset
from agentic_redteam.persistence import JsonlStore
from agentic_redteam.config import PreprocessingConfig

D = pathlib.Path("/tmp/claude-1000/-workspace-probe-auto-improvement/642f7272-1606-4ff4-99f1-d98f1bd3bfc4/scratchpad/deltaexp")
OUT = D / "sub"
PREP = PreprocessingConfig(provider="openrouter", model="openai/gpt-5.1", max_concurrent=50,
                           max_tokens=2048, filter_percentile=0.8, assistant_centric=True)
LABELS = {"hu_harm": ("harmful_to_human", "not_harmful_to_human"),
          "instructions": ("assistant_follows_the_instruction", "assistant_does_not_follow_the_instruction")}

arms = json.load(open(D / "arms.json"))
report = {}
for arm in arms:
    d = D / arm["key"]
    rows = [l for et in ("fp", "fn") for l in open(d / f"{et}.jsonl") if json.loads(l).get("success")]
    pos, neg = LABELS[arm["concept"]]
    for frac in (0.9, 0.8):
        for draw in range(8):
            tag = f'{arm["key"]}_f{int(frac*100)}_d{draw}'
            k = max(1, round(len(rows) * frac))
            keep = random.Random(f'{arm["key"]}|{frac}|{draw}').sample(rows, k)
            jl = OUT / f"{tag}.jsonl"
            if not jl.exists(): jl.write_text("".join(keep))
            recs = [r for r in JsonlStore(path=jl).iter_successes() if r.judge_confidence >= 7]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _build_redteam_dataset(recs, pos, neg, PREP, d / "contrastive_cache.jsonl", False,
                                       model_name="google/gemma-3-27b-it",
                                       combine_consecutive_messages=True, convert_tool_to_assistant=True)
            txt = buf.getvalue()
            unpaired = 0
            for line in txt.splitlines():
                if "cache-only:" in line:
                    unpaired = int(line.split("cache-only:")[1].split()[0])
            report[tag] = dict(kept=k, unpaired=unpaired, done=(OUT / f"{tag}.json").exists())
            print(f"{tag:26s} kept={k:4d} unpaired={unpaired:4d} done={report[tag]['done']}", flush=True)
json.dump(report, open(D / "prepass.json", "w"), indent=1)
done = [t for t, r in report.items() if r["done"]]
reusable = [t for t in done if report[t]["unpaired"] == 0]
print(f"\ncompleted fits: {len(done)}  of which identical under cache-only: {len(reusable)}")
print(f"to run: {256 - len(reusable)}")
