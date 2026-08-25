"""Regenerate D5's 16 contrastive pairs under GENERATION_PROMPT_VERSION 2 (minimal edit).

Same 16 red-team finds, same generator (openai/gpt-5.1 via OpenRouter), same knobs the run
used — only the prompt template differs. Writes the new pairs and prints the length
comparison the change was made to fix.
"""
import json, sys
from pathlib import Path
REPO = Path("/workspace/cc_based_auto_improvement"); sys.path.insert(0, str(REPO/"src"))

from agentic_redteam.config import load_config
from agentic_redteam.preprocessing import generate_contrastive_dataset
from agentic_redteam.token_budget import TokenBudget

CFG = REPO/"configs/gptoss120b_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5.md"
SRC = REPO/"probes/instructions_gemma27b_scopecheck_exp24_tellattacker_iter5/redteam_postprocessed_iter5.jsonl"
OUT = REPO/"results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5/pairs_promptv2.jsonl"

cfg = load_config(CFG); pp = cfg.preprocessing
pos, neg = cfg.probe.pos_class_label, cfg.probe.neg_class_label

rows = [json.loads(l) for l in SRC.open()]
n = len(rows)//2
finds, old_pairs = rows[:n], rows[n:]
assert all(r["label"] == "negative" for r in finds)
print(f"{len(finds)} finds; generator {pp.model} via {pp.provider}; prompt version 2", flush=True)

# generate_contrastive_dataset expects the loader's shape: inputs + a `labels` column
# carrying the human-readable class label, not the canonical enum.
records = [{"inputs": r["inputs"], "labels": neg, "ids": r["id"]} for r in finds]

budget = TokenBudget(cfg.probe.model, pp.max_sample_tokens,
                     combine_consecutive_messages=cfg.eval.combine_consecutive_messages,
                     convert_tool_to_assistant=cfg.eval.convert_tool_to_assistant)

out = generate_contrastive_dataset(
    records, pos_class_label=pos, neg_class_label=neg,
    provider=pp.provider, model=pp.model,
    max_concurrent=pp.max_concurrent, max_tokens=pp.max_tokens,
    max_retries=pp.max_generation_retries,
    cache_path=None,                       # fresh generation, no cache
    assistant_centric=pp.assistant_centric,
    concept_description=pp.concept_description,
    label_guidance=pp.label_guidance,
    eval_data_description=cfg.eval.data_description,
    token_budget=budget)

gen = [r for r in out if r["labels"] == pos]
print(f"generated {len(gen)} of {len(finds)}", flush=True)
with OUT.open("w") as f:
    for i, r in enumerate(gen):
        f.write(json.dumps({"id": f"redteam-v2-{i}", "inputs": r["inputs"], "label": "positive"}, ensure_ascii=False)+"\n")

def asst(ms): return next((m["content"] for m in ms if m["role"] == "assistant"), "") or ""
def user(ms): return next((m["content"] for m in ms if m["role"] == "user"), "") or ""
print(f"\n{'':4s}{'find':>7s}{'v1 gen':>9s}{'v2 gen':>9s}{'v1 ratio':>10s}{'v2 ratio':>10s}  user turn preserved?")
print("-"*78)
import statistics
r1, r2, keep = [], [], 0
for i,(fd, o, g) in enumerate(zip(finds, old_pairs, gen)):
    lf, lo, lg = len(asst(fd["inputs"])), len(asst(o["inputs"])), len(asst(g["inputs"]))
    a, b = lf/max(1,lo), lf/max(1,lg)
    same = user(fd["inputs"]) == user(g["inputs"]); keep += same
    r1.append(a); r2.append(b)
    print(f"{i:<4d}{lf:7d}{lo:9d}{lg:9d}{a:10.2f}{b:10.2f}  {'yes' if same else 'NO'}")
print(f"\nmean find/gen assistant-length ratio — v1 {statistics.mean(r1):.2f}  ->  v2 {statistics.mean(r2):.2f}   (1.00 = same length)")
print(f"user turn preserved: v2 {keep}/{len(gen)}")
print(f"wrote {OUT}")
