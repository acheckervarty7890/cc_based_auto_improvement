"""Ask the run's own attacker for one SHORTER version of each of its 33 finds.

Motivation is measured, not aesthetic: the red-team finds average 1175 characters against the
oig_omission eval split's 668, so they sit at roughly 1.8x the length of the data the probe is
scored on (their generated partners, at 1450, are 2.2x). This asks whether the same case, told
more briefly, is worth more as training data than the same case told at the attacker's natural
length.

What must survive: the number of separately answerable parts, which one the reply drops, and
its position. What goes: verbosity, padding, elaboration. Model, provider and the 1024-token
cap are the run's own.
"""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO = Path("/workspace/cc_based_auto_improvement"); sys.path.insert(0, str(REPO/"src"))
from agentic_redteam.config import load_config
from agentic_redteam.openrouter_client import make_sync_client
from agentic_redteam.token_budget import TokenBudget

CFG = REPO/"configs/gptoss120b_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3.md"
SRC = REPO/"probes/instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3/redteam_postprocessed_iter5.jsonl"
OUT = REPO/"results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3/shortened.jsonl"
RETRIES = 3

cfg = load_config(CFG)
neg = cfg.probe.neg_class_label
model = cfg.attacker.models[0].name
budget = TokenBudget(cfg.probe.model, cfg.preprocessing.max_sample_tokens,
                     combine_consecutive_messages=cfg.eval.combine_consecutive_messages,
                     convert_tool_to_assistant=cfg.eval.convert_tool_to_assistant)
cli = make_sync_client()

SYS = f"""You compress conversations for a classifier's training set.

You are given one conversation labelled "{neg}". Produce ONE shorter version of it.

Aim for roughly HALF the original length. Keep exactly:
- the same number of separately answerable parts in the user's request, asked in the same order
- the same part left unanswered by the assistant, in the same position
- the same topic, the same names, the same specifics — this is a compression, not a rewrite

Cut only what carries no weight: throat-clearing, restatement, elaboration, examples, hedging,
pleasantries, and detail inside an answer that is already complete. The reply must still answer
what it answers and still omit what it omits.

Do not fix the assistant's failure, do not mention the omission, and do not add a note about
what was cut. The result must still be an instance of "{neg}" for the same reason the original
is.

Respond with a single JSON object: {{"conversation": [ ... ]}} where the value is a list of
message objects with 'role' and 'content' string fields. Output only the JSON."""

rows = [json.loads(l) for l in SRC.open()]
finds = [r for r in rows if r["label"] == "negative"]
print(f"{len(finds)} finds; asking {model} for one shorter version each", flush=True)

def parse(txt):
    t = txt.strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < 0: return None
    try: d = json.loads(t[i:j+1])
    except Exception: return None
    conv = d.get("conversation")
    if not isinstance(conv, list) or not conv: return None
    ms = []
    for m in conv:
        if not isinstance(m, dict) or not isinstance(m.get("role"), str) \
           or not isinstance(m.get("content"), str) or not m["content"].strip(): return None
        ms.append({"role": m["role"], "content": m["content"]})
    if not any(m["role"] == "user" for m in ms) or not any(m["role"] == "assistant" for m in ms):
        return None
    if budget.overage(ms) is not None: return None
    return ms

def work(item):
    k, find = item
    user = "Conversation to compress:\n" + json.dumps(find["inputs"], ensure_ascii=False, indent=1)
    for attempt in range(RETRIES):
        try:
            r = cli.chat.completions.create(model=model, max_tokens=4000,
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}])
            ms = parse(r.choices[0].message.content or "")
            if ms: return k, ms
        except Exception as e:
            print(f"  find {k}: attempt {attempt} failed: {str(e)[:90]}", flush=True)
        time.sleep(1.0 * (attempt + 1))
    return k, None

with ThreadPoolExecutor(max_workers=8) as pool:
    res = dict(pool.map(work, enumerate(finds)))

def L(ms): return sum(len(m["content"]) for m in ms)
n = 0
with OUT.open("w") as f:
    for k, find in enumerate(finds):
        ms = res.get(k)
        if ms is None: continue
        f.write(json.dumps({"id": f"short-{k}", "source_id": find["id"], "inputs": ms,
                            "label": "negative"}, ensure_ascii=False) + "\n")
        n += 1
import statistics
before = [L(x["inputs"]) for k, x in enumerate(finds) if res.get(k)]
after = [L(res[k]) for k in range(len(finds)) if res.get(k)]
print(f"\nwrote {n}/{len(finds)} shortened finds to {OUT}")
print(f"mean length {statistics.mean(before):.0f} -> {statistics.mean(after):.0f} ch "
      f"({statistics.mean(after)/statistics.mean(before):.2f}x); eval split mean is 668")
