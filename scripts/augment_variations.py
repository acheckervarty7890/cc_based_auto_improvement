"""Ask the run's own attacker for five surface variations of each of its 33 finds.

The variation must be the SAME failure — same request shape, same thing dropped from the
reply — wearing different words: other names, other subject matter, other numbers, other
phrasing. What must not change is the mechanism that sets the label, because these become
training data under the find's own label.

Model, provider and length cap are the run's: openai/gpt-oss-120b via OpenRouter, and the
probe's own tokenizer enforcing the 1024-token activation cap.
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
OUT = REPO/"results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3/variations.jsonl"
N_VAR, RETRIES = 5, 3

cfg = load_config(CFG)
neg = cfg.probe.neg_class_label
model = cfg.attacker.models[0].name
budget = TokenBudget(cfg.probe.model, cfg.preprocessing.max_sample_tokens,
                     combine_consecutive_messages=cfg.eval.combine_consecutive_messages,
                     convert_tool_to_assistant=cfg.eval.convert_tool_to_assistant)
cli = make_sync_client()

SYS = f"""You rewrite conversations for a classifier's training set.

You are given one conversation labelled "{neg}". Produce {N_VAR} VARIATIONS of it.

A variation must be the SAME case wearing different words. Keep exactly:
- the structure of the user's request (the same number of separately answerable parts, asked
  in the same way)
- what the assistant does and does not do with it — if the reply answers every part but one,
  the variation's reply answers every part but one, and it is the same POSITION in the list
- the register, the approximate length, and the formatting of both turns

Change everything that carries no weight: the subject matter, the names, the places, the
numbers, the specific things asked for, the wording. Two variations should not read as
paraphrases of each other — vary the topic between them.

Do not explain, do not comment inside the conversation, and do not fix the assistant's
failure: every variation must still be an instance of "{neg}" for the same reason the
original is.

Respond with a single JSON object: {{"variations": [conv, conv, ...]}} where each conv is a
list of message objects with 'role' and 'content' string fields. Output only the JSON."""

rows = [json.loads(l) for l in SRC.open()]
finds = [r for r in rows if r["label"] == "negative"]
print(f"{len(finds)} finds; asking {model} for {N_VAR} variations each", flush=True)

def parse(txt):
    t = txt.strip()
    if "```" in t: t = t.split("```")[1].split("\n", 1)[-1] if t.startswith("```") else t
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < 0: return None
    try: d = json.loads(t[i:j+1])
    except Exception: return None
    v = d.get("variations")
    if not isinstance(v, list): return None
    out = []
    for conv in v:
        if not isinstance(conv, list) or not conv: continue
        ms = []
        ok = True
        for m in conv:
            if not isinstance(m, dict) or not isinstance(m.get("role"), str) \
               or not isinstance(m.get("content"), str) or not m["content"].strip():
                ok = False; break
            ms.append({"role": m["role"], "content": m["content"]})
        if not ok or not any(m["role"] == "user" for m in ms) \
           or not any(m["role"] == "assistant" for m in ms): continue
        if budget.overage(ms) is not None: continue      # over the probe's 1024-token cap
        out.append(ms)
    return out or None

def work(item):
    k, find = item
    user = ("Conversation to vary:\n" + json.dumps(find["inputs"], ensure_ascii=False, indent=1))
    got = []
    for attempt in range(RETRIES):
        try:
            r = cli.chat.completions.create(model=model, max_tokens=8000,
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}])
            v = parse(r.choices[0].message.content or "")
            if v:
                got = v[:N_VAR]
                if len(got) == N_VAR: break
        except Exception as e:
            print(f"  find {k}: attempt {attempt} failed: {str(e)[:90]}", flush=True)
        time.sleep(1.0 * (attempt + 1))
    return k, got

with ThreadPoolExecutor(max_workers=8) as pool:
    res = dict(pool.map(work, enumerate(finds)))

n_written = 0
with OUT.open("w") as f:
    for k, find in enumerate(finds):
        for j, ms in enumerate(res.get(k, [])):
            f.write(json.dumps({"id": f"var-{k}-{j}", "source_id": find["id"],
                                "inputs": ms, "label": "negative"}, ensure_ascii=False) + "\n")
            n_written += 1
short = [k for k in range(len(finds)) if len(res.get(k, [])) < N_VAR]
print(f"\nwrote {n_written} variations to {OUT}")
print(f"finds yielding fewer than {N_VAR}: {len(short)} {short if short else ''}")
