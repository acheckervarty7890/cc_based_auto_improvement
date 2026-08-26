"""Ask the run's own attacker to re-tell each of its 33 finds in the DEV SET's style.

Motivation. The finds and the data the probe is scored on differ in register as much as in
content: a v3 find averages 1175 characters of scenario framing, roles and prose, while an
oig_omission dev sample averages 733 and is bare — newline-separated questions, then a reply
that answers them one line at a time and stops. `shorten_samples.py` already showed that
length alone is worth +0.026 on that split. This asks the sharper version of the question: if
the SAME case is told the way the eval data tells it, is it worth more as training data?

The distinction this script rests on is content vs. form. What must survive: the number of
separately answerable parts, which one the reply drops, its position, the topic and the
specifics. What is rewritten: layout, register, framing, and everything else about how the
exchange is presented.

The exemplars are five NEGATIVE-class dev rows (the class every find belongs to), so they
show both halves of the form at once — how the request is laid out and what an omitting reply
looks like there. They are drawn from `dev_samples/`, never `eval_sets/`, and they are shown
to the attacker as few-shot style references only: no dev row is copied, and none enters the
training set. Note this still moves information from the validation set into the training
data's FORM, which is the price of the experiment and is why it is a separate arm rather than
a change to the pipeline.

Output matches shorten_samples.py / augment_variations.py, so the next step is the shared
`scripts/pair_and_retrain.py restyled`.
"""
import json, random, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO = Path("/workspace/cc_based_auto_improvement"); sys.path.insert(0, str(REPO/"src"))
from agentic_redteam.config import load_config
from agentic_redteam.openrouter_client import make_sync_client
from agentic_redteam.token_budget import TokenBudget

CFG = REPO/"configs/gptoss120b_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3.md"
SRC = REPO/"probes/instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3/redteam_postprocessed_iter5.jsonl"
DEV = REPO/"dev_samples/instructions/oig_omission.jsonl"
OUT = REPO/"results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3/restyled.jsonl"
N_EXEMPLARS, SEED, RETRIES = 5, 42, 3

cfg = load_config(CFG)
neg_label = cfg.probe.neg_class_label
model = cfg.attacker.models[0].name
budget = TokenBudget(cfg.probe.model, cfg.preprocessing.max_sample_tokens,
                     combine_consecutive_messages=cfg.eval.combine_consecutive_messages,
                     convert_tool_to_assistant=cfg.eval.convert_tool_to_assistant)
cli = make_sync_client()

# Five negative-class dev rows, drawn deterministically so the arm is reproducible.
dev_neg = [json.loads(l) for l in DEV.open()]
dev_neg = [d for d in dev_neg if d["labels"] == neg_label]
picks = sorted(random.Random(f"restyle:{SEED}").sample(range(len(dev_neg)), N_EXEMPLARS))
exemplars = [json.loads(dev_neg[i]["inputs"]) for i in picks]
print(f"style exemplars: rows {picks} of {len(dev_neg)} negative dev samples", flush=True)

EX = "\n\n".join(
    f"EXAMPLE {n + 1}:\n" + json.dumps(e, ensure_ascii=False, indent=1)
    for n, e in enumerate(exemplars)
)

SYS = f"""You re-tell conversations so they read like a particular corpus.

Below are {N_EXEMPLARS} conversations drawn from the corpus a classifier is evaluated on. All
of them are labelled "{neg_label}". Study how they are WRITTEN — how the user's request is laid
out, how many turns there are, how the reply is structured, its register, its length, how it
begins and how it ends.

{EX}

You will then be given one conversation of your own, also labelled "{neg_label}". Re-tell it in
the style of the examples above.

KEEP THE CONTENT EXACTLY:
- the same number of separately answerable parts in the user's request, in the same order
- the same part left unanswered by the assistant, in the same position
- the same topic, the same names, the same facts, the same specifics
- everything the reply does answer, it must still answer, with the same substance

CHANGE ONLY THE FORM: layout, framing, register, turn structure, phrasing and length, so that
the result would not stand out if it were dropped into the examples above. Drop scene-setting,
role-play framing, personas, headers and signatures if the examples have none. Match their
length — they are much shorter than most conversations you will be given.

Do not copy any sentence from the examples: they are style references, not content. Do not fix
the assistant's failure, do not mention the omission, and do not remark on the rewriting. The
result must still be an instance of "{neg_label}" for the same reason the original is.

Respond with a single JSON object: {{"conversation": [ ... ]}} where the value is a list of
message objects with 'role' and 'content' string fields. Output only the JSON."""

rows = [json.loads(l) for l in SRC.open()]
finds = [r for r in rows if r["label"] == "negative"]
print(f"{len(finds)} finds; asking {model} to re-tell each in the dev set's style", flush=True)


def parse(txt):
    t = txt.strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < 0: return None
    try: d = json.loads(t[i:j + 1])
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
    user = "Conversation to re-tell:\n" + json.dumps(find["inputs"], ensure_ascii=False, indent=1)
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


LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else len(finds)
todo = list(enumerate(finds))[:LIMIT]
with ThreadPoolExecutor(max_workers=8) as pool:
    res = dict(pool.map(work, todo))

def L(ms): return sum(len(m["content"]) for m in ms)
n = 0
with OUT.open("w") as f:
    for k, find in todo:
        ms = res.get(k)
        if ms is None: continue
        f.write(json.dumps({"id": f"restyled-{k}", "source_id": find["id"], "inputs": ms,
                            "label": "negative"}, ensure_ascii=False) + "\n")
        n += 1
before = [L(x["inputs"]) for k, x in todo if res.get(k)]
after = [L(res[k]) for k, _ in todo if res.get(k)]
dev_mean = statistics.mean(sum(len(m["content"]) for m in json.loads(d["inputs"])) for d in dev_neg)
print(f"\nwrote {n}/{len(todo)} restyled finds to {OUT}")
if after:
    print(f"mean length {statistics.mean(before):.0f} -> {statistics.mean(after):.0f} ch; "
          f"dev negative-class mean is {dev_mean:.0f}")
