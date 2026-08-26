"""Pair every variation, then retrain on the augmented set and score all seven splits.

Step 2: the run's own contrastive generator (openai/gpt-5.1, GENERATION_PROMPT_VERSION 3)
writes an opposite-class partner for each variation, one variation at a time — so the pair is
minimal with respect to THAT variation, not to the find it descends from.

Step 3: fit base 50 + the augmented red-team set, validated on the same 436-row dev set, and
score every eval split. Activations for the variations are new, so this one does load the
extraction model; every other activation comes from cache.
"""
import json, pickle, sys
from pathlib import Path
REPO = Path("/workspace/cc_based_auto_improvement"); sys.path.insert(0, str(REPO/"src"))
from agentic_redteam.config import load_config
from agentic_redteam.evaluation import evaluate_probe
from agentic_redteam.preprocessing import generate_contrastive_dataset
from agentic_redteam.retrain import (_base_activation_cache_paths, _cpu_unpickle,
    _dev_activation_cache_path, _infer_probe_spec, _load_dev_dataset, _resolve_ensemble_seeds,
    _train_with_cached_base_activations, stable_train_test_split)
from agentic_redteam.token_budget import TokenBudget
from tuberlens.interfaces.dataset import LabelledDataset, Message as TLMessage

CFG = REPO/"configs/gptoss120b_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3.md"
RES = REPO/"results_instructions_gemma27b_scopecheck_exp24_tellattacker_iter5_v3"
BP  = REPO/"probes/instructions_gemma27b_evaldesc_omission/probe_iter0.pkl"
BASE= REPO/"data/instructions_llama70b_50.jsonl"; DEV = REPO/"dev_samples/instructions"
BC  = REPO/"results_instructions_gemma27b_shared/base_activations"
EC  = REPO/"results_instructions_gemma27b_shared/eval_activations"
# Which reshaped set to pair and retrain on: "variations" (augment_variations.py) or
# "shortened" (shorten_samples.py). Everything downstream is named off it, so the two never
# share a cache, a dump or a probe.
STEM  = sys.argv[1] if len(sys.argv) > 1 else "variations"
VAR   = RES/f"{STEM}.jsonl"
PAIRS = RES/f"{STEM}_paired.jsonl"
CACHE = RES/f"{STEM}_contrastive_cache.jsonl"

cfg = load_config(CFG); pp = cfg.preprocessing
pos, neg = cfg.probe.pos_class_label, cfg.probe.neg_class_label
C, V = cfg.eval.combine_consecutive_messages, cfg.eval.convert_tool_to_assistant

variations = [json.loads(l) for l in VAR.open()]
print(f"{len(variations)} variations", flush=True)
budget = TokenBudget(cfg.probe.model, pp.max_sample_tokens,
                     combine_consecutive_messages=C, convert_tool_to_assistant=V)
records = [{"inputs": r["inputs"], "labels": neg, "ids": r["id"]} for r in variations]
out = generate_contrastive_dataset(records, pos_class_label=pos, neg_class_label=neg,
    provider=pp.provider, model=pp.model, max_concurrent=pp.max_concurrent,
    max_tokens=pp.max_tokens, max_retries=pp.max_generation_retries, cache_path=CACHE,
    assistant_centric=pp.assistant_centric, concept_description=pp.concept_description,
    label_guidance=pp.label_guidance, eval_data_description=cfg.eval.data_description,
    token_budget=budget)
srcs = [r for r in out if r["labels"] == neg]
gens = [r for r in out if r["labels"] == pos]
print(f"kept {len(srcs)} variations, {len(gens)} partners -> {len(srcs)+len(gens)} rows", flush=True)
with PAIRS.open("w") as f:
    for i, r in enumerate(srcs): f.write(json.dumps({"id": f"{STEM}-src-{i}", "inputs": r["inputs"], "label": "negative"}, ensure_ascii=False)+"\n")
    for i, r in enumerate(gens): f.write(json.dumps({"id": f"{STEM}-gen-{i}", "inputs": r["inputs"], "label": "positive"}, ensure_ascii=False)+"\n")

with BP.open("rb") as f: bp = _cpu_unpickle(f)
spec = _infer_probe_spec(bp); seeds = _resolve_ensemble_seeds(42, 10); TS = 0.0
base = LabelledDataset.load_from(Path(BASE), pos_class_label=pos, neg_class_label=neg,
    combine_consecutive_messages=C, convert_tool_to_assistant=V)
btr, bva = stable_train_test_split(base, test_size=TS, split_field=None, seed=42)
btc, bvc = _base_activation_cache_paths(BC, BASE, bp.model_name, bp.layer, 42, TS, None, C, V, 1.0)
dv, dfiles, _sizes = _load_dev_dataset(DEV, pos, neg, C, V, verbose=False)
dvc = _dev_activation_cache_path(BC, dfiles, bp.model_name, bp.layer, C, V)

rows = srcs + gens
rt = LabelledDataset(
    inputs=[[TLMessage(role=m["role"], content=m["content"]) for m in r["inputs"]] for r in rows],
    ids=[f"{STEM}-{i}" for i in range(len(rows))],
    other_fields={"labels": ["negative"]*len(srcs) + ["positive"]*len(gens)})
rtr, rva = stable_train_test_split(rt, test_size=TS, split_field=None, seed=42)
print(f"training on base {len(base)} + {len(rows)} reshaped red-team rows", flush=True)
probe = _train_with_cached_base_activations(base_train=btr, base_val=bva, redteam_train=rtr,
    redteam_val=rva, dev_val=dv, model_name=bp.model_name, layer=bp.layer, probe_spec=spec,
    pos_class_label=pos, neg_class_label=neg, probe_description=bp.description,
    base_train_cache=btc, base_val_cache=bvc, dev_val_cache=dvc, redteam_cache_dir=BC,
    combine_consecutive_messages=C, convert_tool_to_assistant=V, seed=42,
    ensemble_seeds=seeds, verbose=True)
q = RES/f"probe_{STEM}.pkl"
with q.open("wb") as f: pickle.dump(probe, f)
df = evaluate_probe(probe_path=q, eval_dataset_dir=REPO/"eval_sets/instructions",
    activations_cache_dir=EC, splits=None, max_samples=None, seed=42,
    combine_consecutive_messages=C, convert_tool_to_assistant=V)
df.to_csv(RES/f"eval_{STEM}.csv", index=False)
print(df.to_string(index=False))
