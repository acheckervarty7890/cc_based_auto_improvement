# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`agentic_redteam` (the package keeps its historical name) is a **training-set curation
loop** for [tuberlens](https://github.com/blandfort/tuberlens) activation probes. It is
*not* a red-teaming tool any more — the earlier attacker/judge/tool-use scaffold was
replaced wholesale on the `dev_new_scaffolding` branch; see git history up to
`815770e4` for it. Two LLM roles drive the loop, each on one of two providers picked
per section in the config:

- **`claude_sdk`** — the Anthropic Python SDK (plain Messages API; no Agent SDK, no
  MCP, no tools).
- **`openrouter`** — the official `openai` SDK pointed at [OpenRouter](https://openrouter.ai/).

The **generator** writes batches of labelled conversations for the probe's concept —
it labels its own samples, half per class, from the probe's `description` — and the
**judge** reads how each batch moved the probe's dev-set AUROC and steers the next
round. The probe itself is the arbiter: a batch joins the training set only if a
probe trained with it scores higher on the dev set than the current probe.

One iteration (`cli.iterative_generate_main`), for a current probe *P* with mean dev
AUROC *A*:

1. **Directions.** The `n_batches` directions written for this iteration — the
   judge's, from the previous iteration's results; at iteration 0 the generator
   proposes them itself (`Generator.propose_directions`).
2. **Generate.** `n_batches` concurrent generator calls, batch *k* under direction *k*
   by `models[k % len(models)]`, each asked for `batch_size` samples, `batch_size/2`
   per class. Three guards per sample: length (`max_sample_tokens` through
   `TokenBudget`), label (must normalize to one of the probe's two), novelty (never
   generated earlier in the run, accepted or not, nor by a sibling batch in flight).
   A short batch gets up to `max_retries` in-context top-up asks naming only what is
   missing. The generator sees **no verdict** within an iteration.
3. **Warm the per-sample activation cache** for every new sample in one extraction
   model load (`retrain.warm_sample_activation_cache`), so the fits below load nothing.
4. **Score each batch on its own.** `retrain_probe(samples = accepted-so-far ∪ batch)`
   → candidate probe → per-split dev AUROC, read right after the fit while the dev
   activations are still staged. Δ = mean − *A*. Δ > `loop.min_auroc_gain` ⇒
   **accepted**; |Δ| ≤ `loop.exhausted_gain` ⇒ flagged **exhausted**; Δ < 0 ⇒ harmful.
   Batches are scored **independently against the same baseline** — never greedily.
5. **Union retrain.** `probe_iter{i+1}.pkl` trained on base ∪ every accepted batch of
   every iteration so far; its dev AUROC is the next baseline. Nothing accepted ⇒ the
   current probe file is copied over unchanged.
6. **Judge.** Every batch (direction, sample excerpts, per-split Δ, verdict) →
   `LLMJudge.guide` → a rewritten, bounded **rolling memo** + the next iteration's
   `n_batches` **directions** (`GuidanceRecord`, iteration `i+1`).
7. Optional `--eval` on the eval splits.

`loop.iterations` (CLI `--iterations` overrides) repeats this.

## Environment

The project's venv lives at:

```
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/
```

`tuberlens` is installed into it as an editable checkout under
`.venv_claude/src/tuberlens/`. Other key packages: `anthropic` (`provider: claude_sdk`),
`openai` (`provider: openrouter`), `pyyaml`.

**Always invoke the venv's Python by absolute path** (`source activate` adds nothing):

```bash
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/python -c "..."
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/pip install -e .
```

Required environment variables:

- `ANTHROPIC_API_KEY` — only when any section uses `provider: claude_sdk`.
- `OPENROUTER_API_KEY` — only when any section uses `provider: openrouter`.
- Optional: `OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`),
  `OPENROUTER_HTTP_REFERER`, `OPENROUTER_APP_TITLE`, `OPENROUTER_TIMEOUT_S` (default 60).
- Optional circuit-breaker tuning (see `circuit_breaker.py`):
  `OPENROUTER_MAX_CONSECUTIVE_ERRORS` (default 10),
  `OPENROUTER_MAX_CONSECUTIVE_FATAL_ERRORS` (default 3),
  `OPENROUTER_MAX_CONNECTION_OUTAGE_S` (default 1800) and
  `OPENROUTER_CONNECTION_BACKOFF_S` (default `60,120,480`).
- Optional activation-extraction tuning (see `model_loading.py`):
  `AGENTIC_REDTEAM_TRUNCATE_LAYERS` (default on — load only layers `0..probe.layer`;
  `0` loads the full model), `AGENTIC_REDTEAM_MAX_MEMORY` (e.g. `"0=21GiB,cpu=45GiB"`)
  and tuberlens' own `BATCH_SIZE` (default 1), which drives both `get_activations` and
  the per-sample chunking in `retrain`.
- Optional probe-fit staging (see `retrain._to_device_for_fit`):
  `AGENTIC_REDTEAM_STAGE_ACTIVATIONS` (default on; `0` fits host-resident) and
  `AGENTIC_REDTEAM_FIT_STAGING_RESERVE_GIB` (default 2).
- Optional probe-fit tuning read from tuberlens' settings (`tuberlens/probes/fused_ensemble.py`):
  `PROBE_FUSED_ENSEMBLE` (default on — fit *and* score an ensemble's members in one
  pass; `0` reverts both to the sequential paths; this repo reads the same setting in
  `ensemble.fusion_enabled()`, and `scripts/verify_ensemble_fusion.py` pins the
  dispatch), `PROBE_FUSED_MAX_MEMBERS`, `PROBE_EVAL_BATCH_SIZE`, and
  `PROBE_RESTORE_BEST_CHECKPOINT` (default **off**, see the checkpoint note below).
- Also from tuberlens' settings: `MAX_MEMORY` / `MODEL_MAX_MEMORY` and `OFFLOAD_BUFFERS`.
  `AGENTIC_REDTEAM_MAX_MEMORY` takes precedence on the `load_extraction_model` path.

**Model ids:** `probe.model` must be a full HF id (`meta-llama/Llama-3.2-1B-Instruct`,
`google/gemma-3-27b-it`). tuberlens' short keys (`llama-1b`) are NOT resolved by
`LLMModel.load`, which is what every load here goes through.

## Common commands

```bash
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/pip install -e .      # after dependency changes

# The loop
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/python scripts/iterative_generate.py \
  configs/example_generate.md --base-training-data data/highstakes_llama70b_50.jsonl \
  --probe-out-dir probes/my_run --eval --eval-dataset-dir eval_sets/highstakes

# End-to-end checks (fake LLMs; `real` also does real llama-1b fits, ~minutes on a laptop GPU)
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/python scripts/verify_generation_loop.py --mode fake
${REPO_ROOT}cc_based_auto_improvement/.venv_claude/bin/python scripts/verify_generation_loop.py --mode real
```

`--base-training-data` is **required**: it trains the initial probe (unless
`probe.path` warm-starts one) and is part of every retrain. **A dev set is required**
(`validation.dev_data` or `--dev-data`; the CLI errors without one): it is the fit's
validation set (early stopping) *and* the ΔAUROC scoring set, and it **must be disjoint
from the eval splits**. Base data and generated samples always train in full
(`test_size` is forced to 0.0 inside `retrain`); there is no CLI `--test-size` any more.
`--seed` governs the eval subsample and the (degenerate) split; `--base-data-fraction`,
`--ensemble-size`, `--probe-arch`, `--layer`, `--eval*`, `--[no-]combine-consecutive-messages`,
`--[no-]convert-tool-to-assistant`, `--base-activation-cache-dir`, `--activations-cache-dir`,
`--comparison-csv` and `--results-dir` behave as documented in `--help`.

**Budget the dev set's size deliberately** — it is resident for every fit and scored
every epoch. `get_activations` pads to 1024, so a row costs `1024 × hidden × 2` bytes:
`dev_samples/highstakes` (1908 rows) is **8.0 GB** on llama-1b and **21 GB** on
gemma-3-27b. And this loop fits `n_batches + 1` probes per iteration, each against the
whole dev set, so an oversized dev set multiplies through. Subsample the dev dir rather
than dropping it.

**Noise floor.** A single-probe fit's dev AUROC moves by roughly ±0.005 between
near-identical training sets, so with `ensemble_size: 1` and `min_auroc_gain: 0.0` some
accepted batches are noise. `probe.ensemble_size` (score-averaging deep ensemble) is
the lever that tightens it; `loop.exhausted_gain` only changes what the judge is told.

**Resume (`--resume`, default on) is three-tiered.** `probe_iter{N}.pkl` (written only
once an iteration's union retrain finishes) picks the iteration; `batches.jsonl` rows
with status `generated` are scored without regenerating and rows with status `scored`
/ `empty` / `generation_failed` are skipped; `guidance.jsonl` restores the directions
(and, if the run died between scoring and the judge, the judge is re-asked on the
stored batches). `--no-resume` calls `forget_loaded()` on both stores: the files are
still appended to, but nothing earlier is reused, restored or deduped against.

## Architecture

### `agentic_redteam/config.py`
Parses one markdown file: YAML frontmatter + `# Generator` / `# Judge` sections
(system prompts). Paths resolve relative to the config file. See
`configs/example_generate.md` for every key with comments. Shape:

```yaml
generator:
  provider: claude_sdk | openrouter   # default for bare-string models
  models: [<name> | {name, provider}] # batch k → models[k % len]
  n_batches: int                      # n (default 5)
  batch_size: int                     # m, even (default 20)
  concurrency: int                    # parallel generator calls (default 5)
  max_tokens: int                     # response cap per call (default 8192)
  max_sample_tokens: int              # default 1024; 0 disables the length guard
  max_retries: int                    # top-up calls per short batch (default 2)
judge:
  provider, model, max_tokens (2048)
  memo_word_budget: int               # default 400; the memo is in every generator prompt
  max_samples_per_batch: int          # excerpts shown per batch (default 6; 0 = all)
probe:
  path: <pkl>                         # OPTIONAL warm start
  model, layer, pos_class_label, neg_class_label, description, architecture  # from scratch
  ensemble_size: int                  # OPTIONAL 1..10; unset ⇒ inherit from the probe retrained
loop:
  iterations: int                     # default 3
  min_auroc_gain: float               # default 0.0
  exhausted_gain: float               # default 0.002
validation:
  dev_data: <jsonl | dir>             # REQUIRED by the CLI
eval:
  combine_consecutive_messages, convert_tool_to_assistant   # applied to training, samples, dev AND eval
  eval_max_samples: int               # 0 = full split
  data_description: str               # OPTIONAL, shown to the judge (coverage steering)
kaggle:                               # OPTIONAL precomputed eval/dev activations (see kaggle_activations.py)
  owner, eval_dataset_slug, eval_file_name, dev_dataset_slug, dev_file_name
output:
  run_dir: <dir>                      # batches.jsonl, guidance.jsonl, runlog.jsonl, auroc_history.csv, accepted_iter{N}.jsonl
  run_id, comparison_csv, activations_cache_dir, base_activation_cache_dir
```

`config.generator.models` is a list of `GeneratorModel(name, provider)` — not strings;
use `.model_names` for names. `batch_size` must be even (validated at parse time).

### `agentic_redteam/persistence.py`
`Message` / `Conversation` (frozen), `GeneratedSample(conversation, label)` — `label` is
the **human-readable** class label, `to_training_row()` gives the `{inputs, labels}`
shape `retrain` reads, `key` is the canonical text used for dedup.

`BatchRecord` — one batch: `run_id, iteration, batch_index, direction, generator_model,
provider, samples, n_requested, status, auroc_before, auroc_after, delta, accepted,
exhausted, n_dropped_too_long / _duplicate / _bad_label, n_generation_calls, error,
candidate_probe_path`. `auroc_*` map split name → AUROC plus `"mean"`; `delta` is on the
mean. `status` ∈ `generated` (written before scoring — the resume checkpoint),
`scored`, `empty`, `generation_failed`. A scored batch is a **second row** for the same
`(iteration, batch_index)`; every reader takes the newest row per key.

`BatchStore` (`batches.jsonl`, append-only, reloads on init): `get(i, k)`,
`for_iteration(i)`, `accepted_samples(before_iteration=)` (newest-row-per-key, in
iteration/batch order — the training-set additions), and the novelty guard:
`seen_keys` holds every sample ever generated in the run, `reserve(sample)` claims a
key for an in-flight batch **synchronously** (no `await` between check and claim, so
concurrent batches can't both admit the same conversation). `forget_loaded()` backs
`--no-resume`.

`GuidanceRecord(run_id, iteration, memo, directions, source, baseline_auroc)` —
`iteration` is the iteration the directions are **for**; `source` is `judge` or
`generator_proposal` (iteration 0). `GuidanceStore` (`guidance.jsonl`):
`for_iteration(i)`, `latest_memo_before(i)`. `RunLogger` (`runlog.jsonl`) is the event
sidecar.

### `agentic_redteam/generator.py`
`ProbeMeta` (labels, description, model name — **read off the probe**, never the config).
`build_generator_system_prompt(config, probe, memo)` = the `# Generator` prompt + the
concept block (labels + `description` verbatim) + the length limit + the JSON output
format + the judge's memo last. `_batch_request` asks for exactly `m/2` + `m/2` under
the direction and tells the generator to keep the two classes matched on surface
features; `_topup_request` names only what is still missing and why the rest was dropped.

`Generator` binds config + probe + `TokenBudget` + clients. `call()` is the only
provider touch point (Anthropic `AsyncAnthropic.messages.create`; OpenRouter through
`_openrouter_create_with_retry`, which reports to the breaker and retries connection
errors on the breaker's outage clock). `generate_batch()` runs the call + parse + guards
(`_admit`: length → class cap → `store.reserve`) + top-ups; any exception other than
`OpenRouterOutageError` is captured on the returned `BatchGeneration.error`, so one dead
batch never aborts an iteration. `propose_directions(n, memo, existing)` pads with a
free-choice brief if the model under-delivers twice, so the loop always has `n`.
`generate_batches` fans out under an `asyncio.Semaphore(concurrency)`; `batch_indices`
restricts a resumed iteration to the batches it still needs.

`parse_samples(text, pos, neg)` accepts a JSON array, a `{"samples": [...]}` wrapper or
bare objects, through `json_extract`; `normalize_label` maps case variants,
`positive`/`negative`, and a label string embedding exactly one class name.

### `agentic_redteam/llm_judge.py`
`LLMJudge.guide(batches, iteration, n_directions, prior_memo, auroc_before, auroc_after,
min_gain, exhausted_gain) → Guidance(memo, directions)`. The reply format is
`## Memo` (markdown, ≤ `memo_word_budget` words, rewritten not appended) then
`## Directions` (one fenced JSON array of exactly `n` strings) — prose in markdown,
list in JSON, so a long memo never has to survive inside a JSON string.
`parse_guidance` splits on the heading; a reply with no parseable list yields
`directions=[]` and the CLI fills the gap via `Generator.propose_directions`.

The judge is shown, per batch: direction, generator, class counts, the verdict
(accepted / exhausted / rejected-harmful / rejected-below-threshold / failed), per-split
Δ, and `max_samples_per_batch` excerpts alternating classes (`_pick_samples`), plus the
baseline before and after the union retrain and the acceptance rule itself. It never
sees per-sample probe scores.

**Prompt register is neutral analyst** ("training-set curation for a text classifier"),
not adversarial — the previous scaffold's adversarially-phrased summaries drew refusals.
**Refusal guard**: `_looks_like_refusal` (three guards: markdown opener ⇒ write-up; quoted
spans blanked so a *cited* refusal phrase isn't an *uttered* one; a marker must start
within 60 chars) → one in-context re-ask (`_REFUSAL_RETRY_NUDGE`) → `JudgeRefusalError`,
which the CLI re-raises (a refusal stored as the memo would be injected into every
generator prompt). A JSON/markdown reply never trips it: `{` and `#` are not refusal
openers. Transient judge failures are logged (`guidance_error`) and the next iteration
re-asks on the stored batches.

### `agentic_redteam/json_extract.py`
`extract_json_values(text, accept)` — fenced blocks first (each parsed independently, so
a `max_tokens`-guillotined last block still yields the complete ones before it), then
balanced `[...]`/`{...}` spans (a span nested inside an accepted one is skipped), then
the whole text; `loads_forgiving` retries a fragment with the closers it is missing
(`json_repairs`). `extract_string_list` is the directions parser.

### `agentic_redteam/retrain.py`
Trains probes on base data ∪ in-memory samples. Entry points:

- `train_initial_probe(...) → RetrainResult` — first probe from base data alone.
- `retrain_probe(samples, base_probe_path, base_training_data_path, new_probe_path,
  ...) → RetrainResult` — `samples` are `GeneratedSample`s or `{inputs, labels}` dicts
  (`samples_to_dataset`); labels map through the base probe's class labels. Inherits
  architecture (`_infer_probe_spec`) and ensemble size from the base probe unless told
  otherwise. **There is no JSONL input and no preprocessing/contrastive step** any more.
- `RetrainResult(new_probe_path, n_extra_samples, n_training_samples_total,
  ensemble_size, dev_auroc)` — `dev_auroc` is `{split: auroc, ..., "mean": ...}`,
  computed by `_per_split_auroc` **inside `_train_with_cached_base_activations` right
  after the fit**, while the validation activations are still resident/staged. The
  function returns `(probe, dev_scores)`; `dev_splits` (`[(name, n_rows), ...]`, from
  `_load_dev_dataset`'s third return value, in concatenation order) is what makes a
  per-split number readable off the single dev blob. `"mean"` is the unweighted mean
  over splits carrying both classes.
- `score_probe_on_dev(probe_path, dev_data_path, cache_dir, ...)` — the baseline for a
  warm-started or resumed probe, on the **same dev blob** the fits use.
- `warm_sample_activation_cache(samples, base_probe_path, cache_dir, ...)` — one model
  load for every uncached sample, so the `n_batches` candidate fits are cache hits.
- `load_probe(path)` (CPU unpickle + move every member's classifier to tuberlens'
  `DEVICE`/`DTYPE` via `iter_probe_members`) and `read_probe_metadata(path)`.

Validation always comes from the dev set on the CLI path: `dev_data_path` forces
`test_size = 0.0`, so `stable_train_test_split` puts every base and generated sample
on the train side and the dev set is the sole validation set. (The split machinery is
kept; `stable_train_test_split` is content-deterministic — `sha256(seed : content)`.)
Note tuberlens uses the validation set to select a best-val-AUROC epoch and to
early-stop — only for the pytorch architectures; `SklearnProbe.fit` ignores it.

**What a fit returns is the LAST epoch, not the best one — deliberately.** tuberlens'
`PytorchAdamClassifier.train` used to keep a shallow `state_dict().copy()` whose tensors
the optimizer kept updating; every probe ever trained here is the last-epoch probe.
The copy is now a real clone but **restoring it is opt-in** (`PROBE_RESTORE_BEST_CHECKPOINT=1`):
flipping the default would make every cross-iteration comparison incommensurable, and it
is a *different* probe, not a better one (measured: dev AUROC up, held-out eval a wash).

**Deep ensembles (`ensemble_size`).** `_resolve_ensemble_seeds(seed, n)` draws member
seeds from the repo-pinned `ENSEMBLE_SEEDS[:n]` (never a walk off `--seed`; `n == 1`
returns `[seed]` so the single-probe path is byte-identical to what it always was).
`_train_with_cached_base_activations` fits members together via tuberlens'
`ProbeFactory.build_ensemble` when `ensemble.fusion_enabled()`, else one
`ProbeFactory.build` per seed through `_build` (which calls `seed_everything`, not just
`torch.manual_seed`). Only the fit repeats — extraction, caches and merge are shared.

**Activation caching (base blob + per-sample + dev blob).** `_base_activation_cache_paths`
keys the base split on the file hash + `model | layer | seed | test_size | split_field |
fraction | combine | convert`; `_sample_activation_cache_path` keys each generated
conversation on its own (transformed) messages + `model | layer | combine | convert`
(subdir `sample_acts_<model>_L<layer>/`); `_dev_activation_cache_path` keys the dev set
on the dev files' bytes + `model | layer | combine | convert`. Per-sample caching is
what makes the growing set cheap: a sample first seen in iteration *k* is forwarded
once — and the loop warms the cache before the fits, so a candidate fit loads no
model. Per-sample blobs are written through per chunk of `extraction_batch_size()`
(resumable; each row stored at its own width, `_concatenate_consuming` re-pads at
merge). All three load **by path without validating inputs** — anything that changes
which samples are selected or how they're tokenized must be folded into the key.

**Host-RAM budget.** The extraction model is released (`_release_model`, hooks stripped
first) immediately after the last extraction, before the merge and fit;
`_concatenate_consuming` merges at ~1× peak (it **consumes** its inputs — capture `len()`
first; `torch.empty` + explicit pad-fill is load-bearing).

**Where a fit's wall-clock goes: `_to_device_for_fit`.** Stages the merged
train/validation activations on the fit device once (the per-epoch `.to()` becomes a
no-op; measured 18.35 → 0.16 ms/sample, a 4.5 h fit → ~7 min). It **sorts by size** (the
bigger set wins the card — for `dev_samples/highstakes` that is the dev set, for the
other two concepts the training set), is **capacity-checked** against
`_allocatable_bytes` (driver-free + torch's reserved-but-unallocated pool) less a
reserve, moves a dataset **whole or not at all** (a split dataset raises in tuberlens'
`Activation.__post_init__`), and restores everything on failure. Bit-identical fits
either way; `scripts/verify_fit_staging.py` pins it without a GPU.

### `agentic_redteam/ensemble.py`
`EnsembleProbe` — `n` same-architecture probes fit on the same activations under
`ENSEMBLE_SEEDS[:n]`, **probabilities averaged before the threshold**; duck-typed to the
probe surface (`model_name` / `layer` / labels / `description` / `predict_proba` /
`predict_proba_from_inputs`), so nothing below `predict_proba` knows it is an ensemble.
`_mean_proba` scores all members in one fused pass (`stacked_probs`) when
`fusion_enabled()`, else per member. `iter_probe_members(probe)` / `ensemble_size(probe)`
let code that must reach *inside* a probe treat both cases uniformly — a
`getattr(probe, "_classifier")` on an ensemble silently returns `None`. `ENSEMBLE_SEEDS`
are frozen; raising `MAX_ENSEMBLE_SIZE` (10) means *appending*. `DETERMINISTIC_ARCHS`
(`sklearn`, `difference_of_means`, `lda`) only trigger a warning: `n` identical members.

### `agentic_redteam/model_loading.py`
`load_extraction_model(model_name, layer)` — the single loader for the frozen extraction
LLM (**never call `LLMModel.load` directly for extraction**). It loads only layers
`0..layer` (`_truncated_config`; exact, since the stack is causal — verified
bit-identical, hence absent from every cache key), resolves the memory budget in the
precedence `AGENTIC_REDTEAM_MAX_MEMORY` > tuberlens `MAX_MEMORY`/`MODEL_MAX_MEMORY` >
unpinned, and carries `offload_buffers` from tuberlens' `OFFLOAD_BUFFERS`. `unhook_model`
strips accelerate's hooks so a release actually frees the card.
`extraction_batch_size()` reads tuberlens' `BATCH_SIZE`.

### `agentic_redteam/token_budget.py`
The length safeguard against tuberlens' 1024-token activation cap. `count_tokens`
reproduces `tokenize_inputs` exactly — two traps: its `<bos>` strip is a **no-op** (never
subtract 1) and the tokenizer is called with `add_special_tokens=False`. `TokenBudget`
binds it to a run; `overage(messages)` returns the count only when over the cap, else
`None` — **fails open** (uncountable ⇒ allowed). `warmup()` loads the tokenizer before
the async fan-out (the count runs on the loop thread).

### `agentic_redteam/circuit_breaker.py`
Process-global breaker that stops a run when OpenRouter is **durably** down, because
every call site is individually fault-tolerant (a dead batch is recorded and the
iteration continues; a judge hiccup is re-asked next iteration) and would otherwise
grind through a drained balance producing empty batches. Three failure classes:
**transient** (429/5xx/empty choices — `OPENROUTER_MAX_CONSECUTIVE_ERRORS`, 2/4/8 s
backoff), **connection** (retried on a minutes-scale schedule and bounded by **elapsed
time**, not a count — with `concurrency: 10` one network event is observed ten times at
once), **fatal** (401/402/403 — `OPENROUTER_MAX_CONSECUTIVE_FATAL_ERRORS`, no backoff).
Every OpenRouter call reports `record_success` / `record_failure`; `raise_if_tripped()`
before each call; backoff sleeps go through `sleep_sync` / `sleep_async` (5 s chunks,
re-checking the trip). The CLI's `@_exit_on_outage` turns the error into exit code 3.

### `agentic_redteam/evaluation.py`
`evaluate_probe(probe_path, eval_dataset_dir, activations_cache_dir, splits=None,
max_samples=100, seed=42, combine…, convert…, kaggle_source=None)` scores one probe on
the auto-discovered `<dir>/*.jsonl` splits via tuberlens `get_performances`.
`seed_everything(seed)` before `subsample_balanced_subset` keeps the subset identical
across probes, which is what keeps the path-keyed cache (`acts_n{N}_seed{S}.pt` /
`acts_full.pt`) correct. `_assign_cached_activations` attaches cached blobs *before*
`get_performances`, which would otherwise load the model on meeting any split without
activations.

### `agentic_redteam/kaggle_activations.py`
`prefetch_eval_activations` / `prefetch_dev_activations` download **precomputed**
activations published on Kaggle into the exact cache paths the eval / the fit look for
(the dev set is assembled from per-split datasets into the single content-hashed blob,
in `sorted(glob)` order, right-padded to the common width). Explicit
`KaggleActivationSource(owner, dataset_slug, file_name)` with `{split}` / `{slug}`
templates (`{slug}` because Kaggle rejects underscores); every blob is validated against
the probe's `model_name`/`layer` and the split's row count. Auth: `KAGGLE_CONFIG_DIR`
(the **directory** holding `kaggle.json`) or `KAGGLE_API_TOKEN`.
`scripts/publish_kaggle_eval_activations.py` / `scripts/extract_eval_activations.py`
are the upload side.

### Eval dataset splits on disk
Eval splits live under **`eval_sets/<concept>/`** and dev splits under
**`dev_samples/<concept>/`**, sharing the concept names `highstakes`, `hu_ha`,
`instructions`. Every row is a tuberlens `LabelledDataset` record: `inputs` is a
**JSON-encoded string** of `[{role, content}, ...]` and `labels` is the class string;
other columns are provenance.

- **`eval_sets/highstakes/`** — `anthropic_hh_balanced` (2984), `mt_balanced` (604),
  `mts_balanced` (86), `toolace_balanced` (734); labels `high-stakes` / `low-stakes`.
  The real rationale column is `scale_label_explanation` (`label_explanation` is boilerplate).
- **`eval_sets/hu_ha/`** — `eval_ai_dilemmas` (136), `eval_balanced_refusal` (400),
  `eval_daily_dilemmas` (196), `eval_ant_hh` (134); labels `harmful_to_human` /
  `not_harmful_to_human`; three of four splits are paired (same prompt, both classes).
- **`eval_sets/instructions/`** — seven splits (`anthropic_harmless_refusal`,
  `bbq_substitution`, `hc_context_drift`, `hc_contradiction`, `mm_substitution`,
  `oig_context_drift`, `oig_omission`); labels `assistant_follows_the_instruction` /
  `assistant_does_not_follow_the_instruction`.

**`dev_samples/`** holds the matching dev sets, **verified disjoint** from eval:
`highstakes/` (1908: 1028/278/274/328), `hu_ha/` (290), `instructions/` (436). Anything
added here must stay disjoint from the concept's eval dir — a dev row in eval means the
best-epoch checkpoint *and every acceptance decision* are made on the test set.

Base training sets under `data/` (`highstakes_llama70b_50.jsonl`,
`hu_harm_llama70b_50.jsonl`, `instructions_llama70b_50.jsonl`, …) are `{inputs, labels}`
JSONLs in the same schema. Note the **49-row optimizer-step threshold**: under ~49
training rows the pytorch probe never takes an optimizer step; the 50-row base sets sit
right at it, so keep `base_data_fraction` at 1.0 with them.

### `agentic_redteam/cli.py`
`iterative_generate_main` (script `scripts/iterative_generate.py`, console script
`iterative-generate`). Resolves flags > config > defaults exactly as before
(`--ensemble-size` > `probe.ensemble_size` > inherit; cache dirs; transforms), requires
a dev set, does the optional Kaggle dev prefetch, obtains the initial probe (resume >
warm start > `train_initial_probe`), scores the baseline (from the fresh fit's
`dev_auroc`, else `score_probe_on_dev`), builds `Generator` + `LLMJudge` from the
**probe's** metadata, then runs the loop described at the top. Per iteration it writes:
`batches.jsonl` rows (`generated` then `scored`), `candidates/probe_iter{i}_batch{k}.pkl`,
`accepted_iter{i+1}.jsonl`, `probe_iter{i+1}.pkl`, `auroc_history.csv`, a
`GuidanceRecord` for `i+1`, and (with `--eval`) the comparison CSV. `_free_gpu()` runs
after every fit, scoring and eval.

## Conventions to preserve

- **Probe metadata is the source of truth.** Read `pos_class_label` / `neg_class_label` /
  `description` / `model_name` / `layer` off the loaded probe (`read_probe_metadata`);
  the config's `probe:` fields only describe a from-scratch probe.
- **The generator labels; the probe decides.** No judge labelling, no confidence gate:
  a batch is kept or dropped purely on its dev ΔAUROC against the current probe, and
  every batch of an iteration is scored against the **same** baseline (independent, not
  greedy). Don't reintroduce per-sample verdicts into the generator loop — the generator
  is deliberately blind within an iteration and steered only by the judge's memo.
- **Never tell the generator a quota it can satisfy early.** The batch size is a
  workload, not a target; there is no success count anywhere in a prompt.
- **A probe pickle may be an `EnsembleProbe`**, and that must stay invisible below
  `predict_proba`. Reach inside only through `iter_probe_members`.
- **Count tokens through `token_budget`, never by hand**, and any new producer of
  training samples needs the same guard, failing **open**.
- **`OpenRouterOutageError` must never be swallowed.** Every `except Exception` around an
  OpenRouter call (`Generator.generate_batch`, `propose_directions`, the judge call in
  `cli`) has an explicit `except OpenRouterOutageError: raise` before it, mirroring
  `JudgeRefusalError`. New call sites must report to the breaker.
- **Sidecars are append-only; newest row per key wins.** A batch is re-scored by
  appending, never by rewriting.
- **Load the extraction LLM through `model_loading.load_extraction_model`**, free GPU
  memory between heavy phases (`_free_gpu`), and fold anything that changes activations
  into the cache keys — the caches load by path without validating inputs.
- **Use `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`** as the
  canonical Anthropic model ids.
