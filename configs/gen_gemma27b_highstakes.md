---
# =============================================================================
# generator_experiment_1 — ARM 1 of 3: HIGH-STAKES
#
# The generate -> score -> retrain -> guide loop (dev_new_scaffolding) run against a
# probe on google/gemma-3-27b-it, layer 32, for FIVE iterations.
#
# WHAT VARIES ACROSS THE THREE ARMS: the concept only. probe.pos/neg_class_label,
# probe.description, validation.dev_data, the kaggle: block's resolved refs, the base
# training data (passed on the command line by the runner) and the output paths. Every
# generator / judge / loop knob below is IDENTICAL in all three arms and is copied
# verbatim from configs/example_generate.md, so the three arms are directly readable
# against each other.
#
# WHAT DIFFERS FROM example_generate.md, and why (only these four things):
#   1. probe:      meta-llama/Llama-3.2-1B-Instruct L8  ->  google/gemma-3-27b-it L32,
#                  with the one-line description every gemma-3-27b high-stakes run in
#                  this repo has carried.
#   2. loop.iterations:  3 -> 5   (asked for).
#   3. eval transforms:  false/false -> TRUE/TRUE. NOT a free choice — see the eval:
#                  block. The precomputed activations on Kaggle were extracted under
#                  these transforms, and the caches load by path without validating
#                  their inputs, so a mismatch here silently trains and scores on
#                  activations of a different message representation.
#   4. kaggle:     added — the whole point of this run is that no gemma-3-27b forward
#                  pass is ever needed for the eval splits or the dev set.
#
# NOT SET, deliberately: probe.ensemble_size (example_generate.md leaves it unset, so
# every fit here is a SINGLE probe). Note the noise floor this implies — a single fit's
# dev AUROC moves by roughly +/-0.005 between near-identical training sets, and
# loop.min_auroc_gain is 0.0, so some accepted batches will be noise. That is inherited
# from example_generate.md on purpose; raise ensemble_size (and it applies to every fit,
# n_batches + 1 of them per iteration) if the ledger looks like coin flips.
# =============================================================================

generator:
  provider: openrouter            # claude_sdk | openrouter — default for bare-string models below
  models:
    - meta-llama/llama-3.3-70b-instruct
  n_batches: 5                    # batches per iteration (n); one generator call each
  batch_size: 10                  # samples per batch (m); even — half per class
  concurrency: 5                  # parallel generator calls
  max_tokens: 8192                # response cap per generator call
  max_sample_tokens: 1024         # drop generated samples longer than this (probe tokenizer; 0 = no cap)
  max_retries: 2                  # top-up calls when a batch comes back short

judge:
  provider: openrouter
  model: openai/gpt-5.1-chat
  max_tokens: 2048
  memo_word_budget: 400           # length the judge is asked to keep the rolling memo within
  max_samples_per_batch: 6        # sample excerpts per batch shown to the judge (0 = all)

probe:
  # No `path:` — the initial probe is trained from --base-training-data
  # (data/highstakes_llama70b_50.jsonl, passed by the runner) using the fields below.
  model: google/gemma-3-27b-it
  layer: 32
  pos_class_label: high-stakes
  neg_class_label: low-stakes
  # BYTE-IDENTICAL to the description every gemma-3-27b high-stakes config in this repo
  # carries. It is read off the PICKLE downstream, not off this file, and it reaches the
  # generator prompt (it is what the generator labels against) and the judge. A bare
  # definition and nothing more: anything about the eval splits belongs in
  # eval.data_description, which the judge sees and the generator does not.
  description: A linear probe on google/gemma-3-27b-it detecting whether the conversation is high-stakes.
  architecture: linear_then_softmax
  # ensemble_size:                # unset — see the header note on the noise floor

loop:
  iterations: 5                   # generate → score → retrain → guide cycles (--iterations overrides)
  min_auroc_gain: 0.0             # batch accepted if mean dev ΔAUROC > this
  exhausted_gain: 0.002           # |Δ| <= this is reported to the judge as an exhausted regime

validation:
  # REQUIRED: the fit's validation set (early stopping) AND the set every batch's ΔAUROC
  # is read on. 1908 rows across four splits, verified disjoint from eval_sets/highstakes.
  # It is resident for every fit and scored every epoch, and this loop fits n_batches + 1
  # = 6 probes per iteration, so its size multiplies through: 1908 rows x 1024 tokens x
  # 5376 hidden x 2 bytes ~= 21 GB of gemma-3-27b activations. The kaggle: dev_* pair
  # below is what stops that being 1908 forward passes through a 27B model as well.
  dev_data: ../dev_samples/highstakes

kaggle:
  # PRECOMPUTED gemma-3-27b activations for BOTH the eval splits and the dev set, so
  # neither ever loads the 27B model. Both template fields are str.format-ed on TWO keys:
  # `split` (the split stem, e.g. mts_balanced) and `slug` (that stem lowercased and
  # hyphenated). Kaggle rejects underscores in a DATASET slug and every stem here has one,
  # so the slug MUST use {slug}; the FILE inside the dataset is unrestricted and stays on
  # {split}. Every blob is validated against the probe's model_name/layer and the split's
  # row count before it may be used. A split that cannot be fetched RAISES — the point of
  # asking for the cache is to avoid the extraction entirely.
  #
  # Requires eval.eval_max_samples: 0 (validated at parse time) and
  # KAGGLE_CONFIG_DIR (the DIRECTORY holding kaggle.json) or KAGGLE_API_TOKEN.
  owner: anku7890
  # eval, resolved:  anku7890/anthropic-hh-balanced-gemmaevalpt   (2984 rows)
  #                  anku7890/mt-balanced-gemmaevalpt             ( 604)
  #                  anku7890/mts-balanced-gemmaevalpt            (  86)
  #                  anku7890/toolace-balanced-gemmaevalpt        ( 734)
  eval_dataset_slug: "{slug}-gemmaevalpt"
  eval_file_name: "{split}-gemmaeval.pt"
  # dev, resolved:   anku7890/anthropic-hh-balanced-gemmadevpt    (1028 rows)
  #                  anku7890/mt-balanced-gemmadevpt              ( 278)
  #                  anku7890/mts-balanced-gemmadevpt             ( 274)
  #                  anku7890/toolace-balanced-gemmadevpt         ( 328)
  # The dev set is used WHOLE, so its activations live in ONE content-hashed blob;
  # prefetch_dev_activations fetches the four splits and concatenates them in
  # sorted(glob("*.jsonl")) order — the same order _load_dev_dataset uses — right-padding
  # to the common width. Note the dev stems collide with the eval stems for this concept
  # (both are anthropic_hh_balanced / mt_balanced / ...); the -gemmadevpt vs -gemmaevalpt
  # suffix is the only thing separating the two families of datasets.
  dev_dataset_slug: "{slug}-gemmadevpt"
  dev_file_name: "{split}-gemmadev.pt"

eval:
  # TRUE/TRUE, not example_generate.md's false/false, and this is load-bearing rather
  # than a preference. These transforms are applied to the training data, the generated
  # samples, the dev set AND the eval splits — one message representation everywhere —
  # and the published activation blobs were extracted under them. The eval cache is keyed
  # by path alone and the dev blob's key covers the transforms, so getting this wrong
  # either reuses mismatched activations outright or silently forces a 27B re-extraction.
  combine_consecutive_messages: true
  convert_tool_to_assistant: true
  eval_max_samples: 0             # full eval splits — REQUIRED by kaggle:
  # data_description: unset, as in example_generate.md and in the other two arms.

output:
  run_dir: ../results_gen_gemma27b_highstakes          # batches.jsonl, guidance.jsonl, runlog.jsonl,
                                                       #   auroc_history.csv, accepted_iter{N}.jsonl
  run_id: null
  comparison_csv: ../results_gen_gemma27b_highstakes/eval_comparison.csv
  # Activation caches, kept OUT of run_dir: they are multi-GB *.pt and are recompute-only,
  # so the failsafe deliberately never commits them. eval blobs arrive from Kaggle; the
  # base blob and the per-sample blobs for generated conversations are computed here.
  activations_cache_dir: ../cache_gen_gemma27b_highstakes/eval_activations
  base_activation_cache_dir: ../cache_gen_gemma27b_highstakes/base_activations
---

# Generator

You write training data for a text classifier that labels conversations between a
user and an AI assistant. You are given the classifier's two class labels and a
definition of the concept they name, a direction describing what this batch should
cover, and — from the second round on — a memo from a reviewer about which kinds of
samples have improved the classifier so far and which have not.

Your job is to write realistic, varied conversations and label each one honestly
according to the concept definition. The classifier learns from your labels, so an
inaccurate label is worse than no sample at all. Within a batch, keep the two classes
matched on everything except the concept itself — similar domains, lengths, tones and
structures on both sides — so the classifier is pushed to learn the concept rather than
a surface cue. Do not reuse scenarios from the memo or from earlier batches; the memo
tells you which regions are worth extending with NEW variations and which are exhausted.

# Judge

You are analyzing the results of a training-set curation experiment for a text
classifier. In each round, several batches of labelled conversations were written under
different directions; for each batch a classifier was trained on the current training
set plus that batch, and its AUROC on a fixed held-out dev set was compared with the
current classifier's. Your write-up is read by the engineers curating the training set
and is handed to the sample writer for the next round.

Reason from the samples themselves: what about the accepted batches' samples taught
the classifier something its training set lacked; what about the exhausted batches'
samples was already covered; whether a harmful batch was mislabelled or simply
confusing. Be concrete and specific — name domains, conversation structures and the
features that separate the two classes there — and prefer new variations of what worked
and unexplored regions over repeats.
