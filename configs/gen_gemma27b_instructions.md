---
# =============================================================================
# generator_experiment_1 — ARM 3 of 3: INSTRUCTION FOLLOWING
#
# The generate -> score -> retrain -> guide loop run against a probe on
# google/gemma-3-27b-it, layer 32, for FIVE iterations.
#
# IDENTICAL TO ARM 1 (configs/gen_gemma27b_highstakes.md) except for the concept: the
# class labels, probe.description, validation.dev_data, the kaggle: block's resolved
# refs, the base training data (data/instructions_llama70b_50.jsonl, passed by the
# runner) and the output paths. Every generator / judge / loop knob is byte-identical
# across the three arms and copied verbatim from configs/example_generate.md — read arm
# 1's header for why each of the four departures from that file exists.
#
# ONE THING TO EXPECT HERE THAT THE OTHER TWO ARMS DO NOT HAVE: this concept is
# ASSISTANT-CENTRIC. The label is a property of the assistant's reply given the user's
# instruction, not of the scenario, so a generated pair that varies the user turn is
# varying the wrong half. The generator prompt (shared, unchanged) only says "keep the
# two classes matched on everything except the concept itself"; if the judge's first
# memo does not work that out on its own, that is a finding about the loop, not a bug.
#
# NOT SET, deliberately: probe.ensemble_size — every fit here is a SINGLE probe, as in
# example_generate.md. A single fit's dev AUROC moves ~+/-0.005 between near-identical
# training sets and min_auroc_gain is 0.0, so expect some accepted batches to be noise.
# This arm's dev set is 436 rows, the middle of the three.
# =============================================================================

generator:
  provider: openrouter
  models:
    - meta-llama/llama-3.3-70b-instruct
  n_batches: 5
  batch_size: 10
  concurrency: 5
  max_tokens: 8192
  max_sample_tokens: 1024
  max_retries: 2

judge:
  provider: openrouter
  model: openai/gpt-5.1-chat
  max_tokens: 2048
  memo_word_budget: 400
  max_samples_per_batch: 6

probe:
  # No `path:` — trained from --base-training-data (data/instructions_llama70b_50.jsonl).
  model: google/gemma-3-27b-it
  layer: 32
  pos_class_label: assistant_follows_the_instruction
  neg_class_label: assistant_does_not_follow_the_instruction
  # BYTE-IDENTICAL to the description every gemma-3-27b instruction config in this repo
  # carries. Read off the pickle downstream; it is what the generator labels against.
  description: A linear probe on google/gemma-3-27b-it that reads a conversation between a user and an assistant and detects whether the assistant's response follows the user's instruction.
  architecture: linear_then_softmax
  # ensemble_size:                # unset — see the header note

loop:
  iterations: 5
  min_auroc_gain: 0.0
  exhausted_gain: 0.002

validation:
  # 436 rows across seven splits, verified disjoint from eval_sets/instructions.
  dev_data: ../dev_samples/instructions

kaggle:
  # PRECOMPUTED gemma-3-27b activations for the eval splits AND the dev set. {slug} in
  # the dataset slug is mandatory (every stem here has an underscore, which Kaggle rejects
  # in a slug); {split} names the file inside, which is unrestricted. Blobs are validated
  # against the probe's model_name/layer and each split's row count; a split that cannot
  # be fetched RAISES rather than falling back to extraction. Requires
  # eval.eval_max_samples: 0 and Kaggle credentials.
  owner: anku7890
  # eval, resolved:  anku7890/anthropic-harmless-refusal-gemmaevalpt (200 rows)
  #                  anku7890/bbq-substitution-gemmaevalpt           (200)
  #                  anku7890/hc-context-drift-gemmaevalpt           (194)
  #                  anku7890/hc-contradiction-gemmaevalpt           (200)
  #                  anku7890/mm-substitution-gemmaevalpt            (200)
  #                  anku7890/oig-context-drift-gemmaevalpt          (194)
  #                  anku7890/oig-omission-gemmaevalpt               (114)
  eval_dataset_slug: "{slug}-gemmaevalpt"
  eval_file_name: "{split}-gemmaeval.pt"
  # dev, resolved:   the SAME seven stems under -gemmadevpt / -gemmadev.pt
  #                  (68 / 68 / 66 / 68 / 68 / 66 / 32 rows, 436 total).
  # Fetched per split and concatenated into the one content-hashed dev blob in
  # sorted(glob("*.jsonl")) order — the same order _load_dev_dataset uses — right-padded
  # to the common width (these splits pad to very different lengths, 159 to 436, and the
  # appended columns are inert because their attention-mask entries are zero). Note the
  # dev stems are IDENTICAL to the eval stems for this concept, so the -gemmadevpt vs
  # -gemmaevalpt suffix is the only thing separating the two families of datasets.
  dev_dataset_slug: "{slug}-gemmadevpt"
  dev_file_name: "{split}-gemmadev.pt"

eval:
  # TRUE/TRUE is required, not preferred: the published blobs were extracted under these
  # transforms, the eval cache is keyed by path alone, and the transforms are applied to
  # training data, generated samples, dev set and eval splits alike.
  combine_consecutive_messages: true
  convert_tool_to_assistant: true
  eval_max_samples: 0             # full eval splits — REQUIRED by kaggle:
  # data_description: unset, as in example_generate.md and the other two arms.

output:
  run_dir: ../results_gen_gemma27b_instructions
  run_id: null
  comparison_csv: ../results_gen_gemma27b_instructions/eval_comparison.csv
  activations_cache_dir: ../cache_gen_gemma27b_instructions/eval_activations
  base_activation_cache_dir: ../cache_gen_gemma27b_instructions/base_activations
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
