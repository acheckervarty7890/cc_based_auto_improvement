---
generator:
  provider: openrouter            # claude_sdk | openrouter — default for bare-string models below
  models:
    - meta-llama/llama-3.3-70b-instruct
    # - name: claude-sonnet-4-6   # per-model provider override; batch k uses models[k % len]
    #   provider: claude_sdk
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
  # path: ../data/probe_llama1b.pkl   # optional warm start; otherwise trained from the fields below
  model: meta-llama/Llama-3.2-1B-Instruct
  layer: 8
  pos_class_label: high-stakes
  neg_class_label: low-stakes
  description: >
    A conversation is high-stakes when the situation it describes carries significant
    consequences for someone — safety, health, large financial or legal exposure,
    irreversible decisions — and low-stakes when nothing much rides on the outcome.
  architecture: linear_then_softmax
  # ensemble_size: 5              # optional (1..10): fit N probes per step and average their scores

loop:
  iterations: 3                   # generate → score → retrain → guide cycles (--iterations overrides)
  min_auroc_gain: 0.0             # batch accepted if mean dev ΔAUROC > this
  exhausted_gain: 0.002           # |Δ| <= this is reported to the judge as an exhausted regime

validation:
  dev_data: ../dev_samples/highstakes   # REQUIRED: early-stopping set AND ΔAUROC scoring set

eval:                             # message transforms apply to training, dev AND eval data
  combine_consecutive_messages: false
  convert_tool_to_assistant: false
  eval_max_samples: 0             # 0 = full eval splits (--eval)
  # data_description: >           # optional: what the eval splits hold, shown to the judge
  #   Four splits: Anthropic HH-style chats, medical transcripts (mt/mts), tool-use (toolace).

output:
  run_dir: ../results/generate_example   # batches.jsonl, guidance.jsonl, runlog.jsonl, auroc_history.csv
  run_id: null
  # comparison_csv: ../results/generate_example/eval_comparison.csv
  # activations_cache_dir: ../results/eval_activations
  # base_activation_cache_dir: ../probes/base_activation_cache
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
