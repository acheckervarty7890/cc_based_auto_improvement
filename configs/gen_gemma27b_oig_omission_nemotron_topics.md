---
# =============================================================================
# oig_omission — ARM 4: nemotron, with the EVAL SPLIT'S OWN SUBJECT MATTER named in
# probe.description.
#
# THE ONLY DIFFERENCE FROM configs/gen_gemma27b_oig_omission_nemotron.md IS
# probe.description (and the output paths). Same generator, same prompts, same probe,
# same 5-member ensemble, same dev split, same thresholds.
#
# WHY. Arms 1-3 produced 266 conversations about travel planning, tech support,
# customer service, education help and food ordering. The split they are scored on
# contains none of that: it is 57 encyclopedic four-question lookups about named
# entities — people, moth species, counties, listed buildings, football clubs, albums.
# The generated data was not merely mislabelled, it was drawn from a different
# distribution entirely, and probe.description is the only channel that reaches the
# generator at iteration 0 (propose_directions writes the first round's directions from
# it alone, with no memo and no judge).
#
# WHAT THIS IS *NOT* CLAIMING. Topic does not discriminate the label here and cannot:
# the two classes are PAIRED on the same question, so every subject below appears on
# both sides. Naming the topics is meant to fix the INPUT distribution — what kind of
# conversation gets written — not to hand the generator a label cue. The description
# keeps the omission definition first and primary for exactly that reason; if the memo
# starts recommending samples separated by subject matter, that is the failure mode this
# edit risks, and the judge prompt below already tells it to reject such explanations.
#
# The topic list is measured, not guessed: it is the distribution over the 57 distinct
# `original_text` sources of eval_sets/instructions/oig_omission.jsonl.
# =============================================================================
#
# The arm-2 and arm-1 headers follow verbatim.
#
# =============================================================================
# oig_omission — ARM 2: the SAME run with nvidia/nemotron-3-ultra-550b-a55b as the
# generator, in place of meta-llama/llama-3.3-70b-instruct.
#
# THE ONLY DIFFERENCE FROM configs/gen_gemma27b_oig_omission.md IS generator.models
# (and the output paths, so the two runs do not overwrite each other). Every other knob
# — the generator and judge system prompts, n_batches, batch_size, max_tokens, the probe,
# the 5-member ensemble, min_auroc_gain, the dev split, the eval transforms — is
# byte-identical, so a delta reads as "this generator vs that generator" and nothing else.
#
# The activation cache dirs are deliberately SHARED with arm 1. They are content-keyed
# (the eval blob by path, the dev blob and the base blob by a hash of their rows, each
# generated conversation by its own messages), the splits and the probe are identical,
# and the two arms will not generate the same conversations — so sharing costs nothing
# and saves re-extracting the eval, dev and base blobs through a 27B model.
#
# max_tokens 8192 is UNCHANGED and was checked, not assumed: nemotron-3-ultra is a
# reasoning model, so the budget has to cover reasoning tokens as well as the JSON. One
# real batch-of-10 request under this file's own system prompt returned 4851 completion
# tokens (387 of them reasoning), finish_reason stop — comfortably inside 8192.
# =============================================================================
#
# The original arm-1 header follows verbatim.
#
# =============================================================================
# oig_omission — a SMALL, NARROWLY TARGETED run of the generate → score →
# retrain → guide loop.
#
# WHY THIS SPLIT. In generator_experiment_1 the instruction-following arm gained
# +0.042 mean eval AUROC across seven splits while oig_omission FELL 0.7764 →
# 0.6099, the largest single movement of the whole experiment and the wrong way.
# It is also the hardest split anywhere measured: the ceiling study on
# experiment_instruction_cloud_3 puts a probe trained on eval-distribution data
# at only 0.6725 there (5-fold CV, out-of-fold) — the lowest ceiling of any split
# of any concept, and BELOW the 0.7764 the base probe happens to score. So the
# question this run asks is narrow and concrete: with generation aimed at
# omission alone, does the loop move oig_omission up, or does it damage it the
# way the seven-split run did?
#
# WHAT MAKES IT DIFFERENT FROM THE THREE MAIN ARMS. Everything here is scored on
# ONE split. eval_sets/oig_omission/ and dev_samples/oig_omission/ each hold a
# single JSONL copied verbatim from the instruction-following dirs, so the mean
# the loop accepts on IS oig_omission's own AUROC — there are no other splits to
# dilute it. Read the resulting numbers against the instructions arm's
# oig_omission column, not against its mean.
#
# ---------------------------------------------------------------------------
# THE 32-ROW DEV SET IS THE CENTRAL CONSTRAINT — READ THIS BEFORE TUNING.
# ---------------------------------------------------------------------------
# dev_samples/oig_omission/ holds 32 rows, 16 per class. AUROC over 16x16 = 256
# label pairs is a COUNT of correctly-ordered pairs divided by 256, so it can only
# take values 0/256, 1/256, 2/256 ... — it moves in discrete steps of 0.0039, and
# a "gain" of one step is one pair flipping. That granularity is a property of the
# data, not of the fit: no ensemble size, seed or architecture makes it finer.
#
# Two settings follow directly from it and should be changed together, if at all:
#   * min_auroc_gain: 0.0078  = TWO pairs. One pair is indistinguishable from
#     noise; requiring two is the least that can be called a signal here. The
#     three main arms ran min_auroc_gain: 0.0 and, on much larger dev sets, still
#     accepted ~35% of their batches inside the noise floor. Do not copy that 0.0
#     into this config.
#   * exhausted_gain: 0.0039 = ONE pair, i.e. |Δ| at or below one step is reported
#     to the judge as an exhausted regime rather than as evidence either way.
#
# probe.ensemble_size: 5 is the other half of the same problem. Averaging five
# probes does not change the 0.0039 granularity, but it does cut the variance of
# WHERE a refit lands, so a batch that clears two pairs is more likely to have
# cleared them for a reason. It costs 5x the probe-head fit; at 32 dev rows and a
# few hundred training rows that is seconds, not minutes.
#
# Even with both, treat a single accepted batch here as weak evidence. The honest
# read of this run is the shape of the whole 3 x 3 ledger, not any one Δ.
#
# COST. 3 iterations x (3 candidate fits + 1 union retrain) = 12 fits of 5 members
# each, all on cached activations. The 27B extraction LLM is loaded only to
# activate newly generated conversations (<= 90 of them); the eval split, the dev
# split and the base blob all come from cache. Expect well under an hour.
# =============================================================================

generator:
  provider: openrouter
  # The same generator the three main arms used, so the data source is not a
  # confound when reading this against the instructions arm.
  models:
    - nvidia/nemotron-3-ultra-550b-a55b   # <<< THE ONLY FUNCTIONAL DIFFERENCE FROM ARM 1 >>>
  n_batches: 3                    # small: 3 directions per iteration
  batch_size: 10                  # 5 per class
  concurrency: 3
  max_tokens: 8192
  max_sample_tokens: 1024
  max_retries: 2

judge:
  provider: openrouter
  # openai/gpt-5.1-chat (example_generate.md's default) returns 404 "No endpoints
  # found" on OpenRouter; openai/gpt-5.1 is the working model in that family and is
  # what all three main arms used. A dead judge does NOT stop the run — cli.py logs
  # guidance_error and falls back to Generator.propose_directions — so every
  # iteration would generate unsteered, which is the one thing this run is testing.
  model: openai/gpt-5.1
  max_tokens: 2048
  memo_word_budget: 400
  max_samples_per_batch: 6

probe:
  # No `path:` — trained from --base-training-data. The runner passes
  # data/instructions_llama70b_50.jsonl, the SAME 50-row base set the
  # instruction-following arm started from, so iteration 0 is comparable to it.
  # Note it is not identical: the fit early-stops against this run's 32-row
  # omission dev set instead of the 436-row seven-split one, so iteration 0 is the
  # same training data selected at a different epoch.
  model: google/gemma-3-27b-it
  layer: 32
  # The class labels MUST match the strings in the split JSONLs verbatim or the
  # load raises; these are the instruction-following concept's own labels.
  pos_class_label: assistant_follows_the_instruction
  neg_class_label: assistant_does_not_follow_the_instruction
  # NARROWED ON PURPOSE. The instruction-following configs carry a general
  # "does the response follow the instruction" description. Here the probe is
  # trained and scored on omission rows ONLY, so the description says what those
  # rows actually distinguish — a multi-part request answered in full versus one
  # with a part silently dropped. This matters mechanically: at iteration 0 there
  # is no memo and no judge, and Generator.propose_directions writes its own
  # directions from this description alone. A generic description would spend the
  # first of only three iterations on generic instruction-following data.
  description: >
    A conversation where the user asks for several distinct things at once — a
    multi-part question, or a request with more than one requirement — and the
    label is set by whether the assistant's reply addresses every part. It is
    assistant_follows_the_instruction when each requested item is actually
    answered, and assistant_does_not_follow_the_instruction when the reply is
    fluent, on-topic and plausible but silently omits one of the requested parts.
    The omission is the whole signal: a dropped part is never refused, flagged or
    mentioned, it is simply absent from an otherwise good answer.

    These conversations are encyclopedic factual lookups but they are short and precise. 
    The user names one subject and asks three or four short questions about it in a single message,
    sometimes after pasting a short Background passage to answer from. The subjects
    are: biographies of individual people — politicians, athletes, musicians, actors, 
    writers, academics; individual animal and plant species, moths especially, given by
    binomial name; towns, villages, counties and other administrative places;
    listed buildings, mansions and historic architecture; albums, singles and bands;
    football clubs, arenas and sporting competitions; films; schools and colleges;
    military regiments and naval vessels; companies and commercial products.
  architecture: linear_then_softmax
  # See the header: 5 members to damp refit variance on a 32-row dev set.
  ensemble_size: 5

loop:
  iterations: 1
  # 0.0078 = two of the 256 label pairs. See the header — this is NOT the 0.0 the
  # main arms used, and copying that value here would make acceptance a coin flip.
  min_auroc_gain: 0.0078
  # 0.0039 = one pair; at or below this the judge is told the regime is exhausted.
  exhausted_gain: 0.0039

validation:
  # 32 rows, 16/16. Verified disjoint from eval_sets/oig_omission (0 shared
  # conversations) — a dev row that also sat in eval would mean both the
  # best-epoch checkpoint AND every acceptance decision were made on the test set.
  dev_data: ../dev_samples/oig_omission

eval:
  # Byte-identical to the three main arms, and applied to training, dev AND eval
  # data alike, so the probe trains and is scored on the same message
  # representation. These flags are part of the activation cache key.
  combine_consecutive_messages: true
  convert_tool_to_assistant: true
  eval_max_samples: 0             # full split — REQUIRED by kaggle:
  # data_description is deliberately NOT set. It renders as a single bullet in the
  # user message of the judge's guide() call and nowhere else — not the judge's
  # system prompt, not the generator, no cache key — so with one eval split there is
  # no coverage tradeoff for it to steer. What it would have said about the paired
  # structure of this split is stated in the `# Judge` system prompt below instead,
  # which is sent on every call.

kaggle:
  # Precomputed gemma-3-27b activations for the eval split and the dev split, so
  # neither loads the 27B model. Both stems are `oig_omission`, which contains an
  # underscore — Kaggle rejects underscores in a DATASET slug, hence {slug}
  # (= oig-omission) there; the FILE inside the dataset is unrestricted and stays
  # on {split}. Every blob is validated against probe.model / probe.layer and the
  # split's row count before it may be used, and a split that cannot be fetched
  # RAISES rather than silently extracting.
  #
  # Requires eval.eval_max_samples: 0 (validated at parse time) and
  # KAGGLE_CONFIG_DIR (the DIRECTORY holding kaggle.json) or KAGGLE_API_TOKEN.
  owner: anku7890
  # eval, resolved: anku7890/oig-omission-gemmaevalpt : oig_omission-gemmaeval.pt
  eval_dataset_slug: "{slug}-gemmaevalpt"
  eval_file_name: "{split}-gemmaeval.pt"
  # dev, resolved:  anku7890/oig-omission-gemmadevpt  : oig_omission-gemmadev.pt
  dev_dataset_slug: "{slug}-gemmadevpt"
  dev_file_name: "{split}-gemmadev.pt"

output:
  run_dir: ../results_oig_omission_nemotron_topics
  run_id: null
  comparison_csv: ../results_oig_omission_nemotron_topics/eval_comparison.csv
  activations_cache_dir: ../cache_oig_omission/eval_activations
  base_activation_cache_dir: ../cache_oig_omission/base_activations
---

# Generator

You write training data for a text classifier that labels conversations between a
user and an AI assistant. You are given the classifier's two class labels and a
definition of the concept they name, a direction describing what this batch should
cover, and — from the second round on — a memo from a reviewer about which kinds of
samples have improved the classifier so far and which have not.

Your job is to write realistic, varied conversations and label each one honestly
according to the concept definition. The classifier learns from your labels, so an
inaccurate label is worse than no sample at all.

This concept is one of completeness, not of quality, topic or tone, and that shapes
how a batch has to be built. Read the concept from probe description.

Vary what makes the request multi-part across the batch: number of questions in one
message, a list of items each needing its own treatment. Vary which part gets
dropped — first, middle, last. Do not
reuse scenarios from the memo or from earlier batches; the memo tells you which
regions are worth extending with NEW variations and which are exhausted.

# Judge

You are analyzing the results of a training-set curation experiment for a text
classifier. In each round, several batches of labelled conversations were written
under different directions; for each batch a classifier was trained on the current
training set plus that batch, and its AUROC on a fixed held-out dev set was compared
with the current classifier's. Your write-up is read by the engineers curating the
training set and is handed to the sample writer for the next round.

Reason from the samples themselves: what about the accepted batches' samples taught
the classifier something, and what about the rejected ones did not. Be concrete
about the structure of the conversations, not just their subject matter. Probe description hints 
the topics to look from. 
