#!/usr/bin/env python
"""Are the v1 / v2 / v3 vintages made of genuinely different conversations?

The vintage sweep (``attribution_vintage.py``, ``results_hs_gemma27b_batch_ablation/
vintage/``) fits a probe on ``v1 ⊂ v2 ⊂ v3`` — the iteration-3 red-team pairs whose
source success already existed at iteration 1, 2, 3 — and reads an AUROC curve off it.
Every reading of that curve ("iteration 2 bought the most", "iteration 3 regressed")
assumes the *increments* are new material. If iteration 3 mostly re-found what
iteration 1 already had, the curve is measuring set size, not new coverage.

This script tests that assumption on the conversations themselves. No activations and
no LLM are needed — the dumps, the contrastive cache and the raw attempt logs are all
that is read.

Cohorts
-------
The unit is the **pair**, i.e. the red-team success (both its rows follow it), same as
the sweep. ``first_seen[key] = min k such that the source appears in
redteam_postprocessed_iter{k}.jsonl``; the universe is iteration 3's dump. So

    C1 = v1,   C2 = v2 \\ v1,   C3 = v3 \\ v2

and the sweep's vintages are the prefixes ``C1``, ``C1+C2``, ``C1+C2+C3``. (The
sweep's vintages are not strictly nested — a handful of iteration-1 sources are absent
from the iteration-2 dump because ``filter_dataset`` refits each cycle — so ``C1`` is
defined by *first* appearance, which is what "when was this conversation found"
means.)

Four measurements, weakest-assumption first
-------------------------------------------
1. **Near-duplication** — ``difflib.SequenceMatcher(autojunk=False)`` on the first 600
   chars of the first user turn: the repo's own clone metric (``clone_rate.py``, and
   the submit-time ``near_dup_guard`` these arms ran with at tau=0.8). For each pair,
   its nearest neighbour among *earlier* cohorts, and among its own.
2. **Lexical nearest neighbour** — TF-IDF cosine over the whole conversation, word
   1-2grams and char 4-grams. Catches the re-skin difflib misses (same scenario,
   reordered), and is order-insensitive.
3. **Distributional separability** — can a bag-of-words classifier tell cohort A from
   cohort B? 5-fold stratified CV AUROC, against a **null built from the same data**:
   the same fit on random splits of the pooled cohorts, which is the only honest
   baseline at n~150 and 20k features. AUROC well above that null = the cohorts are
   drawn from measurably different regions; AUROC at the null = exchangeable.
4. **Provenance** — the raw attempt logs joined by canonical text: which error type
   each success came from, what the probe scored it, what the judge said. Two cohorts
   can be lexically similar and still be different *attacks*.

Anchors, so the similarity numbers have a scale
-----------------------------------------------
Bare cosines are unreadable without knowing what "same" and "different" look like on
this data. Three references are computed from the same pipeline:

* ``self-pair`` — each success against the contrastive counterpart written *from it*.
  Under prompt v2 that is a deliberate minimal edit, i.e. the same conversation.
* ``base-data`` — ``data/hs_ls_200.jsonl`` against itself: independently collected
  conversations of the same concept.
* ``cross-arm`` — one arm's successes against the other arm's. Two different attacker
  models, same probe and same prompt: the amount of convergence you get from the task
  alone, with no shared history at all.

Usage:
    .venv_claude/bin/python scripts/vintage_cohort_novelty.py
    .venv_claude/bin/python scripts/vintage_cohort_novelty.py --arms deepseekv4pro
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_lib as A

REPO = A.REPO

NEAR_DUP_PREFIX = 600  # persistence._NEAR_DUP_PREFIX
ITERATIONS = 3

# The difflib nearest-neighbour passes are the whole runtime of this script: O(n*m)
# SequenceMatcher over ~300 x ~300 conversations, several times per arm. It is pure
# CPU in C-less Python, so processes (not threads) are the lever.
_WORKERS = int(os.environ.get("AGENTIC_COHORT_WORKERS", str(min(16, os.cpu_count() or 1))))
_PARALLEL_MIN_PAIRS = 20_000  # below this the pool/pickle overhead dominates


# --- experiments ---------------------------------------------------------------------
#
# One entry per run this can be pointed at. Every arm here used the same loader
# transforms (combine + convert, both on), the same submit-time near-dup guard (tau 0.8)
# and the same 3-iteration batch-submission schedule, so the cohorts are constructed
# identically; only the concept, the attacker models and the paths differ.


@dataclass(frozen=True)
class Experiment:
    name: str
    root: Path                       # dir the arm/log paths below are relative to
    concept: str                     # human-readable, for the report
    pos_label: str
    neg_label: str
    assistant_centric: bool          # does the label live in the assistant's reply?
    arms: dict[str, str]             # arm key -> probe dir
    raw_logs: dict[str, list[str]]   # arm key -> [fp log, fn log]
    labels: dict[str, str]           # arm key -> display name
    base_training_data: str | None = None
    combine_consecutive_messages: bool = True
    convert_tool_to_assistant: bool = True

    def arm_dir(self, arm: str) -> Path:
        return self.root / self.arms[arm]

    def logs(self, arm: str) -> list[Path]:
        return [self.root / p for p in self.raw_logs[arm]]


SCRATCH = Path(
    os.environ.get(
        "AGENTIC_COHORT_SCRATCH",
        "/tmp/claude-1000/-home-ankush-Documents-cc-based-auto-improvement/"
        "89869c95-7b42-4739-8822-700ca0e59f50/scratchpad",
    )
)

EXPERIMENTS: dict[str, Experiment] = {
    # experiment9_cloud — this checkout.
    "hs": Experiment(
        name="hs",
        root=REPO,
        concept="high-stakes",
        pos_label="high-stakes",
        neg_label="low-stakes",
        assistant_centric=False,
        arms={
            "deepseekv4pro": "probes/hs_gemma27b_deepseekv4pro_batch",
            "gptoss120b": "probes/hs_gemma27b_gptoss120b_batch",
        },
        raw_logs={
            "deepseekv4pro": [
                "results_hs_gemma27b_deepseekv4pro_batch/deepseekv4pro_probing_fp.jsonl",
                "results_hs_gemma27b_deepseekv4pro_batch/deepseekv4pro_probing_fn.jsonl",
            ],
            "gptoss120b": [
                "results_hs_gemma27b_gptoss120b_batch/gptoss120b_probing_fp.jsonl",
                "results_hs_gemma27b_gptoss120b_batch/gptoss120b_probing_fn.jsonl",
            ],
        },
        labels={"deepseekv4pro": "deepseek-v4-pro", "gptoss120b": "gpt-oss-120b"},
        base_training_data="data/hs_ls_200.jsonl",
    ),
    # experiment11_cloud — human-harm, same two attackers.
    "hu_harm": Experiment(
        name="hu_harm",
        root=SCRATCH / "exp11",
        concept="harmful-to-human",
        pos_label="harmful_to_human",
        neg_label="not_harmful_to_human",
        assistant_centric=True,
        arms={
            "deepseekv4pro": "probes/hu_harm_gemma27b_deepseekv4pro_batch",
            "gptoss120b": "probes/hu_harm_gemma27b_gptoss120b_batch",
        },
        raw_logs={
            "deepseekv4pro": [
                "results_hu_harm_gemma27b_deepseekv4pro_batch/deepseekv4pro_probing_fp.jsonl",
                "results_hu_harm_gemma27b_deepseekv4pro_batch/deepseekv4pro_probing_fn.jsonl",
            ],
            "gptoss120b": [
                "results_hu_harm_gemma27b_gptoss120b_batch/gptoss120b_probing_fp.jsonl",
                "results_hu_harm_gemma27b_gptoss120b_batch/gptoss120b_probing_fn.jsonl",
            ],
        },
        labels={"deepseekv4pro": "deepseek-v4-pro", "gptoss120b": "gpt-oss-120b"},
    ),
    # experiment_instruction_cloud_1 — instruction-following, different attacker pair.
    "instructions": Experiment(
        name="instructions",
        root=SCRATCH / "instr",
        concept="instruction-following",
        pos_label="assistant_follows_the_instruction",
        neg_label="assistant_does_not_follow_the_instruction",
        assistant_centric=True,
        arms={
            "gptoss120b": "probes/instructions_gemma27b_gptoss",
            "nemotron": "probes/instructions_gemma27b_nemotron",
        },
        raw_logs={
            "gptoss120b": [
                "results_instructions_gemma27b_gptoss/gptoss120b_probing_fp.jsonl",
                "results_instructions_gemma27b_gptoss/gptoss120b_probing_fn.jsonl",
            ],
            "nemotron": [
                "results_instructions_gemma27b_nemotron/nemotron_probing_fp.jsonl",
                "results_instructions_gemma27b_nemotron/nemotron_probing_fn.jsonl",
            ],
        },
        labels={"gptoss120b": "gpt-oss-120b", "nemotron": "nemotron-3-ultra"},
    ),
}


# --- text ---------------------------------------------------------------------------


def first_user_text(messages) -> str:
    for m in messages:
        if m.role == "user":
            return (m.content or "").strip()
    return (messages[0].content or "").strip() if messages else ""


def full_text(messages) -> str:
    return "\n".join(f"{m.role}: {m.content or ''}" for m in messages)


def canon(messages) -> str:
    """Canonical text of a conversation — the join key, matching ``attribution_lib``."""
    return A.canon(messages)


def apply_transforms(dialogue, exp: "Experiment"):
    """The run's two loader transforms, in ``LabelledDataset.load_from`` order."""
    from tuberlens.interfaces.dataset import LabelledDataset

    if exp.convert_tool_to_assistant:
        dialogue = LabelledDataset._convert_tool_to_assistant(dialogue)
    if exp.combine_consecutive_messages:
        dialogue = LabelledDataset._combine_consecutive_messages(dialogue)
    return dialogue


def assistant_text(messages) -> str:
    """The assistant side only.

    For an assistant-centric concept (harm, instruction-following) this is where the
    label lives, so a clone of the *reply* matters as much as a clone of the prompt —
    and the submit-time guard, which only ever looks at the first user turn, cannot see
    it. Concatenated because a conversation can carry several assistant turns.
    """
    return "\n".join(m.content or "" for m in messages if m.role == "assistant")


def content_text(messages) -> str:
    """Message contents with the role prefixes dropped.

    ``full_text`` prefixes every message with its role, which is right for the
    similarity metrics (two conversations differing only in who said what are not the
    same conversation) but wrong for topic clustering: "user" and "assistant" then
    appear in every document and surface as cluster-defining terms.
    """
    return "\n".join(m.content or "" for m in messages)


# Function words carry register and length, not subject matter, so they are removed
# before clustering. sklearn's list misses the fragments its own tokenizer produces
# from contractions ("don't" -> "don", "t"), which showed up as cluster terms.
_EXTRA_STOP = {
    "ll", "ve", "re", "s", "t", "m", "d", "don", "didn", "doesn", "isn", "wasn",
    "aren", "weren", "couldn", "wouldn", "shouldn", "won", "can", "cant", "im",
    "id", "ive", "thats", "youre", "theyre", "gonna", "just", "like", "really",
    "know", "need", "want", "think", "going", "got", "get", "make", "way", "thing",
    "things", "lot", "sure", "okay", "ok", "yeah", "hi", "hello", "thanks",
    "user", "assistant", "system",
}


def topic_stop_words() -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return sorted(set(ENGLISH_STOP_WORDS) | _EXTRA_STOP)


def difflib_ratio(a: str, b: str) -> float:
    """Matches ``persistence._is_near``: autojunk off, so the metric is symmetric."""
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def nn_difflib(queries: list[str], pool: list[str], *, same_set: bool) -> list[float]:
    """Max difflib ratio of each query against ``pool`` (excluding itself if same set).

    Identical results to the naive double loop, two constant-factor tricks:

    * **Pool item in seq2, iterated outermost.** ``SequenceMatcher`` caches the index of
      its *second* sequence and rebuilds it on every ``set_seq2``; ``set_seq1`` is cheap.
      Looping pool-outer therefore builds each index once instead of once per pair.
    * **Prune on the upper bounds.** ``real_quick_ratio`` and ``quick_ratio`` bound
      ``ratio`` from above, so a candidate that cannot beat the best seen for that query
      is skipped before the O(n*m) matcher runs. We only want the maximum, so this
      cannot change the answer.

    Argument order matches ``difflib_ratio(query, pool_item)``. That matters even with
    ``autojunk=False``, since ``ratio`` is only symmetric once autojunk is off.
    """
    if not queries or not pool:
        return [0.0] * len(queries)
    if _WORKERS > 1 and len(queries) * len(pool) >= _PARALLEL_MIN_PAIRS:
        return _nn_difflib_parallel(queries, pool, same_set=same_set)
    return _nn_difflib_chunk(queries, pool, same_set, 0, len(pool))


def _nn_difflib_chunk(queries, pool, same_set, lo: int, hi: int) -> list[float]:
    """The maximum over ``pool[lo:hi]`` only — a partial answer, max-combinable."""
    best = [0.0] * len(queries)
    sm = difflib.SequenceMatcher(None, autojunk=False)
    for j in range(lo, hi):
        sm.set_seq2(pool[j])
        for i, q in enumerate(queries):
            if same_set and i == j:
                continue
            sm.set_seq1(q)
            b = best[i]
            if sm.real_quick_ratio() <= b or sm.quick_ratio() <= b:
                continue
            r = sm.ratio()
            if r > b:
                best[i] = r
    return best


def _nn_difflib_parallel(queries, pool, *, same_set: bool) -> list[float]:
    """Same maximum, computed over pool slices in parallel.

    Splitting the *pool* keeps each worker's answer a partial maximum over a disjoint
    slice, so combining is an elementwise max and the result is exact. Splitting the
    queries instead would work too, but the pool is what the ``same_set`` index test is
    keyed on, and slicing it keeps that test a plain ``i == j`` on absolute indices.
    """
    from concurrent.futures import ProcessPoolExecutor

    n = len(pool)
    step = max(1, (n + _WORKERS - 1) // _WORKERS)
    spans = [(i, min(i + step, n)) for i in range(0, n, step)]
    best = [0.0] * len(queries)
    with ProcessPoolExecutor(max_workers=min(_WORKERS, len(spans))) as ex:
        for partial in ex.map(
            _nn_difflib_chunk,
            *zip(*[(queries, pool, same_set, lo, hi) for lo, hi in spans]),
        ):
            for i, v in enumerate(partial):
                if v > best[i]:
                    best[i] = v
    return best


# --- cohorts ------------------------------------------------------------------------


def generated_to_source(exp: "Experiment", arm: str) -> dict[str, str]:
    """Each generated row's canonical text -> the success it was written from.

    ``attribution_lib.generated_to_source`` does this against its own hardcoded arms and
    transform constants; this is the same logic driven by the experiment spec.
    """
    from tuberlens.interfaces.dataset import Message

    def as_key(raw):
        return canon(
            apply_transforms(
                [Message(role=m["role"], content=m["content"]) for m in raw], exp
            )
        )

    out: dict[str, str] = {}
    with (exp.arm_dir(arm) / "contrastive_cache.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line).get("record", {})
            if not rec.get("inputs") or not rec.get("original_messages"):
                continue
            out[as_key(rec["inputs"])] = as_key(rec["original_messages"])
    return out


def load_dump_rows(exp: "Experiment", arm: str, iteration: int):
    """``[(messages, label)]`` of one postprocessed dump."""
    from tuberlens.interfaces.dataset import Message

    path = exp.arm_dir(arm) / f"redteam_postprocessed_iter{iteration}.jsonl"
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows.append(
                (
                    [Message(role=m["role"], content=m["content"]) for m in r["inputs"]],
                    r["label"],
                )
            )
    return rows


def build_cohorts(exp: "Experiment", arm: str):
    """``(cohort_of_key, pairs)`` — pairs keyed by source canonical text.

    ``pairs[key] = {"source": messages, "generated": messages|None, "cohort": k}``
    over the iteration-3 universe.
    """
    gen2src = generated_to_source(exp, arm)

    dump_sources: list[set[str]] = []
    universe: dict[str, dict] = {}
    for k in range(1, ITERATIONS + 1):
        rows = load_dump_rows(exp, arm, k)
        srcs = set()
        for messages, label in rows:
            key = canon(messages)
            src_key = gen2src.get(key, key)
            srcs.add(src_key)
            if k != ITERATIONS:
                continue
            slot = universe.setdefault(
                src_key, {"source": None, "generated": None, "labels": {}}
            )
            if key in gen2src:
                slot["generated"] = messages
                slot["labels"]["generated"] = label
            else:
                slot["source"] = messages
                slot["labels"]["source"] = label
        dump_sources.append(srcs)

    for key, slot in universe.items():
        slot["cohort"] = next(
            (k for k in range(1, ITERATIONS + 1) if key in dump_sources[k - 1]),
            ITERATIONS,
        )
    return universe, dump_sources


# --- provenance ---------------------------------------------------------------------


def load_raw_successes(exp: "Experiment", arm: str) -> dict[str, dict]:
    """Canonical text (post-transform) -> the attempt row, for successful attempts."""
    from tuberlens.interfaces.dataset import Message

    out: dict[str, dict] = {}
    for path in exp.logs(arm):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                if not d.get("success"):
                    continue
                sample = d["sample"]
                raw_msgs = sample["messages"] if isinstance(sample, dict) else sample
                msgs = apply_transforms(
                    [Message(role=m["role"], content=m["content"]) for m in raw_msgs],
                    exp,
                )
                out[canon(msgs)] = d
    return out


def assistant_side_block(src_asst: dict[str, str], keys, cohort, by_cohort) -> dict:
    """Near-duplication measured on the assistant side.

    **Conversations with no assistant turn are excluded**, and that is load-bearing:
    ``difflib`` defines the ratio of two empty sequences as **1.0**, so every pair of
    reply-less conversations reads as a perfect duplicate. 16% of the human-harm
    successes have no assistant turn, which alone produced 58 spurious "duplicates" in
    one cohort before this filter. The count of excluded rows is reported rather than
    silently dropped — for an assistant-centric concept, a success with no reply to
    judge is itself worth knowing about.
    """
    have = [k for k in keys if src_asst[k].strip()]
    out: dict = {
        "n_with_assistant_turn": len(have),
        "n_without_assistant_turn": len(keys) - len(have),
    }
    if not have:
        return out
    keep = set(have)
    for c in (1, 2, 3):
        cur = [k for k in by_cohort[c] if k in keep]
        earlier = [k for k in have if cohort[k] < c]
        out[f"C{c}_vs_earlier"] = describe(
            nn_difflib([src_asst[k] for k in cur],
                       [src_asst[k] for k in earlier], same_set=False)
        )
        out[f"C{c}_within"] = describe(
            nn_difflib([src_asst[k] for k in cur],
                       [src_asst[k] for k in cur], same_set=True)
        )
    out["all_within"] = describe(
        nn_difflib([src_asst[k] for k in have], [src_asst[k] for k in have],
                   same_set=True)
    )
    return out


def cross_cohort_duplicates(universe, keys, cohort, by_cohort, raw,
                            threshold: float = 0.8) -> dict:
    """Every cross-cohort opener pair at or above the guard's own threshold.

    The guard runs at submit time against the successes **of one error type**, because
    each hunt writes its own JSONL and gets its own store. So a candidate whose opener
    clones a success from the *other* hunt is something it structurally cannot see. This
    reports each such pair with both error types and with the similarity of the
    **assistant** sides, which decides what the pair actually is:

    * replies diverge -> a matched minimal contrast (same prompt, compliant vs not),
      which for an assistant-centric concept is desirable training data;
    * replies also near-identical -> two near-identical rows carrying **opposite**
      labels, which is the unlearnable-twin failure mode.
    """
    op = {k: first_user_text(universe[k]["source"])[:NEAR_DUP_PREFIX] for k in keys}
    asst = {k: assistant_text(universe[k]["source"])[:NEAR_DUP_PREFIX] for k in keys}
    et = {k: raw[k]["error_type"] for k in keys if k in raw}

    pairs = []
    for c in (2, 3):
        cur = by_cohort[c]
        earlier = [k for k in keys if cohort[k] < c]
        if not cur or not earlier:
            continue
        # One vectorised pass finds which candidates have any hit at all; only those
        # few are then compared exhaustively to recover the partner.
        best = nn_difflib([op[k] for k in cur], [op[k] for k in earlier],
                          same_set=False)
        for q, b in zip(cur, best):
            if b < threshold:
                continue
            for pk in earlier:
                r = difflib_ratio(op[q], op[pk])
                if r < threshold:
                    continue
                ra = (difflib_ratio(asst[q], asst[pk])
                      if asst[q].strip() and asst[pk].strip() else None)
                pairs.append({
                    "opener_similarity": round(r, 4),
                    "assistant_similarity": None if ra is None else round(ra, 4),
                    "cohort_new": c,
                    "cohort_earlier": cohort[pk],
                    "error_type_new": et.get(q),
                    "error_type_earlier": et.get(pk),
                    "same_error_type": et.get(q) == et.get(pk),
                })

    same = sum(1 for p in pairs if p["same_error_type"])
    twins = [p for p in pairs
             if p["assistant_similarity"] is not None
             and p["assistant_similarity"] >= threshold]
    return {
        "threshold": threshold,
        "n_pairs": len(pairs),
        "n_same_error_type": same,
        "n_cross_error_type": len(pairs) - same,
        # Near-identical on BOTH sides while carrying opposite labels.
        "n_contradictory_twins": len(twins),
        "pairs": pairs,
    }


def guard_stats(exp: "Experiment", arm: str) -> dict:
    """How often the submit-time near-dup guard fired, by iteration.

    This is the control on measurement 1. A run whose guard never fires reports an
    *organic* absence of clones; a run whose guard fires constantly reports an
    *enforced* one, and the two mean completely different things about the attacker.
    Guard rejections never reach the JSONL by design (``persistence`` keeps them in an
    in-memory ring), so the runlog is the only record — and it is written only when the
    guard actually rejects something, so an absent event type means zero, not missing.
    """
    per_iter: Counter = Counter()
    attempts: Counter = Counter()
    for path in exp.logs(arm):
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    attempts[json.loads(line).get("iteration")] += 1
        runlog = path.with_suffix("").with_suffix(".runlog.jsonl")
        if not runlog.exists():
            runlog = Path(str(path)[: -len(".jsonl")] + ".runlog.jsonl")
        if not runlog.exists():
            continue
        with runlog.open(encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                if d.get("event") == "near_dup":
                    per_iter[d.get("iteration")] += 1

    n_rej = sum(per_iter.values())
    n_att = sum(attempts.values())
    return {
        "n_rejected": n_rej,
        "n_recorded_attempts": n_att,
        # Rejections are turned away *before* probe/judge, so they are not among the
        # recorded attempts — the denominator is the two added together.
        "n_submissions": n_att + n_rej,
        "reject_rate": n_rej / (n_att + n_rej) if (n_att + n_rej) else 0.0,
        "rejected_by_iteration": {str(k): v for k, v in sorted(per_iter.items(),
                                                              key=lambda kv: str(kv[0]))},
        "iterations_with_rejections": sorted(
            {int(k) for k in per_iter if k is not None}
        ),
    }


# --- statistics ---------------------------------------------------------------------


def describe(values) -> dict:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "max": float(v.max()),
        "frac_ge_0.8": float((v >= 0.8).mean()),
        "frac_ge_0.7": float((v >= 0.7).mean()),
        "frac_ge_0.6": float((v >= 0.6).mean()),
    }


def tfidf_matrices(texts: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    wv = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    word = wv.fit_transform(texts)
    vocab = wv.get_feature_names_out()
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(4, 4), min_df=2, sublinear_tf=True
    ).fit_transform(texts)
    from sklearn.preprocessing import normalize

    return normalize(word), normalize(char), vocab


def topic_matrix(texts: list[str]):
    """TF-IDF for clustering: content words only, so clusters name subjects.

    Separate from :func:`tfidf_matrices`, which keeps every token because the
    similarity and separability measurements are about how alike two conversations
    are *as written* — register and function-word habits included.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True,
        stop_words=topic_stop_words(),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # drop bare digits and 1-char tokens
    )
    M = vec.fit_transform(texts)
    return normalize(M), vec.get_feature_names_out()


def nn_cosine(M, rows_q: list[int], rows_pool: list[int], *, same_set: bool):
    """Max cosine of each query row against the pool rows."""
    if not rows_q or not rows_pool:
        return []
    S = (M[rows_q] @ M[rows_pool].T).toarray()
    if same_set:
        for i, r in enumerate(rows_q):
            for j, c in enumerate(rows_pool):
                if r == c:
                    S[i, j] = -1.0
    return S.max(axis=1).tolist()


def cv_auroc(X, y, *, seed: int = 0) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    if len(set(y.tolist())) < 2 or min(Counter(y.tolist()).values()) < 5:
        return float("nan")
    scores = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.zeros(len(y)), y):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        scores[te] = clf.decision_function(X[te])
    return float(roc_auc_score(y, scores))


def separability(M, rows_a: list[int], rows_b: list[int], *, n_null: int = 20) -> dict:
    """CV AUROC for A-vs-B, plus the null from random splits of the same pool.

    The null is the point of the function: at n~150 and ~20k features a linear model
    can fit noise, so "AUROC 0.62" means nothing until you know what AUROC the same
    fit gets separating two random halves of the *same* cohort pool.
    """
    rows = rows_a + rows_b
    X = M[rows]
    y = np.array([0] * len(rows_a) + [1] * len(rows_b))
    obs = cv_auroc(X, y, seed=0)

    rng = np.random.default_rng(0)
    null = []
    for _ in range(n_null):
        perm = rng.permutation(len(y))
        null.append(cv_auroc(X, y[perm], seed=int(rng.integers(1 << 30))))
    null = np.array([v for v in null if np.isfinite(v)])
    return {
        "auroc": obs,
        "null_mean": float(null.mean()) if null.size else float("nan"),
        "null_p95": float(np.percentile(null, 95)) if null.size else float("nan"),
        "n_a": len(rows_a),
        "n_b": len(rows_b),
        "p_value": float((null >= obs).mean()) if null.size else float("nan"),
    }


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(x, y):
        mask = x > 0
        return float((x[mask] * np.log2(x[mask] / y[mask])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def topic_contingency(M, rows_by_cohort: dict[int, list[int]], *, k: int = 15,
                      seed: int = 0, n_null: int = 200, vocab=None) -> dict:
    """Cluster every source, then ask whether the cohorts occupy the same clusters.

    Lexical distance says whether two conversations share wording; this says whether
    they are the same *kind* of scenario. A cohort of genuinely new attacks lands in
    clusters the earlier ones did not populate; a cohort that re-works old ground
    reproduces the earlier cohort's cluster histogram.

    The statistic is the Jensen-Shannon divergence between two cohorts' cluster
    histograms, reported against a null from reshuffling the cohort labels — with 60
    to 150 items over 15 clusters, sampling noise alone gives a non-zero JS.
    """
    from sklearn.cluster import KMeans

    all_rows = sorted({r for rows in rows_by_cohort.values() for r in rows})
    pos = {r: i for i, r in enumerate(all_rows)}
    X = M[all_rows]
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    lab = km.labels_
    names = None
    if vocab is not None:
        names = [
            ", ".join(vocab[i] for i in np.argsort(km.cluster_centers_[c])[::-1][:6])
            for c in range(k)
        ]

    hist = {}
    for c, rows in rows_by_cohort.items():
        h = np.bincount([lab[pos[r]] for r in rows], minlength=k).astype(float)
        hist[c] = h

    sizes = {c: int(h.sum()) for c, h in hist.items()}
    rng = np.random.default_rng(seed)
    out = {"k": k, "sizes": sizes, "cluster_terms": names,
           "histograms": {f"C{c}": h.astype(int).tolist() for c, h in hist.items()}}

    for a, b in ((1, 2), (2, 3), (1, 3)):
        if a not in hist or b not in hist:
            continue
        obs = js_divergence(hist[a] + 1e-12, hist[b] + 1e-12)
        pool = np.concatenate(
            [np.repeat(np.arange(k), hist[a].astype(int)),
             np.repeat(np.arange(k), hist[b].astype(int))]
        )
        null = []
        for _ in range(n_null):
            perm = rng.permutation(pool)
            ha = np.bincount(perm[: sizes[a]], minlength=k).astype(float)
            hb = np.bincount(perm[sizes[a] :], minlength=k).astype(float)
            null.append(js_divergence(ha + 1e-12, hb + 1e-12))
        null = np.array(null)
        out[f"C{a}_vs_C{b}"] = {
            "js": float(obs),
            "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "p_value": float((null >= obs).mean()),
        }

    # Clusters that a cohort populates and its predecessors do not, and vice versa.
    for c in (2, 3):
        if c not in hist:
            continue
        earlier = sum((hist[e] for e in hist if e < c), np.zeros(k))
        out[f"C{c}_new_clusters"] = int(((hist[c] > 0) & (earlier == 0)).sum())
        out[f"C{c}_share_in_clusters_unseen_before"] = float(
            hist[c][earlier == 0].sum() / max(hist[c].sum(), 1)
        )
        out[f"C{c}_abandoned_clusters"] = int(((hist[c] == 0) & (earlier > 0)).sum())
    return out


def top_terms(texts_a: list[str], texts_b: list[str], k: int = 12):
    """Log-odds-ratio with an informative Dirichlet prior (Monroe et al. 2008)."""
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(lowercase=True, min_df=3, ngram_range=(1, 1))
    X = vec.fit_transform(texts_a + texts_b)
    vocab = np.array(vec.get_feature_names_out())
    a = np.asarray(X[: len(texts_a)].sum(axis=0)).ravel().astype(float)
    b = np.asarray(X[len(texts_a) :].sum(axis=0)).ravel().astype(float)
    prior = (a + b) / (a + b).sum() * 500.0
    na, nb, n0 = a.sum(), b.sum(), prior.sum()
    la = np.log((a + prior) / (na + n0 - a - prior))
    lb = np.log((b + prior) / (nb + n0 - b - prior))
    delta = la - lb
    var = 1.0 / (a + prior) + 1.0 / (b + prior)
    z = delta / np.sqrt(var)
    order = np.argsort(z)
    return {
        "a": [(vocab[i], round(float(z[i]), 2)) for i in order[::-1][:k]],
        "b": [(vocab[i], round(float(z[i]), 2)) for i in order[:k]],
    }


# --- per-arm analysis ---------------------------------------------------------------


def analyse_arm(exp: "Experiment", arm: str, n_null: int) -> dict:
    print(f"\n=== {exp.name} :: {arm} ===", flush=True)
    universe, dump_sources = build_cohorts(exp, arm)
    raw = load_raw_successes(exp, arm)

    keys = [k for k, v in universe.items() if v["source"] is not None]
    cohort = {k: universe[k]["cohort"] for k in keys}
    by_cohort = {c: [k for k in keys if cohort[k] == c] for c in (1, 2, 3)}
    print("  pairs per cohort:", {c: len(v) for c, v in by_cohort.items()}, flush=True)
    n_no_source = sum(1 for v in universe.values() if v["source"] is None)
    n_no_gen = sum(1 for v in universe.values() if v["generated"] is None)

    src_text = {k: full_text(universe[k]["source"]) for k in keys}
    src_open = {k: first_user_text(universe[k]["source"])[:NEAR_DUP_PREFIX] for k in keys}
    # For an assistant-centric concept the label lives in the reply, and the submit-time
    # guard never looks there — so a clone of the assistant side would pass unseen.
    src_asst = {
        k: assistant_text(universe[k]["source"])[:NEAR_DUP_PREFIX] for k in keys
    }

    # --- 1. difflib near-duplication -------------------------------------------------
    print("  [1/4] difflib nearest neighbours ...", flush=True)
    dup = {}
    for c in (1, 2, 3):
        cur = by_cohort[c]
        earlier = [k for k in keys if cohort[k] < c]
        dup[f"C{c}_vs_earlier"] = describe(
            nn_difflib([src_open[k] for k in cur], [src_open[k] for k in earlier],
                       same_set=False)
        )
        dup[f"C{c}_within"] = describe(
            nn_difflib([src_open[k] for k in cur], [src_open[k] for k in cur],
                       same_set=True)
        )
    dup["all_within"] = describe(
        nn_difflib([src_open[k] for k in keys], [src_open[k] for k in keys],
                   same_set=True)
    )

    dup_asst = assistant_side_block(src_asst, keys, cohort, by_cohort)

    # --- 2. TF-IDF nearest neighbours ------------------------------------------------
    print("  [2/4] TF-IDF nearest neighbours ...", flush=True)
    idx = {k: i for i, k in enumerate(keys)}
    Mw, Mc, vocab_w = tfidf_matrices([src_text[k] for k in keys])
    cos = {}
    for name, M in (("word", Mw), ("char", Mc)):
        for c in (1, 2, 3):
            cur = [idx[k] for k in by_cohort[c]]
            earlier = [idx[k] for k in keys if cohort[k] < c]
            cos[f"{name}_C{c}_vs_earlier"] = describe(
                nn_cosine(M, cur, earlier, same_set=False)
            )
            cos[f"{name}_C{c}_within"] = describe(
                nn_cosine(M, cur, cur, same_set=True)
            )

    # --- 3. separability -------------------------------------------------------------
    print("  [3/4] separability ...", flush=True)
    sep = {}
    for a, b in ((1, 2), (2, 3), (1, 3)):
        ra = [idx[k] for k in by_cohort[a]]
        rb = [idx[k] for k in by_cohort[b]]
        sep[f"C{a}_vs_C{b}"] = separability(Mw, ra, rb, n_null=n_null)
    terms = {
        f"C{a}_vs_C{b}": top_terms(
            [src_text[k] for k in by_cohort[a]], [src_text[k] for k in by_cohort[b]]
        )
        for a, b in ((1, 3),)
    }

    # --- 4. provenance ---------------------------------------------------------------
    print("  [4/4] provenance ...", flush=True)
    prov = {}
    for c in (1, 2, 3):
        rows = [raw[k] for k in by_cohort[c] if k in raw]
        et = Counter(r["error_type"] for r in rows)
        prov[f"C{c}"] = {
            "n_pairs": len(by_cohort[c]),
            "n_joined_to_raw": len(rows),
            "error_type": dict(et),
            "frac_false_positive": (
                et.get("false_positive", 0) / len(rows) if rows else float("nan")
            ),
            "judge_label": dict(Counter(r["judge_label"] for r in rows)),
            "mean_probe_score": (
                float(np.mean([r["probe_score"] for r in rows])) if rows else float("nan")
            ),
            "mean_judge_confidence": (
                float(np.mean([r["judge_confidence"] for r in rows])) if rows else float("nan")
            ),
            "rounds": dict(sorted(Counter(r["round"] for r in rows).items())),
            "mean_words": float(
                np.mean([len(src_text[k].split()) for k in by_cohort[c]])
            ),
            "mean_messages": float(
                np.mean([len(universe[k]["source"]) for k in by_cohort[c]])
            ),
        }

    # --- 5. the error-type confound --------------------------------------------------
    # The cohorts do not hold the attack DIRECTION fixed: for deepseekv4pro C1 is 100%
    # false-positive hunting and C2 97% false-negative. Those are different tasks, so a
    # classifier separating C1 from C2 may only be re-detecting the direction. Repeating
    # the comparison inside one error type removes that explanation.
    print("  [5/5] separability within error type ...", flush=True)
    err_of = {k: raw[k]["error_type"] for k in keys if k in raw}
    sep_et: dict[str, dict] = {}
    nn_et: dict[str, dict] = {}
    for et in ("false_positive", "false_negative"):
        sub = {c: [k for k in by_cohort[c] if err_of.get(k) == et] for c in (1, 2, 3)}
        sep_et[f"{et}_sizes"] = {f"C{c}": len(v) for c, v in sub.items()}
        for a, b in ((1, 2), (2, 3), (1, 3)):
            if len(sub[a]) < 15 or len(sub[b]) < 15:
                continue
            sep_et[f"{et}_C{a}_vs_C{b}"] = separability(
                Mw, [idx[k] for k in sub[a]], [idx[k] for k in sub[b]], n_null=n_null
            )
        for c in (2, 3):
            earlier = [k for c2 in range(1, c) for k in sub[c2]]
            if not sub[c] or not earlier:
                continue
            nn_et[f"{et}_C{c}_vs_earlier"] = describe(
                nn_difflib([src_open[k] for k in sub[c]],
                           [src_open[k] for k in earlier], same_set=False)
            )
            nn_et[f"{et}_C{c}_within"] = describe(
                nn_difflib([src_open[k] for k in sub[c]],
                           [src_open[k] for k in sub[c]], same_set=True)
            )

    # --- 6. topic occupancy ----------------------------------------------------------
    print("  [6/6] topic clustering ...", flush=True)
    rows_by_cohort = {c: [idx[k] for k in by_cohort[c]] for c in (1, 2, 3)}
    Mt, vocab_t = topic_matrix([content_text(universe[k]["source"]) for k in keys])
    topics = topic_contingency(Mt, rows_by_cohort, vocab=vocab_t)
    # The same clustering on the unfiltered matrix, kept so the report can say whether
    # the rotation is a fact about subject matter or an artifact of function words.
    topics_all_tokens = topic_contingency(Mw, rows_by_cohort, vocab=vocab_w)

    # --- anchors ---------------------------------------------------------------------
    print("  [anchor] source vs its own counterpart ...", flush=True)
    self_pair = []
    for k in keys:
        gen = universe[k]["generated"]
        if gen is None:
            continue
        self_pair.append(difflib_ratio(src_open[k], first_user_text(gen)[:NEAR_DUP_PREFIX]))

    return {
        "arm": arm,
        "experiment": exp.name,
        "label": exp.labels[arm],
        "n_pairs": len(keys),
        "n_pairs_missing_source": n_no_source,
        "n_pairs_missing_generated": n_no_gen,
        "cohort_sizes": {f"C{c}": len(v) for c, v in by_cohort.items()},
        "difflib": dup,
        "difflib_assistant_side": dup_asst,
        "cosine": cos,
        "separability": sep,
        "separability_within_error_type": sep_et,
        "difflib_within_error_type": nn_et,
        "topics": topics,
        "topics_all_tokens": topics_all_tokens,
        "top_terms": terms,
        "provenance": prov,
        "guard": guard_stats(exp, arm),
        "opener_chars": {
            f"C{c}": float(np.mean([len(src_open[k]) for k in by_cohort[c]]))
            for c in (1, 2, 3)
        },
        "anchor_self_pair_difflib": describe(self_pair),
        "_keys": {f"C{c}": by_cohort[c] for c in (1, 2, 3)},
        "_src_open": src_open,
        "_src_text": src_text,
    }


def anchor_base_data(exp: "Experiment") -> dict | None:
    """Independently collected conversations of the same concept, against themselves."""
    from tuberlens.interfaces.dataset import LabelledDataset

    if not exp.base_training_data:
        return None
    path = exp.root / exp.base_training_data
    if not path.exists():
        return None
    ds = LabelledDataset.load_from(
        path,
        pos_class_label=exp.pos_label,
        neg_class_label=exp.neg_label,
        combine_consecutive_messages=exp.combine_consecutive_messages,
        convert_tool_to_assistant=exp.convert_tool_to_assistant,
    )
    opens = [first_user_text(m)[:NEAR_DUP_PREFIX] for m in ds.inputs]
    out = describe(nn_difflib(opens, opens, same_set=True))
    # difflib's ratio is length-sensitive, so the anchor is only readable next to the
    # opener length it was measured on.
    out["mean_opener_chars"] = float(np.mean([len(o) for o in opens]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="hs", choices=sorted(EXPERIMENTS))
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--null-draws", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--dups-only",
        action="store_true",
        help="recompute only the cross-cohort duplicate listing and patch it in",
    )
    ap.add_argument(
        "--assistant-only",
        action="store_true",
        help="recompute only the assistant-side duplication block and patch it in",
    )
    ap.add_argument(
        "--guard-only",
        action="store_true",
        help="recompute only the near-dup-guard stats and patch them into an existing "
             "JSON, without redoing the hours-long similarity passes",
    )
    args = ap.parse_args()

    exp = EXPERIMENTS[args.experiment]
    arms = args.arms or list(exp.arms)
    out = args.out or (
        REPO / f"results_cohort_novelty/{exp.name}_cohort_novelty.json"
    )

    if args.dups_only:
        payload = json.loads(out.read_text(encoding="utf-8"))
        for a in arms:
            universe, _ = build_cohorts(exp, a)
            raw = load_raw_successes(exp, a)
            keys = [k for k, v in universe.items() if v["source"] is not None]
            cohort = {k: universe[k]["cohort"] for k in keys}
            by_cohort = {c: [k for k in keys if cohort[k] == c] for c in (1, 2, 3)}
            blk = cross_cohort_duplicates(universe, keys, cohort, by_cohort, raw)
            payload["arms"][a]["cross_cohort_duplicates"] = blk
            print(f"  {a}: {blk['n_pairs']} pair(s) >= 0.8 "
                  f"({blk['n_cross_error_type']} cross-error-type, "
                  f"{blk['n_contradictory_twins']} contradictory twin(s))", flush=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"patched {out}")
        return

    if args.assistant_only:
        payload = json.loads(out.read_text(encoding="utf-8"))
        for a in arms:
            universe, _ = build_cohorts(exp, a)
            keys = [k for k, v in universe.items() if v["source"] is not None]
            cohort = {k: universe[k]["cohort"] for k in keys}
            by_cohort = {c: [k for k in keys if cohort[k] == c] for c in (1, 2, 3)}
            src_asst = {
                k: assistant_text(universe[k]["source"])[:NEAR_DUP_PREFIX] for k in keys
            }
            block = assistant_side_block(src_asst, keys, cohort, by_cohort)
            payload["arms"][a]["difflib_assistant_side"] = block
            print(f"  {a}: {block['n_without_assistant_turn']} of {len(keys)} had no "
                  f"assistant turn and were excluded", flush=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"patched {out}")
        return

    if args.guard_only:
        payload = json.loads(out.read_text(encoding="utf-8"))
        for a in arms:
            payload["arms"][a]["guard"] = guard_stats(exp, a)
            print(f"  {a}: {payload['arms'][a]['guard']['n_rejected']} rejection(s)")
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"patched {out}")
        return

    results = {a: analyse_arm(exp, a, args.null_draws) for a in arms}

    print("\n=== anchors ===", flush=True)
    anchors = {}
    base = anchor_base_data(exp)
    if base is not None:
        anchors["base_data_within"] = base
    if len(arms) == 2:
        a, b = arms
        ta = [results[a]["_src_open"][k] for k in results[a]["_src_open"]]
        tb = [results[b]["_src_open"][k] for k in results[b]["_src_open"]]
        anchors["cross_arm"] = describe(nn_difflib(ta, tb, same_set=False))

        # Cross-arm separability, on a shared vectoriser: the ceiling for "two sets of
        # successes that share nothing but the task".
        texts = [results[a]["_src_text"][k] for k in results[a]["_src_text"]] + [
            results[b]["_src_text"][k] for k in results[b]["_src_text"]
        ]
        Mw, _, _ = tfidf_matrices(texts)
        na = len(results[a]["_src_text"])
        anchors["cross_arm_separability"] = separability(
            Mw, list(range(na)), list(range(na, len(texts))), n_null=args.null_draws
        )

    payload = {
        "experiment": exp.name,
        "concept": exp.concept,
        "assistant_centric": exp.assistant_centric,
        "labels": {a: exp.labels[a] for a in arms},
        "arms": {
            a: {k: v for k, v in r.items() if not k.startswith("_")}
            for a, r in results.items()
        },
        "anchors": anchors,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
