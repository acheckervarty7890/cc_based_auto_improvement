"""Topic + length analysis of red-team successes, per iteration, per arm.

For each run (a results dir holding ``*_fp.jsonl`` / ``*_fn.jsonl`` plus the
matching probe dir holding ``contrastive_cache.jsonl``) this reports, per
iteration:

* how many successes were found, split by error type and combined;
* average conversation length -- separately for the **original** success
  conversations and for the **generated contrastive pairs** those successes
  produced -- broken down by user vs. assistant turn (words and characters);
* the most common topics, via a TF-IDF + KMeans topic model fit **once across
  every run** so the cluster ids mean the same thing in every table.

The original/contrastive pairing reuses ``preprocessing._cache_key`` so a
success is matched to the pair that was actually generated from it (keyed on
the source messages + the target label), rather than guessing from file order.
Successes dropped by ``filter_dataset`` before contrastive generation simply
have no pair; the coverage is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# --- pairing: mirrors agentic_redteam.preprocessing ------------------------


def cache_key(messages: Sequence[dict], target_label: str) -> str:
    payload = json.dumps({"messages": list(messages), "target": target_label}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def success_human_label(rec: dict, pos_label: str, neg_label: str) -> str:
    """The true-class label a success is trained with (mirrors retrain.py)."""
    if rec.get("judge_label") == pos_label:
        return pos_label
    if rec.get("judge_label") == neg_label:
        return neg_label
    return neg_label if rec.get("error_type") == "false_positive" else pos_label


# --- loading ---------------------------------------------------------------


@dataclass
class Success:
    arm: str
    iteration: int
    error_type: str
    messages: list[dict]
    label: str
    contrastive: list[dict] | None = None
    # Why the preprocessing LLM wrote the pair the way it did, and which model wrote it.
    contrastive_explanation: str = ""
    contrastive_model: str = ""
    probe_score: float = 0.0
    probe_predicts_positive: bool = False
    judge_label: str = ""
    judge_reason: str = ""
    pos_class_label: str = ""
    neg_class_label: str = ""
    # Whether this success is present in the *latest* redteam_postprocessed_iter*.jsonl,
    # i.e. whether it survived filter_dataset into the most recent retrain.
    in_final_training: bool = False
    # Whether it appears in any postprocessed snapshot (it trained some probe).
    in_any_training: bool = False


@dataclass
class Arm:
    name: str
    results_dir: Path
    probe_dir: Path
    successes: list[Success] = field(default_factory=list)
    n_attempts: Counter = field(default_factory=Counter)


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def canon(messages: Sequence[dict]) -> str:
    """Canonical text of a conversation, for set membership."""
    return json.dumps(
        [{"role": m.get("role", ""), "content": m.get("content", "")} for m in messages],
        sort_keys=True,
    )


def _training_sets(probe_dir: Path) -> tuple[set[str], set[str]]:
    """(present in the latest postprocessed snapshot, present in any snapshot).

    ``redteam_postprocessed_iter{N}.jsonl`` is exactly the red-team data that
    trained probe N, so membership answers "did this success reach training?"
    without re-deriving filter_dataset.
    """
    snapshots = sorted(
        probe_dir.glob("redteam_postprocessed_iter*.jsonl"),
        key=lambda p: int(re.search(r"iter(\d+)", p.name).group(1)),
    )
    any_set: set[str] = set()
    latest: set[str] = set()
    for i, path in enumerate(snapshots):
        rows = {canon(r["inputs"]) for r in load_jsonl(path) if isinstance(r.get("inputs"), list)}
        any_set |= rows
        if i == len(snapshots) - 1:
            latest = rows
    return latest, any_set


def load_arm(name: str, results_dir: Path, probe_dir: Path) -> Arm:
    arm = Arm(name=name, results_dir=results_dir, probe_dir=probe_dir)

    cache: dict[str, dict] = {}
    cache_path = probe_dir / "contrastive_cache.jsonl"
    if cache_path.exists():
        for row in load_jsonl(cache_path):
            if "key" in row and "record" in row:
                cache[row["key"]] = row["record"]
    latest_training, any_training = _training_sets(probe_dir)

    for path in sorted(results_dir.glob("*_f[pn].jsonl")):
        for rec in load_jsonl(path):
            arm.n_attempts[rec["iteration"]] += 1
            if not rec.get("success"):
                continue
            pos, neg = rec["pos_class_label"], rec["neg_class_label"]
            messages = [
                {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
                for m in rec["sample"]["messages"]
            ]
            label = success_human_label(rec, pos, neg)
            target = neg if label == pos else pos
            pair = cache.get(cache_key(messages, target))
            key = canon(messages)
            arm.successes.append(
                Success(
                    arm=name,
                    iteration=rec["iteration"],
                    error_type=rec["error_type"],
                    messages=messages,
                    label=label,
                    contrastive=pair["inputs"] if pair else None,
                    contrastive_explanation=str(pair.get("generation_explanation", "")) if pair else "",
                    contrastive_model=str(pair.get("generation_model", "")) if pair else "",
                    probe_score=float(rec.get("probe_score", 0.0)),
                    probe_predicts_positive=bool(rec.get("probe_predicts_positive", False)),
                    judge_label=str(rec.get("judge_label", "")),
                    judge_reason=str(rec.get("judge_reason", "")),
                    pos_class_label=pos,
                    neg_class_label=neg,
                    in_final_training=key in latest_training,
                    in_any_training=key in any_training,
                )
            )
    return arm


# --- length stats ----------------------------------------------------------

_WORD_RE = re.compile(r"\S+")


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def length_stats(conversations: list[list[dict]]) -> dict:
    """Mean words/chars per turn by role, plus per-conversation totals."""
    per_role_words: dict[str, list[int]] = defaultdict(list)
    per_role_chars: dict[str, list[int]] = defaultdict(list)
    per_role_turns: dict[str, list[int]] = defaultdict(list)
    conv_words, conv_chars, conv_turns = [], [], []

    for msgs in conversations:
        role_count: Counter = Counter()
        tw = tc = 0
        for m in msgs:
            role = m.get("role", "")
            content = m.get("content", "") or ""
            w, c = len(_WORD_RE.findall(content)), len(content)
            per_role_words[role].append(w)
            per_role_chars[role].append(c)
            role_count[role] += 1
            tw += w
            tc += c
        for role in ("user", "assistant", "system"):
            per_role_turns[role].append(role_count.get(role, 0))
        conv_words.append(tw)
        conv_chars.append(tc)
        conv_turns.append(len(msgs))

    out = {
        "n_conversations": len(conversations),
        "turns_per_conversation": _mean(conv_turns),
        "words_per_conversation": _mean(conv_words),
        "chars_per_conversation": _mean(conv_chars),
        "by_role": {},
    }
    for role in ("user", "assistant", "system"):
        if not per_role_words.get(role):
            continue
        out["by_role"][role] = {
            "n_turns": len(per_role_words[role]),
            "turns_per_conversation": _mean(per_role_turns[role]),
            "words_per_turn": _mean(per_role_words[role]),
            "chars_per_turn": _mean(per_role_chars[role]),
            "median_words_per_turn": statistics.median(per_role_words[role]),
        }
    return out


# --- topics ----------------------------------------------------------------

_STOP_EXTRA = [
    "assistant", "user", "just", "like", "know", "want", "need", "really",
    "think", "make", "sure", "thing", "things", "way", "ok", "okay", "hi",
    "hey", "hello", "thanks", "id", "ive", "im", "dont", "doesnt", "cant",
    "wont", "youre", "theyre", "isnt", "wasnt", "didnt", "got", "get",
    "tell", "help", "going", "let", "does", "did", "say", "said", "ask",
]


def topic_text(msgs: list[dict]) -> str:
    """Text a conversation's topic is read from: its user turns."""
    parts = [m["content"] for m in msgs if m.get("role") == "user"]
    if not parts:  # degenerate: fall back to everything
        parts = [m.get("content", "") for m in msgs]
    return "\n".join(parts)


def fit_topic_model(texts: list[str], k: int, seed: int):
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

    stop = list(ENGLISH_STOP_WORDS) + _STOP_EXTRA
    vec = TfidfVectorizer(
        stop_words=stop,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.4,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    terms = vec.get_feature_names_out()
    top_terms = {
        int(c): [terms[i] for i in km.cluster_centers_[c].argsort()[::-1][:12]]
        for c in range(k)
    }
    return labels, top_terms, X, vec


def top_keywords(texts: list[str], n: int = 10) -> list[str]:
    """Plain TF-IDF keyword pass (the no-clustering view)."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

    if len(texts) < 2:
        return []
    stop = list(ENGLISH_STOP_WORDS) + _STOP_EXTRA
    vec = TfidfVectorizer(
        stop_words=stop, ngram_range=(1, 2), min_df=2, max_df=0.5,
        sublinear_tf=True, strip_accents="unicode",
    )
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return []
    scores = X.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    return [terms[i] for i in scores.argsort()[::-1][:n]]


# --- reporting -------------------------------------------------------------


def group_key(s: Success, split: str) -> str:
    return "combined" if split == "combined" else s.error_type


def build_report(arms: list[Arm], topic_names: dict[int, str], k: int, seed: int, top_n: int) -> dict:
    all_success = [s for a in arms for s in a.successes]
    labels, top_terms, _, _ = fit_topic_model([topic_text(s.messages) for s in all_success], k, seed)
    for s, lab in zip(all_success, labels):
        s.topic = int(lab)  # type: ignore[attr-defined]

    report = {
        "topic_model": {
            "k": k,
            "seed": seed,
            "clusters": {
                str(c): {"name": topic_names.get(c, f"cluster {c}"), "top_terms": top_terms[c]}
                for c in range(k)
            },
        },
        "arms": {},
    }

    for arm in arms:
        # Union with n_attempts so an iteration that produced *no* successes
        # still shows up (that is a result, not an absence of data).
        iters = sorted({s.iteration for s in arm.successes} | set(arm.n_attempts))
        arm_out: dict = {
            "results_dir": str(arm.results_dir),
            "probe_dir": str(arm.probe_dir),
            "n_successes": len(arm.successes),
            "iterations": {},
        }
        for it in iters:
            it_rows = [s for s in arm.successes if s.iteration == it]
            per_split: dict = {}
            for split in ("false_positive", "false_negative", "combined"):
                rows = [s for s in it_rows if group_key(s, split) == split or split == "combined"]
                if split != "combined":
                    rows = [s for s in it_rows if s.error_type == split]
                if not rows:
                    continue
                originals = [s.messages for s in rows]
                pairs = [s.contrastive for s in rows if s.contrastive]
                counts = Counter(getattr(s, "topic") for s in rows)
                per_split[split] = {
                    "n_successes": len(rows),
                    "contrastive_coverage": len(pairs) / len(rows),
                    "original": length_stats(originals),
                    "contrastive": length_stats(pairs),
                    "topics": [
                        {
                            "topic": topic_names.get(c, f"cluster {c}"),
                            "cluster": c,
                            "count": n,
                            "pct": 100 * n / len(rows),
                        }
                        for c, n in counts.most_common(top_n)
                    ],
                    "keywords": top_keywords([topic_text(s.messages) for s in rows]),
                }
            arm_out["iterations"][str(it)] = {
                "n_attempts": arm.n_attempts.get(it, 0),
                "splits": per_split,
            }
        report["arms"][arm.name] = arm_out
    return report


DEFAULT_ARMS = [
    ("gpt51_memo", "results_hu_harm_llama70b50_gpt51_memo", "probes/llama70b50_gpt51_memo"),
    ("gpt51_nomemo", "results_hu_harm_llama70b50_gpt51_nomemo", "probes/llama70b50_gpt51_nomemo"),
    ("deepseekv4_memo", "results_hu_harm_llama70b50_deepseekv4_memo", "probes/llama70b50_deepseekv4_memo"),
    ("deepseekv4_nomemo", "results_hu_harm_llama70b50_deepseekv4_nomemo", "probes/llama70b50_deepseekv4_nomemo"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.cwd(),
                    help="base dir the default arm paths resolve against")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME:RESULTS_DIR:PROBE_DIR",
                    help="override/add an arm; repeatable")
    ap.add_argument("--clusters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--topic-names", type=Path, help="JSON {cluster_id: name} for readable topics")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    specs = [tuple(a.split(":", 2)) for a in args.arm] or DEFAULT_ARMS
    arms = []
    for name, res, probe in specs:
        res_p, probe_p = Path(res), Path(probe)
        if not res_p.is_absolute():
            res_p = args.root / res_p
        if not probe_p.is_absolute():
            probe_p = args.root / probe_p
        if not res_p.exists():
            print(f"skip {name}: {res_p} missing")
            continue
        arm = load_arm(name, res_p, probe_p)
        print(f"{name}: {len(arm.successes)} successes, "
              f"{sum(1 for s in arm.successes if s.contrastive)} with a contrastive pair")
        arms.append(arm)

    topic_names = {}
    if args.topic_names and args.topic_names.exists():
        topic_names = {int(k): v for k, v in json.loads(args.topic_names.read_text()).items()}

    report = build_report(arms, topic_names, args.clusters, args.seed, args.top_n)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
