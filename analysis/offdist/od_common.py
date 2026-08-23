"""Shared loading, pairing and behaviour detection for the off-distribution study.

The question this package answers: **which red-team rows are unlike the eval set, and does
removing them help?** Three ways of being unlike, all computed here or in
``characterize.py``:

* **surface** — lexical and structural. A discriminator trained to tell red-team text from
  eval text; a row's out-of-fold P(red-team) is its off-distribution score.
* **convention** — whether a pair's (assistant behaviour -> label) mapping runs the same
  way the eval splits' does, or the opposite way.
* **axis** — whether a contrastive pair contrasts on the ASSISTANT's turn, as the eval pairs
  do, or on the USER's turn.

Everything reads the same objects the ceiling analysis reads, so a removal condition here is
comparable to that study's N=0 point row for row.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "ceiling_analysis" / "scripts"))

RESULTS = Path(__file__).resolve().parent / "results"

POS = "harmful_to_human"
NEG = "not_harmful_to_human"
CANON = {"positive": POS, "negative": NEG}
OPPOSITE = {POS: NEG, NEG: POS}


@dataclass(frozen=True)
class Arm:
    """One experiment22 arm: its red-team set, its pair cache, its ceiling-study concept."""

    key: str
    attacker: str
    probe_dir: Path
    concept: str  # the ca_common CONCEPTS key, so fits reuse the ceiling harness

    @property
    def redteam_jsonl(self) -> Path:
        return self.probe_dir / "redteam_postprocessed_iter5.jsonl"

    @property
    def contrastive_cache(self) -> Path:
        return self.probe_dir / "contrastive_cache.jsonl"


ARMS = {
    "gptoss120b": Arm(
        "gptoss120b", "openai/gpt-oss-120b",
        REPO / "probes/hu_harm_gemma27b_gptoss120b_datadesc",
        "hu_ha_dd_gptoss120b",
    ),
    "deepseekv4pro": Arm(
        "deepseekv4pro", "deepseek/deepseek-v4-pro",
        REPO / "probes/hu_harm_gemma27b_deepseekv4pro_datadesc",
        "hu_ha_dd_deepseekv4pro",
    ),
}

EVAL_DIR = REPO / "eval_sets/hu_ha"
DEV_DIR = REPO / "dev_samples/hu_ha"


# ---------------------------------------------------------------------------- loading
def load_redteam(arm: Arm) -> list[dict]:
    """The postprocessed rows, in file order — which is the order `redteam_source` uses.

    `ca_common.redteam_source` concatenates the 50 base rows and then this file, unchanged,
    so red-team row *i* here is pool row *50 + i*. `pool_index` records that, and it is the
    only thing that lets a flag computed on text address a row of the activation blob.
    """
    rows = []
    for i, line in enumerate(arm.redteam_jsonl.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append({
            "i": i,
            "pool_index": 50 + i,      # 50 base rows come first in redteam_source
            "id": r.get("id", f"redteam-{i}"),
            "messages": r["inputs"],
            "label": CANON.get(r["label"], r["label"]),
        })
    return rows


def load_eval() -> list[dict]:
    out = []
    for path in sorted(EVAL_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out.append({"split": path.stem, "messages": json.loads(r["inputs"]),
                        "label": r["labels"]})
    return out


def load_pair_cache(arm: Arm) -> dict[str, dict]:
    cache = {}
    if not arm.contrastive_cache.exists():
        return cache
    for line in arm.contrastive_cache.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cache[r["key"]] = r["record"]
    return cache


def canonical(messages) -> str:
    return json.dumps(messages, sort_keys=True)


def recover_pairs(arm: Arm, rows: list[dict]) -> list[tuple[int, int]]:
    """`(source_i, generated_i)` for every contrastive pair present in the set.

    `generate_contrastive_dataset` returns originals **plus** their generated opposites, and
    caches each generation under `sha256(source messages + target label)`. So a row is a
    pair *source* exactly when hashing it against the opposite label hits the cache, and its
    partner is the row whose messages equal that cache record's. Both halves are in the
    postprocessed file, so the pairing is recovered without re-running anything.

    The guidance fingerprint is empty here: neither arm's config sets
    `preprocessing.concept_description` or `label_guidance` (deliberately — that would move
    a third thing at once), so `_cache_key` takes its two-field form.
    """
    from agentic_redteam.preprocessing import _cache_key

    cache = load_pair_cache(arm)
    by_text = {canonical(r["messages"]): r["i"] for r in rows}
    pairs = []
    for r in rows:
        key = _cache_key(r["messages"], OPPOSITE[r["label"]])
        rec = cache.get(key)
        if rec is None:
            continue
        j = by_text.get(canonical(rec["inputs"]))
        if j is not None and j != r["i"]:
            pairs.append((r["i"], j))
    return pairs


# ---------------------------------------------------------------------------- text views
def turns(messages, role: str) -> list[str]:
    return [str(m.get("content") or "") for m in messages if m.get("role") == role]


def joined(messages, role: str | None = None) -> str:
    if role is None:
        return "\n".join(str(m.get("content") or "") for m in messages)
    return "\n".join(turns(messages, role))


# ------------------------------------------------------------------- behaviour detection
# The core vocabulary is `llm_judge._REFUSAL_MARKERS` — the repo's own tested list, reused
# rather than re-derived. Two adjustments, both because that function answers a different
# question (did the JUDGE decline to answer?) than this one (did the ASSISTANT refuse the
# user?):
#
#   * it requires the marker within the first 60 chars, since a refusal *opens* a reply. An
#     assistant in these datasets often refuses after a paragraph of framing, so the scan
#     here covers the whole assistant turn.
#   * a few softer forms an assistant uses and a judge does not ("I'd rather not", "against
#     my guidelines") are added.
#
# What is kept verbatim is `_strip_quoted_spans`, and it is load-bearing for exactly the
# reason CLAUDE.md records: this concept's data is *about* refusal, so a conversation that
# quotes a refusal phrase is common, and a plain substring scan reads the citation as the
# utterance.
_EXTRA_REFUSAL_MARKERS = (
    "i'd rather not",
    "i would rather not",
    "i'm not comfortable",
    "i am not comfortable",
    "i don't feel comfortable",
    "against my guidelines",
    "against my programming",
    "i can't engage",
    "i cannot engage",
    "i won't be able to help",
    "i'm going to decline",
    "i won't do that",
    "i will not do that",
)

# "helped anyway" cues: an assistant that hedges but still delivers.
_HEDGE_THEN_HELP = re.compile(
    r"\b(however|that said|but here|even so|nonetheless|with that in mind)\b", re.I
)


def _markers() -> tuple[str, ...]:
    from agentic_redteam.llm_judge import _REFUSAL_MARKERS

    return tuple(_REFUSAL_MARKERS) + _EXTRA_REFUSAL_MARKERS


def assistant_refuses(messages) -> bool:
    """True when any assistant turn actually utters a refusal (not merely quotes one)."""
    from agentic_redteam.llm_judge import _strip_quoted_spans

    text = _strip_quoted_spans(joined(messages, "assistant")).lower()
    return any(m in text for m in _markers())


def structural_features(messages) -> dict:
    """Shape of a conversation, independent of what it is about."""
    roles = [m.get("role") for m in messages]
    u, a = turns(messages, "user"), turns(messages, "assistant")
    sys_turns = turns(messages, "system")
    all_text = joined(messages)
    return {
        "n_messages": len(messages),
        "n_user": len(u),
        "n_assistant": len(a),
        "has_system": int(bool(sys_turns)),
        "starts_with_user": int(bool(roles) and roles[0] == "user"),
        "chars_total": len(all_text),
        "chars_user": sum(len(t) for t in u),
        "chars_assistant": sum(len(t) for t in a),
        "assistant_share": (sum(len(t) for t in a) / max(len(all_text), 1)),
        "n_newlines": all_text.count("\n"),
        "has_bullets": int(bool(re.search(r"^\s*[-*•]\s", all_text, re.M))),
        "has_numbered": int(bool(re.search(r"^\s*\d+[.)]\s", all_text, re.M))),
        "has_code_fence": int("```" in all_text),
        "digit_share": sum(c.isdigit() for c in all_text) / max(len(all_text), 1),
        "n_questions": all_text.count("?"),
        "refuses": int(assistant_refuses(messages)),
    }


# ---------------------------------------------------------------------------- io
def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def text_hash(messages) -> str:
    return hashlib.sha256(canonical(messages).encode("utf-8")).hexdigest()[:16]
