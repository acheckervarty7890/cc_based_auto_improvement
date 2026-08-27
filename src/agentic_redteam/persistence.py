"""Data model and JSONL sidecars for the generate → score → guide loop.

Everything a run produces that is not a probe pickle lives in the run's ``run_dir``
as append-only JSONL:

- ``batches.jsonl`` (:class:`BatchStore`) — one :class:`BatchRecord` per generated
  batch: the direction it was written under, its samples, the dev AUROC before and
  after training on it, and whether it was accepted into the training set.
- ``guidance.jsonl`` (:class:`GuidanceStore`) — one :class:`GuidanceRecord` per
  iteration: the judge's rolling memo and the ``n`` directions the next iteration's
  batches are generated under.
- ``runlog.jsonl`` (:class:`RunLogger`) — lifecycle and error events.

Both stores reload their sidecar on init, which is what makes a run resumable at
batch granularity: a batch already scored is not regenerated, and guidance already
written is not re-asked (see ``cli.iterative_generate_main``).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


def _iter_jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Yield the decodable JSON objects in a sidecar, skipping junk lines.

    Sidecars are append-only and a run can be killed mid-write, so the last line
    may be a torn fragment — skip it rather than failing the whole reload.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        return cls(role=str(d["role"]), content=str(d["content"]))


@dataclass(frozen=True)
class Conversation:
    messages: tuple[Message, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"messages": [m.to_dict() for m in self.messages]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Conversation":
        return cls(messages=tuple(Message.from_dict(m) for m in d["messages"]))

    @classmethod
    def from_messages(cls, messages: list[dict[str, Any]] | list[Message]) -> "Conversation":
        msgs: list[Message] = []
        for m in messages:
            if isinstance(m, Message):
                msgs.append(m)
            else:
                msgs.append(Message.from_dict(m))
        return cls(messages=tuple(msgs))

    def to_canonical_text(self) -> str:
        """Stable string form for dedup/hashing."""
        return "\n".join(f"{m.role}: {m.content}" for m in self.messages)


@dataclass(frozen=True)
class GeneratedSample:
    """One generator-written training example: a conversation and its self-assigned label.

    ``label`` is the human-readable class label (one of the probe's
    ``pos_class_label`` / ``neg_class_label``), exactly as the training JSONLs carry it.
    """

    conversation: Conversation
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {"messages": [m.to_dict() for m in self.conversation.messages], "label": self.label}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GeneratedSample":
        return cls(
            conversation=Conversation.from_messages(d["messages"]),
            label=str(d["label"]),
        )

    def to_training_row(self) -> dict[str, Any]:
        """The ``{inputs, labels}`` shape ``retrain._dicts_to_labelled_dataset`` reads."""
        return {
            "inputs": [m.to_dict() for m in self.conversation.messages],
            "labels": self.label,
        }

    @property
    def key(self) -> str:
        return self.conversation.to_canonical_text()


@dataclass
class BatchRecord:
    """One generated batch and what training on it did to the dev AUROC.

    ``auroc_before`` / ``auroc_after`` map each dev split name to its AUROC, plus a
    ``"mean"`` entry (the unweighted mean over splits, which is what ``delta`` is
    computed on and what the acceptance rule reads). ``status`` is ``"scored"`` for a
    batch that went through the fit, ``"generation_failed"`` when the generator
    produced nothing usable, and ``"empty"`` when every sample was dropped (too long,
    duplicate, bad label) before the fit.
    """

    run_id: str
    iteration: int
    batch_index: int
    direction: str
    generator_model: str
    provider: str
    samples: list[GeneratedSample]
    n_requested: int
    status: str = "scored"
    auroc_before: dict[str, float] = field(default_factory=dict)
    auroc_after: dict[str, float] = field(default_factory=dict)
    delta: float = 0.0
    accepted: bool = False
    exhausted: bool = False
    # Generation bookkeeping (why the batch is smaller than n_requested).
    n_dropped_too_long: int = 0
    n_dropped_duplicate: int = 0
    n_dropped_bad_label: int = 0
    n_generation_calls: int = 0
    error: str = ""
    candidate_probe_path: str = ""
    created_at: float = field(default_factory=time.time)

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_per_label(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.samples:
            counts[s.label] = counts.get(s.label, 0) + 1
        return counts

    @property
    def per_split_delta(self) -> dict[str, float]:
        return {
            k: self.auroc_after[k] - self.auroc_before[k]
            for k in self.auroc_after
            if k in self.auroc_before
        }

    def to_jsonl_row(self) -> str:
        d = asdict(self)
        d["samples"] = [s.to_dict() for s in self.samples]
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BatchRecord":
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in known and k != "samples"}
        kwargs["samples"] = [GeneratedSample.from_dict(s) for s in d.get("samples", [])]
        return cls(**kwargs)


@dataclass
class BatchStore:
    """Append-only ``batches.jsonl`` with reload-on-init and cross-batch sample dedup.

    ``seen_keys`` holds the canonical text of every sample ever generated in this run
    — accepted or not — so the generator is steered toward novelty: a conversation
    that already sat in a rejected batch tells us nothing new in another one.
    """

    path: Path
    records: list[BatchRecord] = field(default_factory=list)
    seen_keys: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            for row in _iter_jsonl_rows(self.path):
                try:
                    rec = BatchRecord.from_dict(row)
                except (KeyError, TypeError, ValueError):
                    continue
                self.records.append(rec)
                self.seen_keys.update(s.key for s in rec.samples)

    def forget_loaded(self) -> None:
        """Ignore the rows read from disk (``--no-resume``): the file is still appended
        to, but lookups, acceptance and the novelty guard see only this run's rows."""
        self.records.clear()
        self.seen_keys.clear()

    def append(self, record: BatchRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(record.to_jsonl_row() + "\n")
        self.records.append(record)
        self.seen_keys.update(s.key for s in record.samples)

    def is_seen(self, sample: GeneratedSample) -> bool:
        return sample.key in self.seen_keys

    def reserve(self, sample: GeneratedSample) -> bool:
        """Claim a sample's key for an in-flight batch; False if already taken.

        Concurrent batches within an iteration dedup against each other through this,
        before any of them is appended. Synchronous, so asyncio can't interleave two
        batches between the check and the claim.
        """
        if sample.key in self.seen_keys:
            return False
        self.seen_keys.add(sample.key)
        return True

    def for_iteration(self, iteration: int) -> list[BatchRecord]:
        return sorted(
            (r for r in self.records if r.iteration == iteration),
            key=lambda r: r.batch_index,
        )

    def get(self, iteration: int, batch_index: int) -> BatchRecord | None:
        """The newest record for ``(iteration, batch_index)`` — later rows supersede."""
        found = None
        for r in self.records:
            if r.iteration == iteration and r.batch_index == batch_index:
                found = r
        return found

    def accepted_samples(self, before_iteration: int | None = None) -> list[GeneratedSample]:
        """Samples of every accepted batch, in iteration/batch order.

        ``before_iteration`` limits it to iterations strictly earlier than that one —
        the training-set additions a given iteration starts from.
        """
        out: list[GeneratedSample] = []
        # A re-scored batch appends a second row; only the newest per key counts.
        newest: dict[tuple[int, int], BatchRecord] = {}
        for r in self.records:
            newest[(r.iteration, r.batch_index)] = r
        for key in sorted(newest):
            r = newest[key]
            if not r.accepted:
                continue
            if before_iteration is not None and r.iteration >= before_iteration:
                continue
            out.extend(r.samples)
        return out


@dataclass
class GuidanceRecord:
    """The judge's hand-off to one iteration's generator.

    ``iteration`` is the iteration these directions are FOR (the one whose batches are
    generated under them), so iteration 0's record — with no judged batches behind it
    — is written by the generator's own proposal step, not the judge.
    """

    run_id: str
    iteration: int
    memo: str
    directions: list[str]
    source: str = "judge"  # judge | generator_proposal
    baseline_auroc: dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_jsonl_row(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GuidanceRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class GuidanceStore:
    """Append-only ``guidance.jsonl``; newest record per iteration wins."""

    path: Path
    records: list[GuidanceRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            for row in _iter_jsonl_rows(self.path):
                try:
                    self.records.append(GuidanceRecord.from_dict(row))
                except (KeyError, TypeError, ValueError):
                    continue

    def forget_loaded(self) -> None:
        """Ignore the rows read from disk (``--no-resume``); see BatchStore.forget_loaded."""
        self.records.clear()

    def append(self, record: GuidanceRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(record.to_jsonl_row() + "\n")
        self.records.append(record)

    def for_iteration(self, iteration: int) -> GuidanceRecord | None:
        found = None
        for r in self.records:
            if r.iteration == iteration:
                found = r
        return found

    def latest_memo_before(self, iteration: int) -> str:
        """The newest memo written for an iteration <= ``iteration`` (the prior memo)."""
        best: GuidanceRecord | None = None
        for r in self.records:
            if r.iteration <= iteration and r.memo and (best is None or r.iteration >= best.iteration):
                best = r
        return best.memo if best else ""


@dataclass
class RunLogger:
    """Append-only sidecar log for lifecycle and error events.

    Each ``log`` call opens the file, appends one JSON line, and closes it so
    every event is durable even if the process is killed between calls.
    """

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **kwargs: Any) -> None:
        row = {"event": event, "ts": time.time(), **kwargs}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
