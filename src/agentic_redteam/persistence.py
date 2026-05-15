"""Conversation/Message types and JSONL persistence for red-team attempts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


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


@dataclass
class AttemptRecord:
    """One row in the JSONL log — one candidate evaluated by probe + judge.

    The judge always runs and returns a class label (one of pos_class_label /
    neg_class_label, or "" if the judge response was unparseable). `success`
    is True only if the probe and judge disagree in the configured
    error_type direction.
    """

    sample: Conversation
    probe_score: float
    probe_predicts_positive: bool
    judge_label: str  # one of pos_class_label / neg_class_label / "" (unparseable)
    judge_reason: str
    judge_confidence: int  # 1–10; 0 if missing/unparseable
    success: bool
    attacker_model: str
    run_id: str
    round: int
    error_type: str
    pos_class_label: str
    neg_class_label: str

    @property
    def probe_label(self) -> str:
        return self.pos_class_label if self.probe_predicts_positive else self.neg_class_label

    @property
    def probe_disagrees_with_judge(self) -> bool:
        return bool(self.judge_label) and self.probe_label != self.judge_label

    def to_jsonl_row(self) -> str:
        d = {
            "sample": self.sample.to_dict(),
            "probe_score": float(self.probe_score),
            "probe_predicts_positive": bool(self.probe_predicts_positive),
            "judge_label": self.judge_label,
            "judge_reason": self.judge_reason,
            "judge_confidence": int(self.judge_confidence),
            "success": bool(self.success),
            "attacker_model": self.attacker_model,
            "run_id": self.run_id,
            "round": int(self.round),
            "error_type": self.error_type,
            "pos_class_label": self.pos_class_label,
            "neg_class_label": self.neg_class_label,
        }
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_jsonl_row(cls, row: str) -> "AttemptRecord":
        d = json.loads(row)
        return cls(
            sample=Conversation.from_dict(d["sample"]),
            probe_score=float(d["probe_score"]),
            probe_predicts_positive=bool(d["probe_predicts_positive"]),
            judge_label=str(d.get("judge_label", "")),
            judge_reason=str(d.get("judge_reason", "")),
            judge_confidence=int(d.get("judge_confidence", 0)),
            success=bool(d["success"]),
            attacker_model=str(d["attacker_model"]),
            run_id=str(d["run_id"]),
            round=int(d["round"]),
            error_type=str(d["error_type"]),
            pos_class_label=str(d["pos_class_label"]),
            neg_class_label=str(d["neg_class_label"]),
        )


@dataclass
class JsonlStore:
    path: Path
    _seen: set[str] = field(default_factory=set, init=False)
    _success_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            for rec in self.iter_all():
                self._seen.add(rec.sample.to_canonical_text())
                if rec.success:
                    self._success_count += 1

    def append(self, record: AttemptRecord) -> bool:
        """Append a record. Returns True if newly persisted, False if duplicate."""
        key = record.sample.to_canonical_text()
        if key in self._seen:
            return False
        self._seen.add(key)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(record.to_jsonl_row() + "\n")
        if record.success:
            self._success_count += 1
        return True

    def is_duplicate(self, conversation: Conversation) -> bool:
        return conversation.to_canonical_text() in self._seen

    @property
    def success_count(self) -> int:
        return self._success_count

    def iter_all(self) -> Iterator[AttemptRecord]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield AttemptRecord.from_jsonl_row(line)

    def iter_successes(self) -> Iterator[AttemptRecord]:
        for rec in self.iter_all():
            if rec.success:
                yield rec

    def recent_attempts(
        self, limit: int = 10, only_successful: bool = False
    ) -> list[AttemptRecord]:
        records = list(self.iter_successes() if only_successful else self.iter_all())
        return records[-limit:] if limit > 0 else records
