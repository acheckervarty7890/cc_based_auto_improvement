"""Parse the markdown config file with YAML frontmatter + Attacker/Judge sections.

Config file shape:

    ---
    attacker:
      models: [claude-sonnet-4-6, claude-opus-4-7]
      max_turns: 30
      batch_target: 10
    judge:
      model: claude-sonnet-4-6
      max_tokens: 1024
    probe:
      path: data/probe.pkl
      threshold: 0.5
      error_type: false_positive   # or false_negative
    output:
      jsonl_path: results/redteam.jsonl
      run_id: null
    ---

    # Attacker
    <attacker system prompt>

    # Judge
    <judge system prompt>
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

ErrorType = Literal["false_positive", "false_negative"]


@dataclass
class AttackerConfig:
    models: list[str]
    max_turns: int = 30
    batch_target: int = 10
    system_prompt: str = ""


@dataclass
class JudgeConfig:
    model: str
    max_tokens: int = 1024
    confidence_threshold: int = 7
    system_prompt: str = ""


@dataclass
class ProbeConfig:
    path: Path
    threshold: float = 0.5
    error_type: ErrorType = "false_positive"


@dataclass
class OutputConfig:
    jsonl_path: Path
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class RedteamConfig:
    attacker: AttackerConfig
    judge: JudgeConfig
    probe: ProbeConfig
    output: OutputConfig
    source_path: Path

    @property
    def true_class_label_for_success(self) -> str:
        """When error_type=false_positive, success means probe-says-pos but truth is neg."""
        return "negative" if self.probe.error_type == "false_positive" else "positive"


_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z",
    re.DOTALL,
)


def _split_frontmatter(text: str) -> tuple[dict, str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError(
            "Config must start with YAML frontmatter delimited by '---' lines"
        )
    frontmatter = yaml.safe_load(m.group(1)) or {}
    body = m.group(2)
    return frontmatter, body


_HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def _split_sections(body: str) -> dict[str, str]:
    """Split markdown body into {heading_lower: content} on top-level '#' headings."""
    headings = list(_HEADING_RE.finditer(body))
    if not headings:
        return {}
    sections: dict[str, str] = {}
    for i, m in enumerate(headings):
        name = m.group(1).strip().lower()
        start = m.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        sections[name] = body[start:end].strip()
    return sections


def load_config(path: str | Path) -> RedteamConfig:
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text)
    sections = _split_sections(body)

    attacker_prompt = sections.get("attacker")
    judge_prompt = sections.get("judge")
    if attacker_prompt is None or judge_prompt is None:
        raise ValueError(
            "Config body must contain '# Attacker' and '# Judge' top-level sections"
        )

    config_dir = path.parent

    def _resolve(p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (config_dir / p).resolve()

    a = frontmatter.get("attacker") or {}
    j = frontmatter.get("judge") or {}
    pr = frontmatter.get("probe") or {}
    o = frontmatter.get("output") or {}

    if "models" not in a or not a["models"]:
        raise ValueError("attacker.models must be a non-empty list")
    if "model" not in j:
        raise ValueError("judge.model is required")
    if "path" not in pr:
        raise ValueError("probe.path is required")
    if "jsonl_path" not in o:
        raise ValueError("output.jsonl_path is required")

    return RedteamConfig(
        attacker=AttackerConfig(
            models=list(a["models"]),
            max_turns=int(a.get("max_turns", 30)),
            batch_target=int(a.get("batch_target", 10)),
            system_prompt=attacker_prompt,
        ),
        judge=JudgeConfig(
            model=j["model"],
            max_tokens=int(j.get("max_tokens", 1024)),
            confidence_threshold=int(j.get("confidence_threshold", 7)),
            system_prompt=judge_prompt,
        ),
        probe=ProbeConfig(
            path=_resolve(pr["path"]),
            threshold=float(pr.get("threshold", 0.5)),
            error_type=pr.get("error_type", "false_positive"),
        ),
        output=OutputConfig(
            jsonl_path=_resolve(o["jsonl_path"]),
            run_id=o.get("run_id") or uuid.uuid4().hex[:8],
        ),
        source_path=path,
    )
