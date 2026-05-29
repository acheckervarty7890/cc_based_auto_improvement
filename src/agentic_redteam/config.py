"""Parse the markdown config file with YAML frontmatter + Attacker/Judge sections.

Config file shape:

    ---
    attacker:
      provider: openrouter            # claude_sdk | openrouter
      models:
        - anthropic/claude-sonnet-4.5
        - google/gemini-2.5-pro
        # mix providers per-model if you like:
        - name: claude-sonnet-4-6
          provider: claude_sdk
      max_turns: 30
      batch_target: 10
    judge:
      provider: claude_sdk            # claude_sdk | openrouter
      model: claude-sonnet-4-6
      max_tokens: 1024
    probe:
      path: data/probe.pkl            # optional: warm-start from an existing probe.
                                      # If omitted/missing, the first probe is trained
                                      # from base_training_data using the fields below.
      threshold: 0.5
      error_type: false_positive      # or false_negative, or [false_positive, false_negative]
      model: llama-1b                 # LLM to probe (used only when training from scratch)
      layer: 8                        # layer to probe (from scratch)
      pos_class_label: high-stakes    # (from scratch) also used to load base_training_data
      neg_class_label: low-stakes     # (from scratch)
      description: ...                # optional probe description (from scratch)
      architecture: linear_then_softmax  # optional ProbeType name (from scratch)
    preprocessing:                    # optional: collation-style preprocessing of red-team
      provider: openrouter            # successes before each retrain (filter + contrastive)
      model: anthropic/claude-sonnet-4.5
      max_concurrent: 50
      max_tokens: 2048
      filter_percentile: 0.8
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
Provider = Literal["claude_sdk", "openrouter"]

_VALID_PROVIDERS = ("claude_sdk", "openrouter")


@dataclass(frozen=True)
class AttackerModel:
    """One attacker model entry: the model name + which provider serves it."""

    name: str
    provider: Provider


@dataclass
class AttackerConfig:
    models: list[AttackerModel]
    max_turns: int = 30
    batch_target: int = 10
    rounds: int = 1
    concurrency: int = 1  # max parallel attacker sessions
    persistence_from_last_rounds: int | None = None  # None = show all rounds
    system_prompt: str = ""
    # The default provider applied to model entries given as a bare string.
    default_provider: Provider = "claude_sdk"

    @property
    def model_names(self) -> list[str]:
        return [m.name for m in self.models]


@dataclass
class JudgeConfig:
    model: str
    provider: Provider = "claude_sdk"
    max_tokens: int = 1024
    confidence_threshold: int = 7
    system_prompt: str = ""


@dataclass
class PreprocessingConfig:
    """LLM + knobs for the collation-style preprocessing applied to red-team successes.

    Mirrors the collation step of tuberlens' collate_train_evaluate.py: drop
    confounders (``filter_dataset``) then generate contrastive pairs
    (``generate_contrastive_dataset``). The contrastive generator needs an LLM,
    configured here independently of the attacker/judge.
    """

    model: str
    provider: Provider = "claude_sdk"
    max_concurrent: int = 50
    max_tokens: int = 2048
    filter_percentile: float = 0.8


@dataclass
class ProbeConfig:
    # An existing pickled probe to warm-start from. Optional: when absent (or the file
    # doesn't exist) the iterative loop trains the first probe from base_training_data
    # using the fields below.
    path: Path | None = None
    threshold: float = 0.5
    error_types: list[ErrorType] = field(default_factory=lambda: ["false_positive"])
    # Definition of the probe to train from scratch (ignored when warm-starting, since
    # those values are then read off the loaded probe).
    model: str | None = None
    layer: int | None = None
    pos_class_label: str | None = None
    neg_class_label: str | None = None
    description: str | None = None
    architecture: str | None = None

    @property
    def error_type(self) -> ErrorType:
        """Convenience accessor when there is exactly one error type."""
        return self.error_types[0]


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
    # Optional: when present, filter_dataset + generate_contrastive_dataset are applied to
    # red-team successes on every retrain. When None, retraining skips preprocessing.
    preprocessing: PreprocessingConfig | None = None

    @property
    def true_class_label_for_success(self) -> str:
        """When error_type=false_positive, success means probe-says-pos but truth is neg."""
        return "negative" if self.probe.error_type == "false_positive" else "positive"

    def jsonl_path_for(self, error_type: ErrorType) -> Path:
        """Return the JSONL path for a specific error type.

        When the config has a single error type, returns the original path.
        When multiple, suffixes with ``_fp`` or ``_fn``.
        """
        if len(self.probe.error_types) <= 1:
            return self.output.jsonl_path
        stem = self.output.jsonl_path.stem
        suffix = self.output.jsonl_path.suffix
        tag = "fp" if error_type == "false_positive" else "fn"
        return self.output.jsonl_path.with_name(f"{stem}_{tag}{suffix}")


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


def _validate_provider(value: str, where: str) -> Provider:
    if value not in _VALID_PROVIDERS:
        raise ValueError(
            f"{where}: provider must be one of {_VALID_PROVIDERS!r}, got {value!r}"
        )
    return value  # type: ignore[return-value]


def _parse_attacker_models(
    raw_models: list, default_provider: Provider
) -> list[AttackerModel]:
    """Each entry is either a bare string (uses default_provider) or {name, provider}."""
    out: list[AttackerModel] = []
    for i, entry in enumerate(raw_models):
        if isinstance(entry, str):
            out.append(AttackerModel(name=entry, provider=default_provider))
        elif isinstance(entry, dict):
            if "name" not in entry:
                raise ValueError(
                    f"attacker.models[{i}]: dict entries require a 'name' field"
                )
            provider = entry.get("provider", default_provider)
            out.append(
                AttackerModel(
                    name=str(entry["name"]),
                    provider=_validate_provider(
                        str(provider), f"attacker.models[{i}].provider"
                    ),
                )
            )
        else:
            raise ValueError(
                f"attacker.models[{i}]: entries must be strings or dicts, got {type(entry).__name__}"
            )
    return out


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
    pp = frontmatter.get("preprocessing") or {}

    if "models" not in a or not a["models"]:
        raise ValueError("attacker.models must be a non-empty list")
    if "model" not in j:
        raise ValueError("judge.model is required")
    if "jsonl_path" not in o:
        raise ValueError("output.jsonl_path is required")

    attacker_default_provider = _validate_provider(
        str(a.get("provider", "claude_sdk")), "attacker.provider"
    )
    judge_provider = _validate_provider(
        str(j.get("provider", "claude_sdk")), "judge.provider"
    )

    raw_error_type = pr.get("error_type", "false_positive")
    if isinstance(raw_error_type, str):
        error_types: list[ErrorType] = [raw_error_type]  # type: ignore[list-item]
    elif isinstance(raw_error_type, list):
        error_types = list(raw_error_type)
    else:
        raise ValueError(f"probe.error_type must be a string or list, got {type(raw_error_type).__name__}")
    for et in error_types:
        if et not in ("false_positive", "false_negative"):
            raise ValueError(f"probe.error_type entries must be 'false_positive' or 'false_negative', got {et!r}")

    persistence_raw = a.get("persistence_from_last_rounds")
    persistence_from_last_rounds = int(persistence_raw) if persistence_raw is not None else None

    preprocessing: PreprocessingConfig | None = None
    if pp:
        if "model" not in pp:
            raise ValueError("preprocessing.model is required when a preprocessing section is present")
        preprocessing = PreprocessingConfig(
            model=str(pp["model"]),
            provider=_validate_provider(
                str(pp.get("provider", "claude_sdk")), "preprocessing.provider"
            ),
            max_concurrent=int(pp.get("max_concurrent", 50)),
            max_tokens=int(pp.get("max_tokens", 2048)),
            filter_percentile=float(pp.get("filter_percentile", 0.8)),
        )

    return RedteamConfig(
        attacker=AttackerConfig(
            models=_parse_attacker_models(list(a["models"]), attacker_default_provider),
            max_turns=int(a.get("max_turns", 30)),
            batch_target=int(a.get("batch_target", 10)),
            rounds=int(a.get("rounds", 1)),
            concurrency=int(a.get("concurrency", 1)),
            persistence_from_last_rounds=persistence_from_last_rounds,
            system_prompt=attacker_prompt,
            default_provider=attacker_default_provider,
        ),
        judge=JudgeConfig(
            model=j["model"],
            provider=judge_provider,
            max_tokens=int(j.get("max_tokens", 1024)),
            confidence_threshold=int(j.get("confidence_threshold", 7)),
            system_prompt=judge_prompt,
        ),
        probe=ProbeConfig(
            path=_resolve(pr["path"]) if pr.get("path") else None,
            threshold=float(pr.get("threshold", 0.5)),
            error_types=error_types,
            model=pr.get("model"),
            layer=int(pr["layer"]) if pr.get("layer") is not None else None,
            pos_class_label=pr.get("pos_class_label"),
            neg_class_label=pr.get("neg_class_label"),
            description=pr.get("description"),
            architecture=pr.get("architecture"),
        ),
        output=OutputConfig(
            jsonl_path=_resolve(o["jsonl_path"]),
            run_id=o.get("run_id") or uuid.uuid4().hex[:8],
        ),
        source_path=path,
        preprocessing=preprocessing,
    )
