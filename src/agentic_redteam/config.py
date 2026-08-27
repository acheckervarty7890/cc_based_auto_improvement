"""Parse the markdown config file with YAML frontmatter + Generator/Judge sections.

Config file shape:

    ---
    generator:
      provider: openrouter            # claude_sdk | openrouter — default for bare-string models
      models:
        - meta-llama/llama-3.3-70b-instruct
        - name: claude-sonnet-4-6     # per-model provider override
          provider: claude_sdk
      n_batches: 5                    # batches generated per iteration (n)
      batch_size: 20                  # samples per batch (m); even, half per class
      concurrency: 5                  # parallel generator calls
      max_tokens: 8192                # response cap per generator call
      max_sample_tokens: 1024         # drop generated samples longer than this (0 = no cap)
      max_retries: 2                  # top-up calls when a batch comes back short
    judge:
      provider: openrouter
      model: openai/gpt-5.1-chat
      max_tokens: 2048
      memo_word_budget: 400           # rolling memo length the judge is asked for
      max_samples_per_batch: 6        # samples of each batch shown to the judge
    probe:
      path: data/probe.pkl            # optional: warm-start from an existing probe.
      model: llama-1b                 # from-scratch fields (ignored when warm-starting)
      layer: 8
      pos_class_label: high-stakes
      neg_class_label: low-stakes
      description: ...
      architecture: linear_then_softmax
      ensemble_size: 5                # optional (1..10)
    loop:
      iterations: 3                   # generate → score → retrain → guide cycles
      min_auroc_gain: 0.0             # batch accepted if mean dev ΔAUROC > this
      exhausted_gain: 0.002           # |Δ| <= this is reported to the judge as exhausted
    validation:
      dev_data: ../dev_samples/highstakes   # REQUIRED: dev set for early stopping AND ΔAUROC
    eval:
      combine_consecutive_messages: false
      convert_tool_to_assistant: false
      eval_max_samples: 0
      data_description: ...           # optional, shown to the judge
    output:
      run_dir: ../results/my_run      # batches.jsonl, guidance.jsonl, runlog.jsonl, snapshots
      run_id: null
      comparison_csv: ...             # optional (--eval)
      activations_cache_dir: ...      # optional (--eval)
      base_activation_cache_dir: ...  # optional (training activations)
    ---

    # Generator
    <generator system prompt>

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

from agentic_redteam.ensemble import MAX_ENSEMBLE_SIZE
from agentic_redteam.token_budget import MAX_ACTIVATION_TOKENS

Provider = Literal["claude_sdk", "openrouter"]

_VALID_PROVIDERS = ("claude_sdk", "openrouter")

DEFAULT_MEMO_WORD_BUDGET = 400
DEFAULT_MIN_AUROC_GAIN = 0.0
DEFAULT_EXHAUSTED_GAIN = 0.002


@dataclass(frozen=True)
class GeneratorModel:
    """One entry of ``generator.models``: a model name and the provider that serves it."""

    name: str
    provider: Provider


@dataclass
class GeneratorConfig:
    models: list[GeneratorModel]
    # n: batches generated per iteration. Batch k is written by models[k % len(models)]
    # under direction k of the iteration's guidance.
    n_batches: int = 5
    # m: samples per batch. Must be even — the generator is asked for m/2 of each class.
    batch_size: int = 20
    # Max parallel generator calls within an iteration.
    concurrency: int = 5
    # Response token cap per generator call. m conversations of a few hundred tokens
    # each add up; a cap that guillotines the reply costs the tail of the batch.
    max_tokens: int = 8192
    # max_sample_tokens: length cap per generated sample, in tokens of the PROBE's
    #   tokenizer (chat template applied, <bos> included). A longer sample is dropped:
    #   tuberlens' get_activations truncates at 1024, so it would be trained on without
    #   its tail. 0 (or less) disables the check.
    max_sample_tokens: int = MAX_ACTIVATION_TOKENS
    # Top-up calls per batch when the reply is short of m usable samples (parse
    # failures, drops, class imbalance). Each names only what is still missing.
    max_retries: int = 2
    system_prompt: str = ""
    default_provider: Provider = "openrouter"

    @property
    def model_names(self) -> list[str]:
        return [m.name for m in self.models]


@dataclass
class JudgeConfig:
    model: str
    provider: Provider = "openrouter"
    max_tokens: int = 2048
    # Word budget the judge is asked to keep the rolling memo within. It is injected
    # into every generator call of the next iteration, so its length is prompt real
    # estate, not just cost.
    memo_word_budget: int = DEFAULT_MEMO_WORD_BUDGET
    # How many of each batch's samples the judge is shown (balanced across the two
    # classes where possible). 0 = all — with n_batches × batch_size samples per
    # iteration that can make the prompt very large.
    max_samples_per_batch: int = 6
    system_prompt: str = ""


@dataclass
class ProbeConfig:
    # An existing pickled probe to warm-start from. Optional: when absent (or the file
    # doesn't exist) the loop trains the first probe from base_training_data using the
    # fields below.
    path: Path | None = None
    model: str | None = None
    layer: int | None = None
    pos_class_label: str | None = None
    neg_class_label: str | None = None
    description: str | None = None
    architecture: str | None = None
    # Number of probes to fit at every training step and average into one
    # score-averaging deep ensemble (see agentic_redteam.ensemble); member i trains
    # under the pinned ENSEMBLE_SEEDS[i]. None = inherit from the probe being retrained
    # (1 for the initial probe). Capped at MAX_ENSEMBLE_SIZE.
    ensemble_size: int | None = None


@dataclass
class LoopConfig:
    """The outer generate → score → retrain → guide loop."""

    iterations: int = 3
    # A batch is ACCEPTED (its samples join the training set) when the mean dev AUROC
    # of a probe trained on base ∪ accepted-so-far ∪ batch exceeds the current probe's
    # by more than this.
    min_auroc_gain: float = DEFAULT_MIN_AUROC_GAIN
    # |Δ| at or below this is reported to the judge as an EXHAUSTED regime — training
    # on that kind of sample moved nothing, so the judge should steer elsewhere.
    exhausted_gain: float = DEFAULT_EXHAUSTED_GAIN


@dataclass
class EvalConfig:
    """Dataset-loading transforms and eval knobs.

    ``combine_consecutive_messages`` / ``convert_tool_to_assistant`` mirror tuberlens'
    ``LabelledDataset`` loader flags and are applied to the training data, the
    generated samples, the dev set AND the eval splits, so the probe trains and is
    scored on one message representation. ``eval_max_samples`` is the balanced
    subsample per eval split (0 = full; None = the CLI default).
    ``data_description`` is optional free text about what the eval splits hold; it is
    shown to the judge so its directions can target coverage across them.
    """

    combine_consecutive_messages: bool = False
    convert_tool_to_assistant: bool = False
    eval_max_samples: int | None = None
    data_description: str = ""


@dataclass
class ValidationConfig:
    """Held-out dev data — the fit's validation set AND the ΔAUROC scoring set.

    ``dev_data`` is a JSONL, or a directory whose ``*.jsonl`` files are each a split
    (auto-discovered and concatenated). Its ``labels`` strings must be the probe's own
    class labels, and it must be disjoint from the eval splits. Required by the loop:
    every batch is judged by how it moves the dev AUROC. The CLI ``--dev-data`` flag
    overrides.
    """

    dev_data: Path | None = None


@dataclass
class KaggleConfig:
    """Precomputed eval (and optionally dev) activations to pull from Kaggle.

    See ``kaggle_activations.py``. ``eval_dataset_slug`` / ``eval_file_name`` are
    templates formatted with ``split=<stem>`` and ``slug=<hyphenated stem>``; the dev
    pair does the same for ``validation.dev_data`` and must be given together. Only
    valid with full eval splits (``eval.eval_max_samples: 0``).
    """

    owner: str
    eval_dataset_slug: str
    eval_file_name: str
    dev_dataset_slug: str | None = None
    dev_file_name: str | None = None


@dataclass
class OutputConfig:
    run_dir: Path
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    comparison_csv: Path | None = None
    activations_cache_dir: Path | None = None
    base_activation_cache_dir: Path | None = None

    @property
    def batches_path(self) -> Path:
        return self.run_dir / "batches.jsonl"

    @property
    def guidance_path(self) -> Path:
        return self.run_dir / "guidance.jsonl"

    @property
    def runlog_path(self) -> Path:
        return self.run_dir / "runlog.jsonl"


@dataclass
class LoopRunConfig:
    generator: GeneratorConfig
    judge: JudgeConfig
    probe: ProbeConfig
    loop: LoopConfig
    output: OutputConfig
    source_path: Path
    eval: EvalConfig = field(default_factory=EvalConfig)
    kaggle: KaggleConfig | None = None
    validation: ValidationConfig = field(default_factory=ValidationConfig)


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


def _parse_generator_models(
    raw_models: list, default_provider: Provider
) -> list[GeneratorModel]:
    """Each entry is either a bare string (uses default_provider) or {name, provider}."""
    out: list[GeneratorModel] = []
    for i, entry in enumerate(raw_models):
        if isinstance(entry, str):
            out.append(GeneratorModel(name=entry, provider=default_provider))
        elif isinstance(entry, dict):
            if "name" not in entry:
                raise ValueError(
                    f"generator.models[{i}]: dict entries require a 'name' field"
                )
            provider = entry.get("provider", default_provider)
            out.append(
                GeneratorModel(
                    name=str(entry["name"]),
                    provider=_validate_provider(
                        str(provider), f"generator.models[{i}].provider"
                    ),
                )
            )
        else:
            raise ValueError(
                f"generator.models[{i}]: entries must be strings or dicts, "
                f"got {type(entry).__name__}"
            )
    return out


def _positive_int(raw: object, where: str, minimum: int = 1) -> int:
    value = int(raw)  # type: ignore[arg-type]
    if value < minimum:
        raise ValueError(f"{where} must be >= {minimum}; got {value}")
    return value


def load_config(path: str | Path) -> LoopRunConfig:
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text)
    sections = _split_sections(body)

    generator_prompt = sections.get("generator")
    judge_prompt = sections.get("judge")
    if generator_prompt is None or judge_prompt is None:
        raise ValueError(
            "Config body must contain '# Generator' and '# Judge' top-level sections"
        )

    config_dir = path.parent

    def _resolve(p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (config_dir / p).resolve()

    g = frontmatter.get("generator") or {}
    j = frontmatter.get("judge") or {}
    pr = frontmatter.get("probe") or {}
    lp = frontmatter.get("loop") or {}
    o = frontmatter.get("output") or {}
    ev = frontmatter.get("eval") or {}
    kg = frontmatter.get("kaggle") or {}
    va = frontmatter.get("validation") or {}

    if "models" not in g or not g["models"]:
        raise ValueError("generator.models must be a non-empty list")
    if "model" not in j:
        raise ValueError("judge.model is required")
    if "run_dir" not in o:
        raise ValueError("output.run_dir is required")

    generator_default_provider = _validate_provider(
        str(g.get("provider", "openrouter")), "generator.provider"
    )
    generator_models = _parse_generator_models(list(g["models"]), generator_default_provider)
    judge_provider = _validate_provider(
        str(j.get("provider", "openrouter")), "judge.provider"
    )

    batch_size = _positive_int(g.get("batch_size", 20), "generator.batch_size", 2)
    if batch_size % 2 != 0:
        raise ValueError(
            f"generator.batch_size must be even (half the batch per class); got {batch_size}"
        )

    kaggle_cfg = None
    if kg:
        missing = [k for k in ("owner", "eval_dataset_slug", "eval_file_name") if not kg.get(k)]
        if missing:
            raise ValueError(f"kaggle section is missing required key(s): {missing}")
        if ev.get("eval_max_samples") not in (0, None):
            raise ValueError(
                "kaggle: requires eval.eval_max_samples: 0 (full splits), but got "
                f"{ev['eval_max_samples']}."
            )
        dev_slug, dev_file = kg.get("dev_dataset_slug"), kg.get("dev_file_name")
        if bool(dev_slug) != bool(dev_file):
            raise ValueError(
                "kaggle: dev_dataset_slug and dev_file_name must be given together "
                f"(got dev_dataset_slug={dev_slug!r}, dev_file_name={dev_file!r})"
            )
        if dev_slug and not va.get("dev_data"):
            raise ValueError(
                "kaggle: dev_dataset_slug/dev_file_name are set but validation.dev_data "
                "is not — there is no dev set to fetch activations for."
            )
        kaggle_cfg = KaggleConfig(
            owner=str(kg["owner"]),
            eval_dataset_slug=str(kg["eval_dataset_slug"]),
            eval_file_name=str(kg["eval_file_name"]),
            dev_dataset_slug=str(dev_slug) if dev_slug else None,
            dev_file_name=str(dev_file) if dev_file else None,
        )

    raw_ensemble_size = pr.get("ensemble_size")
    probe_ensemble_size: int | None = None
    if raw_ensemble_size is not None:
        probe_ensemble_size = int(raw_ensemble_size)
        if not 1 <= probe_ensemble_size <= MAX_ENSEMBLE_SIZE:
            raise ValueError(
                f"probe.ensemble_size must be between 1 and {MAX_ENSEMBLE_SIZE}; "
                f"got {probe_ensemble_size}"
            )

    min_gain = float(lp.get("min_auroc_gain", DEFAULT_MIN_AUROC_GAIN))
    exhausted_gain = float(lp.get("exhausted_gain", DEFAULT_EXHAUSTED_GAIN))
    if exhausted_gain < 0:
        raise ValueError(f"loop.exhausted_gain must be >= 0; got {exhausted_gain}")

    return LoopRunConfig(
        generator=GeneratorConfig(
            models=generator_models,
            n_batches=_positive_int(g.get("n_batches", 5), "generator.n_batches"),
            batch_size=batch_size,
            concurrency=_positive_int(g.get("concurrency", 5), "generator.concurrency"),
            max_tokens=_positive_int(g.get("max_tokens", 8192), "generator.max_tokens"),
            max_sample_tokens=int(g.get("max_sample_tokens", MAX_ACTIVATION_TOKENS)),
            max_retries=_positive_int(g.get("max_retries", 2), "generator.max_retries", 0),
            system_prompt=generator_prompt,
            default_provider=generator_default_provider,
        ),
        judge=JudgeConfig(
            model=str(j["model"]),
            provider=judge_provider,
            max_tokens=_positive_int(j.get("max_tokens", 2048), "judge.max_tokens"),
            memo_word_budget=_positive_int(
                j.get("memo_word_budget", DEFAULT_MEMO_WORD_BUDGET), "judge.memo_word_budget"
            ),
            max_samples_per_batch=_positive_int(
                j.get("max_samples_per_batch", 6), "judge.max_samples_per_batch", 0
            ),
            system_prompt=judge_prompt,
        ),
        probe=ProbeConfig(
            path=_resolve(pr["path"]) if pr.get("path") else None,
            model=pr.get("model"),
            layer=int(pr["layer"]) if pr.get("layer") is not None else None,
            pos_class_label=pr.get("pos_class_label"),
            neg_class_label=pr.get("neg_class_label"),
            description=pr.get("description"),
            architecture=pr.get("architecture"),
            ensemble_size=probe_ensemble_size,
        ),
        loop=LoopConfig(
            iterations=_positive_int(lp.get("iterations", 3), "loop.iterations", 0),
            min_auroc_gain=min_gain,
            exhausted_gain=exhausted_gain,
        ),
        output=OutputConfig(
            run_dir=_resolve(o["run_dir"]),
            run_id=o.get("run_id") or uuid.uuid4().hex[:8],
            comparison_csv=_resolve(o["comparison_csv"]) if o.get("comparison_csv") else None,
            activations_cache_dir=(
                _resolve(o["activations_cache_dir"]) if o.get("activations_cache_dir") else None
            ),
            base_activation_cache_dir=(
                _resolve(o["base_activation_cache_dir"])
                if o.get("base_activation_cache_dir")
                else None
            ),
        ),
        source_path=path,
        eval=EvalConfig(
            combine_consecutive_messages=bool(ev.get("combine_consecutive_messages", False)),
            convert_tool_to_assistant=bool(ev.get("convert_tool_to_assistant", False)),
            eval_max_samples=(
                int(ev["eval_max_samples"]) if ev.get("eval_max_samples") is not None else None
            ),
            data_description=str(ev.get("data_description") or "").strip(),
        ),
        kaggle=kaggle_cfg,
        validation=ValidationConfig(
            dev_data=_resolve(va["dev_data"]) if va.get("dev_data") else None,
        ),
    )
