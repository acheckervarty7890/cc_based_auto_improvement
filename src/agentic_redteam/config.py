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
      max_sample_tokens: 1024        # drop submissions longer than this (0 = no cap)
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
      ensemble_size: 5                # optional (1..10): fit n probes on the same
                                      # activations under the repo-pinned ENSEMBLE_SEEDS
                                      # and average their scores (score-averaging deep
                                      # ensemble). Applies to the initial training AND
                                      # every retrain. Omit to inherit whatever the base
                                      # probe had.
    preprocessing:                    # optional: collation-style preprocessing of red-team
      provider: openrouter            # successes before each retrain (filter + contrastive)
      model: anthropic/claude-sonnet-4.5
      max_concurrent: 50
      max_tokens: 2048
      filter_percentile: 0.8
      max_sample_tokens: 1024        # regenerate pairs longer than this (0 = no cap)
    output:
      jsonl_path: results/redteam.jsonl
      run_id: null
      comparison_csv: results/iter_run_comparison.csv   # optional (--eval); CLI flag overrides
      activations_cache_dir: results/eval_activations    # optional (--eval); CLI flag overrides
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

from agentic_redteam.ensemble import MAX_ENSEMBLE_SIZE
from agentic_redteam.llm_judge import DEFAULT_ITERATION_MEMO_WORD_BUDGET
from agentic_redteam.token_budget import MAX_ACTIVATION_TOKENS

ErrorType = Literal["false_positive", "false_negative"]
Provider = Literal["claude_sdk", "openrouter"]
Interface = Literal["tools", "prompt"]

_VALID_PROVIDERS = ("claude_sdk", "openrouter")
_VALID_INTERFACES = ("tools", "prompt")


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
    # sessions_per_model: concurrent copies of EACH model launched within each round.
    #   >1 runs multiple independent sessions of the same model in parallel (still
    #   bounded by `concurrency`), without duplicating the model in `models` and
    #   without disabling round_summaries — the rounds stay sequential, so the rolling
    #   memo is unaffected. All copies share the JsonlStore (dedup) and write with the
    #   same round number, so their attempts fold into that round's summary.
    sessions_per_model: int = 1
    persistence_from_last_rounds: int | None = None  # None = show all rounds
    system_prompt: str = ""
    # view_past_attempts knobs (see ViewSampler):
    #   view_reshuffle: master switch for periodic random reshuffling. When False,
    #     the attacker is instead shown the most-recent successful/unsuccessful
    #     attempts (recency, not a random draw), and training seeds are used only
    #     as a fallback for the successful half when there are no real successes.
    #   view_reshuffle_interval: redraw the shown set every N submissions (only
    #     used when view_reshuffle is True).
    #   view_balance: when True, show ≈50/50 successful/unsuccessful (total = limit).
    #   view_training_seeds: blend true-class training examples into the successful pool.
    view_reshuffle: bool = True
    view_reshuffle_interval: int = 20
    view_balance: bool = True
    view_training_seeds: bool = True
    # near_dup_guard: submit-time mechanical clone guard. When True, a candidate
    #   whose first user turn is >= near_dup_threshold similar (difflib ratio) to any
    #   already-recorded SUCCESS is rejected before probe/judge run, so re-skinned
    #   winning templates never get scored or stored. Orthogonal to the view_* knobs
    #   (which only shape what the attacker sees). Default off — existing configs
    #   behave identically.
    near_dup_guard: bool = False
    near_dup_threshold: float = 0.8
    # near_dup_broadcast: when True (guard on), guard-rejected openers are shown to all
    #   sessions as a "recently rejected — avoid these" prompt block (cross-session
    #   steering). In-memory only, never written to the JSONL. Default off.
    near_dup_broadcast: bool = False
    # max_sample_tokens: submit-time length cap, in tokens of the PROBE's tokenizer
    #   (chat template applied, <bos> included). A longer submission is dropped before
    #   probe/judge run and never persisted. Defaults to tuberlens' get_activations
    #   max_length (1024), past which a conversation is truncated — scored, and later
    #   trained on, without its tail. 0 (or less) disables the check.
    max_sample_tokens: int = MAX_ACTIVATION_TOKENS
    # round_summaries: when True (default), rounds run SEQUENTIALLY — round N+1 waits
    #   for round N to finish, then the judge summarizes round N's attempts and that
    #   (cumulative) summary is injected into later rounds' attacker system prompts.
    #   When False, the legacy fully-concurrent scheduling is used (all round×model
    #   sessions launched at once) and no summaries are produced. Models within a
    #   round still run concurrently in both modes.
    round_summaries: bool = True
    # cross_iteration_memos: when True, a separate judge memo is carried ACROSS
    #   iterations (the round memo above resets every iteration). After each
    #   rotation the judge writes a memo about what was tried and what succeeded —
    #   knowing the probe is about to be retrained on those successes, so they
    #   should be treated as patched — and that memo is injected into the next
    #   iteration's attacker system prompts so they don't re-run trained-against
    #   ideas. Persisted to `<jsonl>.iteration_memos.jsonl`, so it also survives a
    #   process restart / --resume. Default off — existing configs behave identically.
    cross_iteration_memos: bool = False
    # How many of this iteration's successes (most recent) are shown to the judge when
    # it writes the cross-iteration memo. 0 = all (can make the judge prompt huge).
    cross_iteration_memo_max_successes: int = 30
    # Word budget the judge is given for the cross-iteration memo. The memo is injected
    # into EVERY attacker system prompt of the next iteration, so its length is a real
    # cost — the same trade the round memo's 200-word target settles. The 900 default is
    # unreachable at judge.max_tokens: 1024 (~625 words at this register's measured
    # density), so a run leaving it alone gets a memo truncated mid-sentence, and the loss
    # compounds through prior_memo. At <= 300 the prompt also switches to "drop the
    # weakest notes wholesale" instead of "compress everything".
    cross_iteration_memo_word_budget: int = DEFAULT_ITERATION_MEMO_WORD_BUDGET
    # interface: how the attacker is driven (OpenRouter only for "prompt").
    #   "tools" (default) — the model is handed the three tools and calls them itself.
    #   "prompt" — classical no-tool mode: no tool schemas are sent. The model emits one
    #     candidate conversation per turn as a fenced JSON block; the probe info and a
    #     view_past_attempts sample are injected into the prompt each turn (always shown,
    #     not called by the model). The submitted conversation still runs through the exact
    #     same probe+judge scoring/persistence path as tools mode.
    interface: Interface = "tools"
    # view_limit: number of past attempts injected each turn in prompt mode (matches the
    #   tools-mode fallback of 10). Unused in tools mode, where the model picks the count.
    view_limit: int = 10
    # Dump the exact prompt sent to the attacker each turn to
    # <jsonl>.prompts.jsonl (prompt mode only). Off by default: the file holds
    # every prompt in full, so it grows much faster than the JSONL.
    capture_prompts: bool = False
    # batch_submissions: ask for ALL `max_turns` candidate conversations in ONE reply
    #   instead of one per turn (prompt mode only; see _run_openrouter_prompt_batch_model).
    #   The session then makes a single API call and ends — the attacker never sees a
    #   probe/judge verdict, so every conversation in the batch is written blind. Off by
    #   default; the per-turn feedback loop is the standard prompt-mode behaviour.
    batch_submissions: bool = False
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
    # Withhold opposite-direction misclassifications (a false positive turned up
    # during a false_negative hunt, or vice versa) from the rolling round memo, so
    # the judge does not write up weaknesses that are unactionable this rotation.
    hide_opposite_direction: bool = True


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
    # Retries per record when a contrastive generation fails; after them the
    # source record and its missing pair are dropped (see generate_contrastive_dataset).
    max_generation_retries: int = 2
    # max_sample_tokens: length cap for a GENERATED contrastive pair, in tokens of the
    #   probe's tokenizer. A pair over it is not accepted: the generator is asked again
    #   and told how long its last attempt was (that retry comes out of
    #   max_generation_retries). Defaults to tuberlens' 1024-token activation cap, past
    #   which the pair would be trained on truncated. 0 (or less) disables the check.
    max_sample_tokens: int = MAX_ACTIVATION_TOKENS
    # When true, the contrastive-generation prompt tells the model that the label
    # is determined by the *assistant's* response (not the user's message) and to
    # focus its edits on the assistant turns. Only correct for concepts that are a
    # property of the assistant reply (e.g. harmful_to_human); leave false for
    # whole-scenario concepts (e.g. high-stakes).
    assistant_centric: bool = False
    # Optional concept detail injected verbatim into the contrastive-generation
    # system prompt, so the generator gets more than just the two label strings.
    #   concept_description — what the concept *is*; shown for both directions.
    #   label_guidance      — keyed by class label, shown when generating *toward*
    #                         that label (e.g. "involve high-stakes consequences
    #                         (significant financial loss, legal exposure, ...)").
    # Both are free text (bullets welcome). Non-empty guidance is folded into the
    # contrastive cache key, so editing it regenerates pairs instead of silently
    # reusing ones written under the old prompt.
    concept_description: str = ""
    label_guidance: dict[str, str] = field(default_factory=dict)


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
    # Number of probes to fit at every training/retraining step and average into
    # one score-averaging deep ensemble (see agentic_redteam.ensemble); member i
    # trains under the pinned ENSEMBLE_SEEDS[i]. 1 = the ordinary single probe
    # (which trains under the run's own --seed, as it always has). None means
    # "unset", which lets a retrain inherit whatever the probe it is retraining
    # from had, exactly as `architecture` does. Capped at MAX_ENSEMBLE_SIZE.
    ensemble_size: int | None = None

    @property
    def error_type(self) -> ErrorType:
        """Convenience accessor when there is exactly one error type."""
        return self.error_types[0]


@dataclass
class EvalConfig:
    """Dataset-loading transforms applied when reading the eval split JSONLs.

    Both mirror tuberlens' ``LabelledDataset`` loader flags (and the
    ``collate_train_evaluate.py`` eval step): they reshape each conversation
    *as it is loaded for evaluation only* — training/retraining data is left
    untouched, exactly as in the reference script.

    - ``combine_consecutive_messages``: merge adjacent messages from the same
      role into one.
    - ``convert_tool_to_assistant``: rewrite ``tool`` messages as ``assistant``
      messages (applied before combining).
    - ``eval_max_samples``: balanced subsample size per eval split (``0`` =
      full split). ``None`` means "unset in config" so the CLI falls back to
      its ``--eval-max-samples`` default; the ``--eval-max-samples`` flag, when
      passed, overrides this.
    - ``data_description``: OPTIONAL free text describing what the eval splits
      hold — in particular the distinct KINDS of conversation the probe is
      scored on. It changes no data path at all; it is prompt material, handed
      to the judge's two SUMMARIZERS (never to its classification prompt, which
      must not learn about the test set) so the rolling and cross-iteration
      memos are organized around those kinds and name the ones a round or cycle
      left untouched. The memos are what a later attacker session reads, so this
      is the one place coverage across the eval splits can be steered. Unset
      (the default), every judge prompt is byte-identical to what it was before
      this knob existed.
    """

    combine_consecutive_messages: bool = False
    convert_tool_to_assistant: bool = False
    eval_max_samples: int | None = None
    data_description: str = ""


@dataclass
class ValidationConfig:
    """Held-out dev data to use as the probe fit's validation set.

    ``dev_data`` is a JSONL, or a directory whose ``*.jsonl`` files are each a split
    (auto-discovered and concatenated, the way ``evaluate_probe`` discovers eval
    splits). Its ``labels`` strings must be the probe's own class labels.

    When set, the validation set is that dev data **alone** — the base training data
    and the red-team successes are no longer split, and train in full. The point is a
    validation set that does not move: with the default ``test_size`` slice, a share
    of every iteration's red-team successes lands in validation, so the set the probe
    early-stops against changes shape every retrain and the best-epoch checkpoints
    are not comparable across iterations.

    The dev data must be disjoint from the eval splits — otherwise the fit selects
    its checkpoint on the test set.
    """

    dev_data: Path | None = None


@dataclass
class KaggleConfig:
    """Precomputed eval activations to pull from Kaggle instead of recomputing.

    ``eval_dataset_slug`` and ``eval_file_name`` are templates formatted with
    ``split=<name>`` (the eval split's filename stem), e.g. ``"{split}gemmaevalpt"``
    and ``"{split}-gemmaeval.pt"``. Present only to skip the (very expensive)
    activation extraction for large probe models — see ``kaggle_activations.py``.

    ``dev_dataset_slug`` / ``dev_file_name`` do the same for the held-out DEV set
    (``validation.dev_data``), which the probe fit early-stops against. They are
    optional and independent of the eval pair: set them and the dev activations are
    downloaded per split and assembled into the single content-hashed dev blob the
    fit looks for, so nothing is extracted locally; leave them out and the dev set is
    computed on the box like any other split. Both must be given together, and only
    alongside ``validation.dev_data`` — there is nothing to fetch otherwise.

    Requires credentials: ``KAGGLE_CONFIG_DIR`` pointing at the DIRECTORY holding
    ``kaggle.json``, or ``KAGGLE_API_TOKEN``. Only valid with full eval splits
    (``eval.eval_max_samples: 0``).
    """

    owner: str
    eval_dataset_slug: str
    eval_file_name: str
    # Optional dev-set counterparts. Both None (extract dev locally) or both set.
    dev_dataset_slug: str | None = None
    dev_file_name: str | None = None


@dataclass
class OutputConfig:
    jsonl_path: Path
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    # Optional eval-output paths (used by the iterative loop with --eval). When None,
    # the CLI falls back to <results-dir>-derived defaults. CLI flags, when passed,
    # take precedence over these.
    comparison_csv: Path | None = None
    activations_cache_dir: Path | None = None
    # Optional cache dir for the base *training* split's activations (computed once
    # per run, reused every retrain). When None, the CLI falls back to
    # <probe-out-dir>/base_activation_cache; the --base-activation-cache-dir flag
    # takes precedence over this.
    base_activation_cache_dir: Path | None = None


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
    # Dataset-loading transforms applied to eval splits only (see EvalConfig).
    eval: EvalConfig = field(default_factory=EvalConfig)
    # Optional: pull precomputed eval activations from Kaggle (see KaggleConfig).
    kaggle: KaggleConfig | None = None
    # Optional held-out dev set used as the probe fit's validation set (see
    # ValidationConfig). Empty (dev_data=None) keeps the test_size-slice behaviour.
    validation: ValidationConfig = field(default_factory=ValidationConfig)

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


def _validate_interface(value: str, where: str) -> Interface:
    if value not in _VALID_INTERFACES:
        raise ValueError(
            f"{where}: interface must be one of {_VALID_INTERFACES!r}, got {value!r}"
        )
    return value  # type: ignore[return-value]


def _parse_label_guidance(raw: object) -> dict[str, str]:
    """Parse ``preprocessing.label_guidance``: a {class label: free text} mapping.

    Keys are the *raw* class labels (as they appear in the probe metadata, e.g.
    ``high-stakes`` / ``harmful_to_human``); values are injected verbatim into the
    generation prompt when writing *toward* that label. Empty/blank values are
    dropped so they can't perturb the cache key with a no-op.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            "preprocessing.label_guidance must be a mapping of class label → text, "
            f"got {type(raw).__name__}"
        )
    guidance: dict[str, str] = {}
    for label, text in raw.items():
        if text is None:
            continue
        if not isinstance(text, str):
            raise ValueError(
                f"preprocessing.label_guidance[{label!r}] must be a string, "
                f"got {type(text).__name__}"
            )
        text = text.strip()
        if text:
            guidance[str(label)] = text
    return guidance


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
    ev = frontmatter.get("eval") or {}
    kg = frontmatter.get("kaggle") or {}
    va = frontmatter.get("validation") or {}

    if "models" not in a or not a["models"]:
        raise ValueError("attacker.models must be a non-empty list")
    if int(a.get("sessions_per_model", 1)) < 1:
        raise ValueError("attacker.sessions_per_model must be >= 1")
    if "model" not in j:
        raise ValueError("judge.model is required")
    if "jsonl_path" not in o:
        raise ValueError("output.jsonl_path is required")

    attacker_default_provider = _validate_provider(
        str(a.get("provider", "claude_sdk")), "attacker.provider"
    )
    attacker_interface = _validate_interface(
        str(a.get("interface", "tools")), "attacker.interface"
    )
    attacker_models = _parse_attacker_models(list(a["models"]), attacker_default_provider)
    # "prompt" (no-tool) mode is implemented only on the OpenRouter driver; the
    # claude_sdk path is MCP/tool-native and has no analog.
    if attacker_interface == "prompt" and any(
        m.provider == "claude_sdk" for m in attacker_models
    ):
        raise ValueError(
            "attacker.interface: 'prompt' mode is only supported for openrouter models, "
            "but one or more models resolve to provider 'claude_sdk'."
        )
    attacker_batch_submissions = bool(a.get("batch_submissions", False))
    # Raise rather than ignore: this knob changes what the session does, so silently
    # dropping it under interface: tools would run a config that looks like a batch
    # arm as a per-turn one, and nothing downstream would say so.
    if attacker_batch_submissions and attacker_interface != "prompt":
        raise ValueError(
            "attacker.batch_submissions requires attacker.interface: prompt "
            f"(got interface: {attacker_interface!r}); the tools-mode drivers let the "
            "model decide when to submit, so there is no single reply to batch."
        )
    judge_provider = _validate_provider(
        str(j.get("provider", "claude_sdk")), "judge.provider"
    )

    kaggle_cfg = None
    if kg:
        missing = [k for k in ("owner", "eval_dataset_slug", "eval_file_name") if not kg.get(k)]
        if missing:
            raise ValueError(f"kaggle section is missing required key(s): {missing}")
        # Full splits only: the published blobs cover whole splits, so a subsampled
        # run would load activations for rows it isn't scoring. Catch it at parse
        # time rather than after the first (long) red-team phase.
        if ev.get("eval_max_samples") not in (0, None):
            raise ValueError(
                "kaggle: requires eval.eval_max_samples: 0 (full splits), but got "
                f"{ev['eval_max_samples']}."
            )
        # The dev pair is optional, but half of it is always a mistake: one template
        # without the other cannot name a file, and either without validation.dev_data
        # would fetch activations for a validation set the run does not use.
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

    # probe.ensemble_size: how many independently-seeded probes each train/retrain
    # fits and averages into one score. Bounded here (not just at the CLI) because
    # every entry point that trains reads it from the config, and a typo like `100`
    # would otherwise turn one retrain into a hundred fits.
    raw_ensemble_size = pr.get("ensemble_size")
    probe_ensemble_size: int | None = None
    if raw_ensemble_size is not None:
        probe_ensemble_size = int(raw_ensemble_size)
        if not 1 <= probe_ensemble_size <= MAX_ENSEMBLE_SIZE:
            raise ValueError(
                f"probe.ensemble_size must be between 1 and {MAX_ENSEMBLE_SIZE}; "
                f"got {probe_ensemble_size}"
            )

    # attacker.cross_iteration_memo_word_budget: the memo lands in every attacker system
    # prompt of the next iteration, so this is a prompt-real-estate knob, not just a cost
    # one. Rejected at parse time rather than clamped: a 0 or negative budget would ask
    # the judge for a memo it cannot write, and the run only finds out an iteration later.
    cross_iteration_memo_word_budget = int(
        a.get("cross_iteration_memo_word_budget", DEFAULT_ITERATION_MEMO_WORD_BUDGET)
    )
    if cross_iteration_memo_word_budget < 1:
        raise ValueError(
            "attacker.cross_iteration_memo_word_budget must be >= 1; got "
            f"{cross_iteration_memo_word_budget}"
        )

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
            max_generation_retries=int(pp.get("max_generation_retries", 2)),
            max_sample_tokens=int(
                pp.get("max_sample_tokens", MAX_ACTIVATION_TOKENS)
            ),
            assistant_centric=bool(pp.get("assistant_centric", False)),
            concept_description=str(pp.get("concept_description", "") or "").strip(),
            label_guidance=_parse_label_guidance(pp.get("label_guidance")),
        )

    return RedteamConfig(
        attacker=AttackerConfig(
            models=attacker_models,
            max_turns=int(a.get("max_turns", 30)),
            batch_target=int(a.get("batch_target", 10)),
            rounds=int(a.get("rounds", 1)),
            concurrency=int(a.get("concurrency", 1)),
            sessions_per_model=int(a.get("sessions_per_model", 1)),
            persistence_from_last_rounds=persistence_from_last_rounds,
            view_reshuffle=bool(a.get("view_reshuffle", True)),
            view_reshuffle_interval=int(a.get("view_reshuffle_interval", 20)),
            view_balance=bool(a.get("view_balance", True)),
            view_training_seeds=bool(a.get("view_training_seeds", True)),
            near_dup_guard=bool(a.get("near_dup_guard", False)),
            near_dup_threshold=float(a.get("near_dup_threshold", 0.8)),
            near_dup_broadcast=bool(a.get("near_dup_broadcast", False)),
            max_sample_tokens=int(a.get("max_sample_tokens", MAX_ACTIVATION_TOKENS)),
            round_summaries=bool(a.get("round_summaries", True)),
            cross_iteration_memos=bool(a.get("cross_iteration_memos", False)),
            cross_iteration_memo_max_successes=int(
                a.get("cross_iteration_memo_max_successes", 30)
            ),
            cross_iteration_memo_word_budget=cross_iteration_memo_word_budget,
            interface=attacker_interface,
            view_limit=int(a.get("view_limit", 10)),
            capture_prompts=bool(a.get("capture_prompts", False)),
            batch_submissions=attacker_batch_submissions,
            system_prompt=attacker_prompt,
            default_provider=attacker_default_provider,
        ),
        judge=JudgeConfig(
            model=j["model"],
            provider=judge_provider,
            max_tokens=int(j.get("max_tokens", 1024)),
            confidence_threshold=int(j.get("confidence_threshold", 7)),
            system_prompt=judge_prompt,
            hide_opposite_direction=bool(j.get("hide_opposite_direction", True)),
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
            ensemble_size=probe_ensemble_size,
        ),
        output=OutputConfig(
            jsonl_path=_resolve(o["jsonl_path"]),
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
        preprocessing=preprocessing,
        eval=EvalConfig(
            combine_consecutive_messages=bool(ev.get("combine_consecutive_messages", False)),
            convert_tool_to_assistant=bool(ev.get("convert_tool_to_assistant", False)),
            eval_max_samples=(
                int(ev["eval_max_samples"]) if ev.get("eval_max_samples") is not None else None
            ),
            # Stripped, and normalized to "" when absent/blank — `llm_judge` keys the
            # whole feature off truthiness, so a key present but empty must read the
            # same as no key at all.
            data_description=str(ev.get("data_description") or "").strip(),
        ),
        kaggle=kaggle_cfg,
        validation=ValidationConfig(
            dev_data=_resolve(va["dev_data"]) if va.get("dev_data") else None,
        ),
    )
