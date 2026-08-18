"""Console entry points for `redteam` and `iterative-retrain`."""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

from agentic_redteam.attacker import run_redteam_sync
from agentic_redteam.circuit_breaker import OpenRouterOutageError
from agentic_redteam.config import load_config
from agentic_redteam.evaluation import (
    DEFAULT_EVAL_MAX_SAMPLES,
    DEFAULT_SEED,
    evaluate_probe,
    seed_everything,
)
from agentic_redteam.ensemble import ENSEMBLE_SEEDS, MAX_ENSEMBLE_SIZE
from agentic_redteam.retrain import (
    DEFAULT_FRESH_PROBE_ARCH,
    retrain_probe,
    train_initial_probe,
)


# Exit code for "OpenRouter is down / out of credits", distinct from 1 (ordinary
# failure) and 2 (usage error) so a wrapper script can tell the cases apart.
OUTAGE_EXIT_CODE = 3


def _exit_on_outage(fn):
    """Turn an :class:`OpenRouterOutageError` into a clean non-zero exit.

    The breaker firing means the run stopped on purpose, not that it crashed, so
    report it as a message rather than a traceback. Little is lost by stopping:
    the JSONL is append-only, and the per-(iteration, error_type) phase markers
    plus the per-round progress sidecar make ``--resume`` pick up at the round
    this left off on once credits/keys are restored.
    """

    @functools.wraps(fn)
    def wrapper(argv: list[str] | None = None) -> int:
        try:
            return fn(argv)
        except OpenRouterOutageError as e:
            print(
                f"\n!!! ABORTED — OpenRouter is not usable.\n    {e}\n\n"
                "    No further rounds, retrains or evals were run; continuing would "
                "only have produced\n    empty red-team rounds and probes trained on "
                "nothing. Fix the account/key, then\n    re-run the same command with "
                "--resume to continue from the last completed round.",
                file=sys.stderr,
            )
            return OUTAGE_EXIT_CODE

    return wrapper


def _free_gpu() -> None:
    """Release reserved GPU **and host** memory between heavy phases.

    train/eval each load a gemma-sized LLM internally (tuberlens); on return the
    model is dereferenced but torch's caching allocator keeps its GPU memory
    reserved. Since every load uses device_map="auto" and, unpinned (the default),
    re-infers the layer split from *free* GPU memory, that leftover reservation
    pushes the next load into CPU/disk offload and ~5-10x slower forwards.
    Collecting + emptying the cache between phases keeps each load on a clean GPU.

    **The host-RAM half matters just as much, and only the GPU half used to be done.**
    accelerate sizes the ``"cpu"`` device's budget from ``psutil.virtual_memory()``'s
    *available* figure at load time, and a phase that held tens of GB of CPU-offloaded
    shards leaves that figure depressed: ``gc.collect()`` frees the Python objects but
    glibc keeps the arenas mapped rather than returning them to the OS. So the *next*
    load sees less RAM than really exists and spills the remainder to **disk** — which
    is far slower than CPU offload, and is what turned a 3.2 s/attempt rotation into a
    90-113 s/attempt one across an error-type boundary on a 57 GB box.
    ``malloc_trim(0)`` hands the free arenas back. It is glibc-only and purely an
    optimisation, so every failure mode (musl, macOS, no ctypes) is swallowed.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001 — best-effort; not available off glibc
        pass


def _latest_probe_iteration(probe_out_dir: Path) -> tuple[int, Path] | None:
    """Return ``(N, path)`` for the highest-numbered ``probe_iter{N}.pkl`` in
    ``probe_out_dir``, or ``None`` if no such file exists.

    Used to resume an interrupted iterative run: ``probe_iter{i}.pkl`` is the
    probe red-teamed at iteration ``i`` (``probe_iter0.pkl`` = initial), so the
    latest one on disk is where the next iteration should pick up.
    """
    import re

    best: tuple[int, Path] | None = None
    for p in probe_out_dir.glob("probe_iter*.pkl"):
        m = re.fullmatch(r"probe_iter(\d+)\.pkl", p.name)
        if m is None:
            continue
        n = int(m.group(1))
        if best is None or n > best[0]:
            best = (n, p)
    return best


def _redteam_phase_marker_path(probe_out_dir: Path, iteration: int, error_type: str) -> Path:
    """Path of the sentinel marking one ``(iteration, error_type)`` red-team phase
    as **complete**.

    ``probe_iter{N}.pkl`` is only written once the *retrain* finishes, so a run
    interrupted between red-teaming and retraining leaves no probe for that cycle
    and resume rewinds to the start of the iteration — re-running an already-
    finished (and expensive) red-team pass. This finer-grained marker, written
    after each error type's red-team rotation returns, lets resume skip a phase
    whose successes are already persisted in the JSONL (append-only + dedup, so
    the retrain reads them regardless). It is diagnostics-cheap JSON.
    """
    return probe_out_dir / f"redteam_done_iter{iteration}_{error_type}.marker"


@_exit_on_outage
def run_redteam_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one round of agentic red-teaming against a tuberlens probe."
    )
    parser.add_argument("config", type=Path, help="Path to the markdown config file")
    parser.add_argument(
        "--round", type=int, default=0, help="Base round number to record in the JSONL log"
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    print(f"Run ID: {config.output.run_id}")
    print(f"Probe:  {config.probe.path}")
    print(f"Error types: {config.probe.error_types}")
    print(f"Rounds per error type: {config.attacker.rounds}")
    print(f"Concurrency: {config.attacker.concurrency}")
    if config.attacker.persistence_from_last_rounds is not None:
        print(f"Persistence from last rounds: {config.attacker.persistence_from_last_rounds}")
    print(
        "Models: "
        + ", ".join(f"{m.name} ({m.provider})" for m in config.attacker.models)
    )
    print()

    total_new = 0
    for et in config.probe.error_types:
        jsonl_path = config.jsonl_path_for(et)
        print(f"\n--- Error type: {et} → {jsonl_path} ---")
        base_round = args.round * config.attacker.rounds
        summaries = run_redteam_sync(
            config, base_round_num=base_round, error_type=et, jsonl_path=jsonl_path,
            iteration=args.round,
        )
        for s in summaries:
            print(
                f"  {s.model}: {s.new_successes} new successes "
                f"({s.total_messages} agent messages; stop={s.stop_reason})"
            )
            total_new += s.new_successes
        _free_gpu()  # same reload boundary as the iterative loop — see _free_gpu

    print(f"\n=== Total new successes: {total_new} ===")
    return 0


@_exit_on_outage
def iterative_retrain_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train an initial probe, then iterate: red-team → retrain "
        "(→ optional eval), n times."
    )
    parser.add_argument("config", type=Path, help="Path to the markdown config file")
    parser.add_argument(
        "--iterations", type=int, default=2, help="Number of red-team + retrain cycles (n)"
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume an interrupted run at the finest checkpoint available: the latest "
        "probe_iterN.pkl in --probe-out-dir (red-team it next), then any finished "
        "(iteration, error_type) phase markers, then the individual rounds recorded in "
        "<jsonl>.rounds_done.jsonl (the rolling strategy memo is restored with them). "
        "Pass --no-resume to force a fresh run from the start.",
    )
    parser.add_argument(
        "--base-training-data",
        type=Path,
        required=True,
        help="JSONL/CSV used to train the initial probe and concatenated with red-team "
        "successes on every retrain",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the training data held out for validation (stable_train_test_split)",
    )
    parser.add_argument(
        "--split-field",
        type=str,
        default=None,
        help="Optional field kept grouped together when splitting train/validation",
    )
    parser.add_argument(
        "--dev-data",
        type=Path,
        default=None,
        help="Held-out dev data (a JSONL, or a directory of *.jsonl splits) to use as "
        "the probe fit's validation set. When given, NOTHING is held out of the base "
        "training data or the red-team successes — they train in full and --test-size / "
        "--split-field are ignored — so the validation set stays identical across "
        "iterations instead of absorbing a share of each iteration's successes. Must be "
        "disjoint from --eval-dataset-dir. Overrides validation.dev_data in config.",
    )
    parser.add_argument(
        "--base-data-fraction",
        type=float,
        default=1.0,
        help="Fraction (0, 1] of the base training data to ingest, chosen by a "
        "deterministic content-addressed random subsample (seeded by --seed) applied "
        "before the train/val split. 1.0 (default) uses all of it. Red-team successes "
        "are never subsampled.",
    )
    parser.add_argument(
        "--probe-out-dir",
        type=Path,
        default=Path("probes"),
        help="Where to write probes (probe_iter0.pkl = initial, probe_iterN.pkl per cycle)",
    )
    parser.add_argument(
        "--base-activation-cache-dir",
        type=Path,
        default=None,
        help="Cache dir for the base training split's activations (computed once, reused "
        "every retrain). Overrides output.base_activation_cache_dir in config; "
        "default <probe-out-dir>/base_activation_cache.",
    )
    parser.add_argument(
        "--layer", type=int, default=None, help="Layer to probe; overrides probe.layer in config"
    )
    parser.add_argument(
        "--probe-arch",
        nargs="?",
        const=DEFAULT_FRESH_PROBE_ARCH,
        default=None,
        help="Probe architecture (ProbeType name); overrides probe.architecture in config. "
        f"Bare flag uses {DEFAULT_FRESH_PROBE_ARCH!r}. On retrains, omitting inherits the "
        "current probe's architecture.",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=None,
        help=f"Fit this many probes (1..{MAX_ENSEMBLE_SIZE}) at every training/"
        "retraining step and average their scores into one score-averaging deep "
        "ensemble; the averaged score is what the threshold, the judge and the eval "
        "all see. Member i is trained under the repo-pinned ensemble.ENSEMBLE_SEEDS[i], "
        "the same on every run (--seed still governs the train/val split and eval "
        "subsampling). Overrides probe.ensemble_size in config. Omit both to inherit "
        "the size of the probe being retrained (1 for a plain probe).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate the initial probe and each retrained probe on the eval datasets",
    )
    parser.add_argument(
        "--eval-dataset-dir",
        type=Path,
        default=Path("eval_datasets"),
        help="Directory of local eval split JSONLs (used with --eval)",
    )
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=None,
        help="Balanced subsample size per eval split (used with --eval); 0 = full split. "
        f"Overrides eval.eval_max_samples in config (default {DEFAULT_EVAL_MAX_SAMPLES})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed (train/val split + reproducible eval subsampling)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Where to write the eval comparison CSV (used with --eval)",
    )
    parser.add_argument(
        "--activations-cache-dir",
        type=Path,
        default=None,
        help="Cache dir for eval activations; overrides output.activations_cache_dir in "
        "config (default <results-dir>/eval_activations)",
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=None,
        help="Path for the cross-round eval comparison CSV; overrides output.comparison_csv "
        "in config (default <results-dir>/iter_run_comparison.csv)",
    )
    parser.add_argument(
        "--combine-consecutive-messages",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Merge adjacent same-role messages in both the training data and eval splits; "
        "overrides eval.combine_consecutive_messages in config",
    )
    parser.add_argument(
        "--convert-tool-to-assistant",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Rewrite tool messages as assistant messages in both the training data and "
        "eval splits; overrides eval.convert_tool_to_assistant in config",
    )
    args = parser.parse_args(argv)

    if not 0.0 < args.base_data_fraction <= 1.0:
        parser.error(
            f"--base-data-fraction must be in (0, 1]; got {args.base_data_fraction}"
        )
    if args.ensemble_size is not None and not 1 <= args.ensemble_size <= MAX_ENSEMBLE_SIZE:
        parser.error(
            f"--ensemble-size must be between 1 and {MAX_ENSEMBLE_SIZE}; "
            f"got {args.ensemble_size}"
        )

    config = load_config(args.config)
    args.probe_out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    error_types = config.probe.error_types
    layer = args.layer if args.layer is not None else config.probe.layer
    arch = args.probe_arch if args.probe_arch is not None else config.probe.architecture
    # Precedence: --ensemble-size flag > config probe.ensemble_size > None. None is
    # passed straight through to retrain_probe, which reads the size off the probe it
    # is retraining (so a resumed run keeps whatever it was already building), and
    # means 1 for train_initial_probe, which has no probe to inherit from.
    ensemble_size = (
        args.ensemble_size
        if args.ensemble_size is not None
        else config.probe.ensemble_size
    )
    # Precedence: --eval-max-samples flag > config eval.eval_max_samples > default.
    # After resolution, 0 means "full split" (→ None).
    if args.eval_max_samples is not None:
        raw_eval_max_samples = args.eval_max_samples
    elif config.eval.eval_max_samples is not None:
        raw_eval_max_samples = config.eval.eval_max_samples
    else:
        raw_eval_max_samples = DEFAULT_EVAL_MAX_SAMPLES
    eval_max_samples = raw_eval_max_samples if raw_eval_max_samples > 0 else None
    # CLI flags (BooleanOptionalAction, default None) override the config eval: section.
    combine_consecutive_messages = (
        args.combine_consecutive_messages
        if args.combine_consecutive_messages is not None
        else config.eval.combine_consecutive_messages
    )
    convert_tool_to_assistant = (
        args.convert_tool_to_assistant
        if args.convert_tool_to_assistant is not None
        else config.eval.convert_tool_to_assistant
    )
    # Write the resolved values back so the red-team scoring path (which reads
    # config.eval inside run_redteam → ProbeJudge.load) honors --[no-]… overrides
    # too, not just the train/eval calls below.
    config.eval.combine_consecutive_messages = combine_consecutive_messages
    config.eval.convert_tool_to_assistant = convert_tool_to_assistant
    # Precedence: --dev-data flag > config validation.dev_data > None (test_size slice).
    dev_data_path = args.dev_data or config.validation.dev_data
    contrastive_cache_path = args.probe_out_dir / "contrastive_cache.jsonl"
    # Precedence: --base-activation-cache-dir flag > config output.base_activation_cache_dir
    # > <probe-out-dir>-derived default.
    base_activation_cache_dir = (
        args.base_activation_cache_dir
        or config.output.base_activation_cache_dir
        or (args.probe_out_dir / "base_activation_cache")
    )
    if config.preprocessing is not None:
        print(
            "Preprocessing enabled: filter_dataset + generate_contrastive_dataset "
            f"({config.preprocessing.model} via {config.preprocessing.provider})"
        )
    if ensemble_size is not None and ensemble_size > 1:
        print(
            f"Deep ensemble enabled: every train/retrain fits {ensemble_size} probes "
            f"on the same activations under the pinned training seeds "
            f"{list(ENSEMBLE_SEEDS[:ensemble_size])}; their mean score is the probe's "
            "score. (--seed still governs the train/val split and eval subsampling.)"
        )
    if dev_data_path is not None:
        print(
            f"Validation: held-out dev data at {dev_data_path} (base training data and "
            "red-team successes train in full; --test-size/--split-field ignored)"
        )
    if combine_consecutive_messages or convert_tool_to_assistant:
        print(
            "Dataset message transforms (training + eval + red-team scoring): "
            f"combine_consecutive_messages={combine_consecutive_messages}, "
            f"convert_tool_to_assistant={convert_tool_to_assistant}"
        )

    # ---- Step 1: obtain the initial probe ----
    # Resume first: if a previous run already wrote probe_iterN.pkl files into
    # --probe-out-dir, pick up from the latest one (red-team it next) instead of
    # retraining/warm-starting iter0. start_iter is the iteration index whose
    # input probe is current_probe_path (probe_iter{i}.pkl feeds iteration i).
    resume = _latest_probe_iteration(args.probe_out_dir) if args.resume else None
    if resume is not None:
        start_iter, current_probe_path = resume
        print(
            f"Resuming from existing probe iteration {start_iter}: {current_probe_path} "
            f"(red-teaming will start at iteration {start_iter}). "
            "Pass --no-resume for a fresh run."
        )
    elif config.probe.path is not None and Path(config.probe.path).exists():
        start_iter = 0
        current_probe_path = Path(config.probe.path)
        print(f"Warm-starting from existing probe: {current_probe_path}")
    else:
        start_iter = 0
        missing = [
            name
            for name, value in (
                ("probe.model", config.probe.model),
                ("probe.layer", layer),
                ("probe.pos_class_label", config.probe.pos_class_label),
                ("probe.neg_class_label", config.probe.neg_class_label),
            )
            if value is None
        ]
        if missing:
            parser.error(
                "Training the initial probe from scratch requires "
                + ", ".join(missing)
                + " — set them in the config probe: section, or point probe.path at an "
                "existing probe to warm-start."
            )
        current_probe_path = args.probe_out_dir / "probe_iter0.pkl"
        print(f"Training initial probe → {current_probe_path}")
        train_initial_probe(
            base_training_data_path=args.base_training_data,
            model_name=config.probe.model,
            layer=layer,
            new_probe_path=current_probe_path,
            pos_class_label=config.probe.pos_class_label,
            neg_class_label=config.probe.neg_class_label,
            probe_description=config.probe.description,
            probe_spec=arch,
            test_size=args.test_size,
            split_field=args.split_field,
            dev_data_path=dev_data_path,
            seed=args.seed,
            base_data_fraction=args.base_data_fraction,
            ensemble_size=ensemble_size or 1,
            base_activation_cache_dir=base_activation_cache_dir,
            combine_consecutive_messages=combine_consecutive_messages,
            convert_tool_to_assistant=convert_tool_to_assistant,
            verbose=True,
        )
        _free_gpu()  # release the training model's GPU memory before eval/red-team reload

    # Precedence: CLI flag > config output: section > <results-dir>-derived default.
    activations_cache_dir = (
        args.activations_cache_dir
        or config.output.activations_cache_dir
        or (args.results_dir / "eval_activations")
    )
    eval_results: dict = {}

    # Optional: pull precomputed eval activations from Kaggle instead of extracting
    # them (see kaggle_activations.py). Only the FIRST eval actually downloads —
    # every later probe in the run hits the same path-keyed cache dir.
    kaggle_source = None
    if config.kaggle is not None:
        from agentic_redteam.kaggle_activations import KaggleActivationSource

        kaggle_source = KaggleActivationSource(
            owner=config.kaggle.owner,
            dataset_slug=config.kaggle.eval_dataset_slug,
            file_name=config.kaggle.eval_file_name,
        )

    def _maybe_eval(label: str, probe_path: Path) -> None:
        if not args.eval:
            return
        args.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== EVALUATING {label}: {probe_path} =====")
        df = evaluate_probe(
            probe_path,
            args.eval_dataset_dir,
            activations_cache_dir,
            max_samples=eval_max_samples,
            seed=args.seed,
            combine_consecutive_messages=combine_consecutive_messages,
            convert_tool_to_assistant=convert_tool_to_assistant,
            kaggle_source=kaggle_source,
        )
        eval_results[label] = df
        print(df.to_string(index=False))
        _free_gpu()  # release the eval model's GPU memory before the next phase reloads

    _maybe_eval(f"iter{start_iter}", current_probe_path)

    print(f"\nStarting probe (iteration {start_iter}): {current_probe_path}")
    print(f"Error types: {error_types}")
    print(f"Rounds per error type: {config.attacker.rounds}")
    if start_iter >= args.iterations:
        print(
            f"\nNothing to do: --iterations={args.iterations} but already resumed at "
            f"iteration {start_iter}. Increase --iterations to continue, or --no-resume "
            "to restart from scratch."
        )

    # ---- Steps 2-4, repeated until we reach --iterations total cycles ----
    for i in range(start_iter, args.iterations):
        print(f"\n########## ITERATION {i} ##########")
        config.probe.path = current_probe_path
        base_round = i * config.attacker.rounds

        iteration_new_total = 0
        for et in error_types:
            jsonl_path = config.jsonl_path_for(et)
            marker_path = _redteam_phase_marker_path(args.probe_out_dir, i, et)
            # Phase-aware resume: a marker means this (iteration, error_type)
            # red-team already finished on a prior run whose retrain was
            # interrupted. Its successes are already in the JSONL, so skip the
            # (expensive) rotation and let the retrain below pick them up. Only
            # the resumed iteration can be in this state, and only when resuming
            # (--no-resume must ignore stale markers and red-team afresh).
            # Gated on args.resume rather than `resume is not None`: a warm-started
            # run that crashed inside iteration 0 has no probe_iterN.pkl yet, so
            # `resume` is None even though its markers are valid — its input probe
            # is the same config.probe.path either way.
            if args.resume and i == start_iter and marker_path.exists():
                print(
                    f"\n--- Error type: {et} → {jsonl_path} ---\n"
                    f"  Skipping red-team: phase already complete "
                    f"({marker_path.name}); reusing saved successes."
                )
                continue
            print(f"\n--- Error type: {et} → {jsonl_path} ---")
            # Round-level resume: within this rotation, skip the individual rounds a
            # previous run already finished (sidecar <jsonl>.rounds_done.jsonl) and
            # restore its rolling memo. Only the iteration we resumed into can have
            # partial progress — later ones never started.
            summaries = run_redteam_sync(
                config, base_round_num=base_round, error_type=et, jsonl_path=jsonl_path,
                iteration=i, base_training_data_path=args.base_training_data,
                resume=args.resume and i == start_iter,
            )
            et_new = sum(s.new_successes for s in summaries)
            iteration_new_total += et_new
            print(f"  {et}: {et_new} new successes across {len(summaries)} model-rounds")
            # run_redteam already called probe.release() on its way out, which frees the
            # GPU — but not the host arenas glibc is still holding from the offloaded
            # shards. The NEXT error type's rotation reloads the same model immediately
            # and re-infers its device map from free GPU + available RAM, so without
            # this the second rotation can land in disk offload while the first sat
            # comfortably in CPU offload. See _free_gpu.
            _free_gpu()
            # Record phase completion so a future resume can skip this rotation.
            marker_path.write_text(
                json.dumps(
                    {
                        "iteration": i,
                        "error_type": et,
                        "jsonl_path": str(jsonl_path),
                        "new_successes": et_new,
                        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                ),
                encoding="utf-8",
            )

        print(f"Iteration {i}: {iteration_new_total} new successes total")

        jsonl_paths = [config.jsonl_path_for(et) for et in error_types]
        new_probe_path = args.probe_out_dir / f"probe_iter{i + 1}.pkl"
        postprocessed_out_path = (
            args.probe_out_dir / f"redteam_postprocessed_iter{i + 1}.jsonl"
        )
        result = retrain_probe(
            jsonl_path=jsonl_paths,
            base_probe_path=current_probe_path,
            base_training_data_path=args.base_training_data,
            new_probe_path=new_probe_path,
            layer=args.layer,
            probe_spec=args.probe_arch,
            preprocessing=config.preprocessing,
            contrastive_cache_path=contrastive_cache_path,
            postprocessed_out_path=postprocessed_out_path,
            min_judge_confidence=config.judge.confidence_threshold,
            test_size=args.test_size,
            split_field=args.split_field,
            dev_data_path=dev_data_path,
            seed=args.seed,
            base_data_fraction=args.base_data_fraction,
            ensemble_size=ensemble_size,
            base_activation_cache_dir=base_activation_cache_dir,
            combine_consecutive_messages=combine_consecutive_messages,
            convert_tool_to_assistant=convert_tool_to_assistant,
            verbose=True,
        )
        print(
            f"Iteration {i}: trained on {result.n_training_samples_total} samples "
            f"({result.n_redteam_samples} from red-team) → {result.new_probe_path}"
            + (
                f" [{result.ensemble_size}-member ensemble]"
                if result.ensemble_size > 1
                else ""
            )
        )
        _free_gpu()  # release the retrain model's GPU memory before eval/next red-team
        current_probe_path = new_probe_path
        _maybe_eval(f"iter{i + 1}", current_probe_path)

    if args.eval and eval_results:
        import pandas as pd

        rows = []
        for label, df in eval_results.items():
            d = df.copy()
            d.insert(0, "round", label)
            rows.append(d)
        comparison = pd.concat(rows, ignore_index=True)
        print("\n===== COMPARISON ACROSS ROUNDS =====")
        print(comparison.to_string(index=False))
        out_csv = (
            args.comparison_csv
            or config.output.comparison_csv
            or (args.results_dir / "iter_run_comparison.csv")
        )
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(out_csv, index=False)
        print(f"\nSaved comparison table to {out_csv}")

    print(f"\nFinal probe: {current_probe_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("redteam", "iterative-retrain"):
        cmd = sys.argv.pop(1)
        if cmd == "redteam":
            sys.exit(run_redteam_main())
        else:
            sys.exit(iterative_retrain_main())
    else:
        print("Usage: python -m agentic_redteam.cli {redteam|iterative-retrain} ...")
        sys.exit(2)
