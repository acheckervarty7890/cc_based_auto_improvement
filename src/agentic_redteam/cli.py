"""Console entry point for `iterative-generate`: the generate → score → retrain → guide loop.

One iteration, for a current probe P with mean dev AUROC A:

1. **Guidance.** Take the directions written for this iteration (the judge's, from the
   previous iteration's results; at iteration 0 the generator proposes them itself).
2. **Generate.** ``n_batches`` concurrent generator calls, batch *k* under direction *k*,
   each returning ``batch_size`` self-labelled samples (half per class).
3. **Warm the activation cache** for every new sample in one model load.
4. **Score each batch independently.** Train a candidate probe on
   base ∪ accepted-so-far ∪ batch (a pure cache hit), read its per-split dev AUROC,
   and compare its mean with A. Δ > ``min_auroc_gain`` ⇒ the batch is ACCEPTED;
   |Δ| ≤ ``exhausted_gain`` ⇒ flagged EXHAUSTED for the judge.
5. **Union retrain.** Train the next probe on base ∪ all accepted batches (this and
   earlier iterations) → ``probe_iter{i+1}.pkl``; its dev AUROC is the next baseline.
   With nothing accepted the probe is carried over unchanged.
6. **Judge.** Show every batch (direction, sample excerpts, Δ per split, verdict) to the
   judge, which rewrites the rolling memo and writes the next iteration's directions.
7. Optional eval on the eval splits.

Everything is resumable: ``probe_iter{N}.pkl`` picks the iteration, ``batches.jsonl``
carries generated-but-unscored and scored batches, ``guidance.jsonl`` the directions.
"""

from __future__ import annotations

import argparse
import csv
import functools
import shutil
import sys
import time
from pathlib import Path

from agentic_redteam.circuit_breaker import OpenRouterOutageError
from agentic_redteam.config import LoopRunConfig, load_config
from agentic_redteam.ensemble import ENSEMBLE_SEEDS, MAX_ENSEMBLE_SIZE
from agentic_redteam.evaluation import (
    DEFAULT_EVAL_MAX_SAMPLES,
    DEFAULT_SEED,
    evaluate_probe,
    seed_everything,
)
from agentic_redteam.generator import (
    Generator,
    ProbeMeta,
    generate_batches_sync,
    propose_directions_sync,
)
from agentic_redteam.llm_judge import JudgeRefusalError, LLMJudge
from agentic_redteam.persistence import (
    BatchRecord,
    BatchStore,
    GeneratedSample,
    GuidanceRecord,
    GuidanceStore,
    RunLogger,
)
from agentic_redteam.retrain import (
    DEFAULT_FRESH_PROBE_ARCH,
    read_probe_metadata,
    retrain_probe,
    score_probe_on_dev,
    train_initial_probe,
    warm_sample_activation_cache,
)
from agentic_redteam.token_budget import TokenBudget

# Exit code for "OpenRouter is down / out of credits", distinct from 1 (ordinary
# failure) and 2 (usage error) so a wrapper script can tell the cases apart.
OUTAGE_EXIT_CODE = 3


def _exit_on_outage(fn):
    """Turn an :class:`OpenRouterOutageError` into a clean non-zero exit.

    The breaker firing means the run stopped on purpose, not that it crashed. Little
    is lost by stopping: the sidecars are append-only, so ``--resume`` picks up at the
    batch this left off on once credits/keys are restored.
    """

    @functools.wraps(fn)
    def wrapper(argv: list[str] | None = None) -> int:
        try:
            return fn(argv)
        except OpenRouterOutageError as e:
            print(
                f"\n!!! ABORTED — OpenRouter is not usable.\n    {e}\n\n"
                "    Fix the account/key, then re-run the same command with --resume to "
                "continue from the last completed batch.",
                file=sys.stderr,
            )
            return OUTAGE_EXIT_CODE

    return wrapper


def _free_gpu() -> None:
    """Release reserved GPU **and host** memory between heavy phases.

    Every tuberlens load uses device_map="auto" and, unpinned, re-infers the layer
    split from *free* GPU memory; torch's caching allocator keeps freed memory
    reserved, so a model left resident pushes the next load into CPU/disk offload.
    The host half matters too: accelerate sizes the CPU budget from *available* RAM,
    and glibc keeps freed arenas mapped — ``malloc_trim(0)`` hands them back (glibc
    only; every failure mode is swallowed).
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
    """``(N, path)`` for the highest-numbered ``probe_iter{N}.pkl``, or None.

    ``probe_iter{i}.pkl`` is the probe iteration ``i`` starts from (``probe_iter0`` =
    initial), so the latest on disk is where a resumed run picks up.
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


def _fmt_scores(scores: dict[str, float] | None) -> str:
    if not scores:
        return "(none)"
    mean = scores.get("mean")
    parts = ", ".join(f"{k} {v:.4f}" for k, v in sorted(scores.items()) if k != "mean")
    return (f"mean {mean:.4f}" if mean is not None else "") + (f" ({parts})" if parts else "")


def _write_history_csv(path: Path, store: BatchStore) -> None:
    """One row per (iteration, batch): the ΔAUROC ledger, newest record per key."""
    newest: dict[tuple[int, int], BatchRecord] = {}
    for r in store.records:
        newest[(r.iteration, r.batch_index)] = r
    split_names: list[str] = sorted(
        {k for r in newest.values() for k in r.auroc_after if k != "mean"}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["iteration", "batch", "status", "generator_model", "n_samples", "auroc_before_mean",
             "auroc_after_mean", "delta", "accepted", "exhausted"]
            + [f"delta_{s}" for s in split_names]
            + ["direction"]
        )
        for key in sorted(newest):
            r = newest[key]
            d = r.per_split_delta
            w.writerow(
                [r.iteration, r.batch_index, r.status, r.generator_model, r.n_samples,
                 f"{r.auroc_before.get('mean', float('nan')):.5f}",
                 f"{r.auroc_after.get('mean', float('nan')):.5f}",
                 f"{r.delta:+.5f}", int(r.accepted), int(r.exhausted)]
                + [f"{d[s]:+.5f}" if s in d else "" for s in split_names]
                + [r.direction.replace("\n", " ")]
            )


def _write_samples_jsonl(path: Path, samples: list[GeneratedSample]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            row = s.to_training_row()
            f.write(json.dumps({"id": f"generated-{i}", **row}, ensure_ascii=False) + "\n")


def _fill_directions(generator: Generator, directions: list[str], n: int, memo: str, iteration: int) -> list[str]:
    """Exactly ``n`` directions: truncate an over-long list, fill a short one via the generator."""
    directions = [d for d in directions if d.strip()][:n]
    if len(directions) < n:
        print(f"  guidance holds {len(directions)}/{n} directions; asking the generator for the rest")
        extra = propose_directions_sync(
            generator, n - len(directions), memo=memo, existing=directions, iteration=iteration
        )
        directions = directions + extra
    return directions[:n]


@_exit_on_outage
def iterative_generate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train an initial probe, then iterate: generate batches → score each "
        "by dev ΔAUROC → retrain on the accepted ones → judge guides the next round."
    )
    parser.add_argument("config", type=Path, help="Path to the markdown config file")
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Number of generate → score → retrain → guide cycles; overrides loop.iterations",
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Resume an interrupted run: the latest probe_iterN.pkl in --probe-out-dir picks "
        "the iteration, batches.jsonl the batches already generated/scored, guidance.jsonl "
        "the directions. --no-resume starts over (existing sidecars are still appended to).",
    )
    parser.add_argument(
        "--base-training-data", type=Path, required=True,
        help="JSONL/CSV used to train the initial probe and included in every retrain",
    )
    parser.add_argument(
        "--dev-data", type=Path, default=None,
        help="Held-out dev data (a JSONL, or a directory of *.jsonl splits): the fit's "
        "validation set AND the set every batch's ΔAUROC is read on. Overrides "
        "validation.dev_data. Required one way or the other; must be disjoint from "
        "--eval-dataset-dir.",
    )
    parser.add_argument(
        "--base-data-fraction", type=float, default=1.0,
        help="Fraction (0, 1] of the base training data to ingest (deterministic subsample).",
    )
    parser.add_argument(
        "--probe-out-dir", type=Path, default=Path("probes"),
        help="Where to write probes (probe_iter0.pkl = initial, probe_iterN.pkl per cycle, "
        "candidates/ for the per-batch probes)",
    )
    parser.add_argument(
        "--base-activation-cache-dir", type=Path, default=None,
        help="Cache dir for training-side activations (base blob, per-sample blobs, dev "
        "blob). Overrides output.base_activation_cache_dir; default "
        "<probe-out-dir>/base_activation_cache.",
    )
    parser.add_argument("--layer", type=int, default=None, help="Layer to probe; overrides probe.layer")
    parser.add_argument(
        "--probe-arch", nargs="?", const=DEFAULT_FRESH_PROBE_ARCH, default=None,
        help="Probe architecture (ProbeType name); overrides probe.architecture. Bare flag "
        f"uses {DEFAULT_FRESH_PROBE_ARCH!r}; omitted retrains inherit the current probe's.",
    )
    parser.add_argument(
        "--ensemble-size", type=int, default=None,
        help=f"Fit this many probes (1..{MAX_ENSEMBLE_SIZE}) at every training step and "
        "average their scores. Overrides probe.ensemble_size. Omit both to inherit.",
    )
    parser.add_argument("--eval", action="store_true", help="Evaluate every probe on the eval splits")
    parser.add_argument(
        "--eval-dataset-dir", type=Path, default=Path("eval_sets/highstakes"),
        help="Directory of local eval split JSONLs (used with --eval)",
    )
    parser.add_argument(
        "--eval-max-samples", type=int, default=None,
        help="Balanced subsample size per eval split; 0 = full split. Overrides "
        f"eval.eval_max_samples (default {DEFAULT_EVAL_MAX_SAMPLES})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
        help="Default location of the eval comparison CSV and eval activations (used with --eval)",
    )
    parser.add_argument(
        "--activations-cache-dir", type=Path, default=None,
        help="Cache dir for eval activations; overrides output.activations_cache_dir",
    )
    parser.add_argument(
        "--comparison-csv", type=Path, default=None,
        help="Path for the cross-iteration eval comparison CSV; overrides output.comparison_csv",
    )
    parser.add_argument(
        "--combine-consecutive-messages", action=argparse.BooleanOptionalAction, default=None,
        help="Merge adjacent same-role messages everywhere; overrides eval.combine_consecutive_messages",
    )
    parser.add_argument(
        "--convert-tool-to-assistant", action=argparse.BooleanOptionalAction, default=None,
        help="Rewrite tool messages as assistant everywhere; overrides eval.convert_tool_to_assistant",
    )
    args = parser.parse_args(argv)

    if not 0.0 < args.base_data_fraction <= 1.0:
        parser.error(f"--base-data-fraction must be in (0, 1]; got {args.base_data_fraction}")
    if args.ensemble_size is not None and not 1 <= args.ensemble_size <= MAX_ENSEMBLE_SIZE:
        parser.error(f"--ensemble-size must be between 1 and {MAX_ENSEMBLE_SIZE}; got {args.ensemble_size}")

    config: LoopRunConfig = load_config(args.config)
    args.probe_out_dir.mkdir(parents=True, exist_ok=True)
    config.output.run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    iterations = args.iterations if args.iterations is not None else config.loop.iterations
    layer = args.layer if args.layer is not None else config.probe.layer
    arch = args.probe_arch if args.probe_arch is not None else config.probe.architecture
    ensemble_size = args.ensemble_size if args.ensemble_size is not None else config.probe.ensemble_size
    if args.eval_max_samples is not None:
        raw_eval_max_samples = args.eval_max_samples
    elif config.eval.eval_max_samples is not None:
        raw_eval_max_samples = config.eval.eval_max_samples
    else:
        raw_eval_max_samples = DEFAULT_EVAL_MAX_SAMPLES
    eval_max_samples = raw_eval_max_samples if raw_eval_max_samples > 0 else None
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
    dev_data_path = args.dev_data or config.validation.dev_data
    if dev_data_path is None:
        parser.error(
            "A dev set is required (every batch is scored by its dev ΔAUROC): pass "
            "--dev-data or set validation.dev_data in the config."
        )
    dev_data_path = Path(dev_data_path)
    base_activation_cache_dir = (
        args.base_activation_cache_dir
        or config.output.base_activation_cache_dir
        or (args.probe_out_dir / "base_activation_cache")
    )
    n_batches = config.generator.n_batches
    min_gain = config.loop.min_auroc_gain
    exhausted_gain = config.loop.exhausted_gain

    print(
        f"Loop: {iterations} iteration(s) × {n_batches} batches × {config.generator.batch_size} "
        f"samples; accept if Δ mean dev AUROC > {min_gain:.4f}; exhausted if |Δ| <= {exhausted_gain:.4f}"
    )
    print(f"Generator models: {config.generator.model_names}; judge: {config.judge.model}")
    print(f"Dev set: {dev_data_path}; run dir: {config.output.run_dir}")
    if ensemble_size is not None and ensemble_size > 1:
        print(
            f"Deep ensemble: every fit trains {ensemble_size} probes under seeds "
            f"{list(ENSEMBLE_SEEDS[:ensemble_size])} and averages their scores."
        )
    if combine_consecutive_messages or convert_tool_to_assistant:
        print(
            f"Message transforms: combine_consecutive_messages={combine_consecutive_messages}, "
            f"convert_tool_to_assistant={convert_tool_to_assistant}"
        )

    batch_store = BatchStore(config.output.batches_path)
    guidance_store = GuidanceStore(config.output.guidance_path)
    if not args.resume:
        # A fresh run: earlier rows stay in the files (append-only) but are invisible
        # to this process — no batch is reused, no guidance restored, no sample deduped
        # against a previous run.
        batch_store.forget_loaded()
        guidance_store.forget_loaded()
    runlog = RunLogger(config.output.runlog_path)
    history_csv = config.output.run_dir / "auroc_history.csv"

    # Optional: pull the dev set's activations from Kaggle and assemble the blob the
    # fit and the scorer look for, before anything trains.
    if config.kaggle is not None and config.kaggle.dev_dataset_slug:
        from agentic_redteam.kaggle_activations import (
            KaggleActivationSource,
            prefetch_dev_activations,
        )
        from agentic_redteam.retrain import _dev_activation_cache_path

        _dev_files = sorted(dev_data_path.glob("*.jsonl")) if dev_data_path.is_dir() else [dev_data_path]
        if not _dev_files:
            parser.error(f"validation.dev_data {dev_data_path} holds no *.jsonl splits")
        if config.probe.model is None or layer is None:
            parser.error("kaggle dev prefetch needs probe.model and probe.layer in the config")
        prefetch_dev_activations(
            _dev_activation_cache_path(
                base_activation_cache_dir, _dev_files, config.probe.model, layer,
                combine_consecutive_messages, convert_tool_to_assistant,
            ),
            _dev_files,
            KaggleActivationSource(
                config.kaggle.owner, config.kaggle.dev_dataset_slug, config.kaggle.dev_file_name
            ),
            model_name=config.probe.model,
            layer=layer,
            verbose=True,
        )

    # ---- Step 1: obtain the initial probe and its dev AUROC ----
    baseline: dict[str, float] | None = None
    resume = _latest_probe_iteration(args.probe_out_dir) if args.resume else None
    if resume is not None:
        start_iter, current_probe_path = resume
        print(f"Resuming from probe iteration {start_iter}: {current_probe_path} (--no-resume to restart)")
    elif config.probe.path is not None and Path(config.probe.path).exists():
        start_iter = 0
        current_probe_path = Path(config.probe.path)
        print(f"Warm-starting from existing probe: {current_probe_path}")
    else:
        start_iter = 0
        missing = [
            name for name, value in (
                ("probe.model", config.probe.model),
                ("probe.layer", layer),
                ("probe.pos_class_label", config.probe.pos_class_label),
                ("probe.neg_class_label", config.probe.neg_class_label),
            ) if value is None
        ]
        if missing:
            parser.error(
                "Training the initial probe from scratch requires " + ", ".join(missing)
                + " — set them in the config probe: section, or point probe.path at an existing probe."
            )
        current_probe_path = args.probe_out_dir / "probe_iter0.pkl"
        print(f"Training initial probe → {current_probe_path}")
        result = train_initial_probe(
            base_training_data_path=args.base_training_data,
            model_name=config.probe.model,
            layer=layer,
            new_probe_path=current_probe_path,
            pos_class_label=config.probe.pos_class_label,
            neg_class_label=config.probe.neg_class_label,
            probe_description=config.probe.description,
            probe_spec=arch,
            dev_data_path=dev_data_path,
            seed=args.seed,
            base_data_fraction=args.base_data_fraction,
            ensemble_size=ensemble_size or 1,
            base_activation_cache_dir=base_activation_cache_dir,
            combine_consecutive_messages=combine_consecutive_messages,
            convert_tool_to_assistant=convert_tool_to_assistant,
            verbose=True,
        )
        baseline = result.dev_auroc
        _free_gpu()

    probe_info = read_probe_metadata(current_probe_path)
    probe_meta = ProbeMeta(
        pos_class_label=probe_info["pos_class_label"],
        neg_class_label=probe_info["neg_class_label"],
        description=probe_info["description"],
        model_name=probe_info["model_name"],
    )
    print(
        f"Probe: {probe_meta.model_name} layer {probe_info['layer']}, classes "
        f"'{probe_meta.pos_class_label}' / '{probe_meta.neg_class_label}'"
        + (f", {probe_info['ensemble_size']}-member ensemble" if probe_info["ensemble_size"] > 1 else "")
    )

    generator = Generator(
        config=config.generator,
        probe=probe_meta,
        token_budget=TokenBudget(
            model_name=probe_meta.model_name,
            max_tokens=config.generator.max_sample_tokens,
            combine_consecutive_messages=combine_consecutive_messages,
            convert_tool_to_assistant=convert_tool_to_assistant,
        ),
        runlog=runlog,
    )
    generator.warmup()
    judge = LLMJudge(
        model=config.judge.model,
        provider=config.judge.provider,
        system_prompt=config.judge.system_prompt,
        pos_class_label=probe_meta.pos_class_label,
        neg_class_label=probe_meta.neg_class_label,
        probe_description=probe_meta.description,
        eval_data_description=config.eval.data_description,
        max_tokens=config.judge.max_tokens,
        memo_word_budget=config.judge.memo_word_budget,
        max_samples_per_batch=config.judge.max_samples_per_batch,
    )
    judge.warmup()

    activations_cache_dir = (
        args.activations_cache_dir
        or config.output.activations_cache_dir
        or (args.results_dir / "eval_activations")
    )
    eval_results: dict = {}
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
        _free_gpu()

    def _score_baseline(probe_path: Path) -> dict[str, float]:
        print(f"\nScoring baseline dev AUROC of {probe_path}")
        scores = score_probe_on_dev(
            probe_path, dev_data_path, base_activation_cache_dir,
            combine_consecutive_messages=combine_consecutive_messages,
            convert_tool_to_assistant=convert_tool_to_assistant, verbose=True,
        )
        _free_gpu()
        return scores

    def _fit(samples: list[GeneratedSample], out_path: Path, what: str):
        print(f"\n--- Fitting {what}: base ∪ {len(samples)} samples → {out_path.name} ---")
        result = retrain_probe(
            samples=samples,
            base_probe_path=current_probe_path,
            base_training_data_path=args.base_training_data,
            new_probe_path=out_path,
            layer=args.layer,
            probe_spec=args.probe_arch,
            dev_data_path=dev_data_path,
            seed=args.seed,
            base_data_fraction=args.base_data_fraction,
            ensemble_size=ensemble_size,
            base_activation_cache_dir=base_activation_cache_dir,
            combine_consecutive_messages=combine_consecutive_messages,
            convert_tool_to_assistant=convert_tool_to_assistant,
            verbose=True,
        )
        _free_gpu()
        return result

    _maybe_eval(f"iter{start_iter}", current_probe_path)
    if baseline is None:
        baseline = _score_baseline(current_probe_path)
    print(f"Baseline dev AUROC (iteration {start_iter}): {_fmt_scores(baseline)}")

    if start_iter >= iterations:
        print(
            f"\nNothing to do: iterations={iterations} but already resumed at iteration "
            f"{start_iter}. Raise --iterations / loop.iterations, or --no-resume."
        )

    # ---- Steps 2-7, repeated ----
    for i in range(start_iter, iterations):
        print(f"\n########## ITERATION {i} ##########")
        runlog.log("iteration_start", iteration=i, probe=str(current_probe_path), baseline=baseline)
        prior_memo = guidance_store.latest_memo_before(i)

        # (1) Directions for this iteration.
        guidance = guidance_store.for_iteration(i)
        if guidance is None:
            prev_batches = batch_store.for_iteration(i - 1) if i > 0 else []
            if prev_batches and any(b.status == "scored" for b in prev_batches):
                # A resumed run that died between scoring iteration i-1 and writing its
                # guidance: ask the judge now, on the same evidence.
                print(f"  Writing missing guidance for iteration {i} from iteration {i - 1}'s batches")
                g = judge.guide(
                    prev_batches, iteration=i - 1, n_directions=n_batches,
                    prior_memo=guidance_store.latest_memo_before(i - 1),
                    auroc_before=prev_batches[0].auroc_before, auroc_after=baseline,
                    min_gain=min_gain, exhausted_gain=exhausted_gain,
                )
                memo, directions, source = g.memo, g.directions, "judge"
            else:
                print(f"  No guidance yet: asking the generator to propose {n_batches} directions")
                memo, source = prior_memo, "generator_proposal"
                directions = propose_directions_sync(generator, n_batches, memo=memo, iteration=i)
            directions = _fill_directions(generator, directions, n_batches, memo, i)
            guidance = GuidanceRecord(
                run_id=config.output.run_id, iteration=i, memo=memo, directions=directions,
                source=source, baseline_auroc=dict(baseline),
            )
            guidance_store.append(guidance)
        directions = _fill_directions(generator, list(guidance.directions), n_batches, guidance.memo, i)
        memo = guidance.memo
        print("  Directions:")
        for k, d in enumerate(directions):
            print(f"    [{k + 1}] {d[:160]}{'…' if len(d) > 160 else ''}")

        # (2) Generate the batches that have no record yet.
        pending: dict[int, BatchRecord] = {}
        to_generate: list[int] = []
        for k in range(n_batches):
            rec = batch_store.get(i, k)
            if rec is None:
                to_generate.append(k)
            elif rec.status == "generated":
                pending[k] = rec
            else:
                print(f"  batch {k + 1}: already {rec.status} (Δ {rec.delta:+.4f}); skipping")
        if to_generate:
            print(f"\n  Generating {len(to_generate)} batch(es) of {config.generator.batch_size} ...")
            t0 = time.monotonic()
            generated = generate_batches_sync(
                generator, iteration=i, directions=directions, memo=memo,
                store=batch_store, batch_indices=to_generate,
            )
            print(f"  Generation took {(time.monotonic() - t0) / 60:.1f} min")
            for k, gen in sorted(generated.items()):
                model = generator.model_for_batch(k)
                status = "generated" if gen.samples else ("generation_failed" if gen.error else "empty")
                rec = BatchRecord(
                    run_id=config.output.run_id, iteration=i, batch_index=k,
                    direction=directions[k], generator_model=model.name, provider=model.provider,
                    samples=list(gen.samples), n_requested=config.generator.batch_size,
                    status=status, auroc_before=dict(baseline),
                    n_dropped_too_long=gen.n_dropped_too_long,
                    n_dropped_duplicate=gen.n_dropped_duplicate,
                    n_dropped_bad_label=gen.n_dropped_bad_label,
                    n_generation_calls=gen.n_calls, error=gen.error,
                )
                counts = rec.n_per_label
                print(
                    f"  batch {k + 1} [{model.name}]: {rec.n_samples}/{rec.n_requested} samples "
                    f"({counts.get(probe_meta.pos_class_label, 0)} pos / "
                    f"{counts.get(probe_meta.neg_class_label, 0)} neg; dropped "
                    f"{gen.n_dropped_too_long} long, {gen.n_dropped_duplicate} dup, "
                    f"{gen.n_dropped_bad_label} malformed; {gen.n_calls} call(s))"
                    + (f" ERROR: {gen.error[:120]}" if gen.error else "")
                )
                batch_store.append(rec)
                if status == "generated":
                    pending[k] = rec

        # (3) Warm the per-sample activation cache for everything about to be fit.
        accepted_before = batch_store.accepted_samples(before_iteration=i)
        if pending:
            all_new = [s for rec in pending.values() for s in rec.samples]
            print(f"\n  Warming activation cache for {len(all_new)} generated samples ...")
            warm_sample_activation_cache(
                all_new + accepted_before, current_probe_path, base_activation_cache_dir,
                combine_consecutive_messages=combine_consecutive_messages,
                convert_tool_to_assistant=convert_tool_to_assistant, verbose=True,
            )
            _free_gpu()

        # (4) Score each pending batch on its own.
        candidates_dir = args.probe_out_dir / "candidates"
        for k in sorted(pending):
            rec = pending[k]
            cand_path = candidates_dir / f"probe_iter{i}_batch{k}.pkl"
            result = _fit(accepted_before + rec.samples, cand_path, f"candidate for batch {k + 1}")
            after = result.dev_auroc or {}
            delta = after.get("mean", float("nan")) - baseline.get("mean", float("nan"))
            accepted = delta > min_gain
            exhausted = abs(delta) <= exhausted_gain
            scored = BatchRecord(
                **{**rec.__dict__, "status": "scored", "auroc_after": after, "delta": float(delta),
                   "accepted": bool(accepted), "exhausted": bool(exhausted),
                   "candidate_probe_path": str(cand_path), "created_at": time.time()}
            )
            batch_store.append(scored)
            print(
                f"  batch {k + 1}: mean dev AUROC {baseline.get('mean', float('nan')):.4f} → "
                f"{after.get('mean', float('nan')):.4f} (Δ {delta:+.4f}) → "
                + ("ACCEPTED" if accepted else "exhausted" if exhausted else "rejected")
            )
            runlog.log("batch_scored", iteration=i, batch=k, delta=float(delta),
                       accepted=bool(accepted), exhausted=bool(exhausted))
        _write_history_csv(history_csv, batch_store)

        # (5) Union retrain on everything accepted so far.
        batches_i = batch_store.for_iteration(i)
        newest_i: dict[int, BatchRecord] = {}
        for b in batches_i:
            newest_i[b.batch_index] = b
        batches_i = [newest_i[k] for k in sorted(newest_i)]
        accepted_now = [b for b in batches_i if b.accepted]
        accepted_all = batch_store.accepted_samples(before_iteration=i + 1)
        new_probe_path = args.probe_out_dir / f"probe_iter{i + 1}.pkl"
        _write_samples_jsonl(config.output.run_dir / f"accepted_iter{i + 1}.jsonl", accepted_all)
        if accepted_now:
            print(
                f"\n  {len(accepted_now)}/{len(batches_i)} batches accepted; retraining on base ∪ "
                f"{len(accepted_all)} accepted samples → {new_probe_path.name}"
            )
            result = _fit(accepted_all, new_probe_path, f"probe_iter{i + 1}")
            new_baseline = result.dev_auroc or baseline
            retrained = True
        else:
            print(f"\n  No batch accepted; carrying {current_probe_path.name} over as {new_probe_path.name}")
            if current_probe_path.resolve() != new_probe_path.resolve():
                shutil.copyfile(current_probe_path, new_probe_path)
            new_baseline = baseline
            retrained = False
        print(
            f"  Dev AUROC after iteration {i}: {_fmt_scores(new_baseline)} "
            f"(was {_fmt_scores(baseline)})"
        )
        runlog.log("iteration_retrain", iteration=i, retrained=retrained,
                   n_accepted_batches=len(accepted_now), n_accepted_samples=len(accepted_all),
                   dev_auroc=new_baseline)

        # (6) Judge → guidance for iteration i+1.
        if guidance_store.for_iteration(i + 1) is None:
            print(f"\n  Judge writing guidance for iteration {i + 1} ...")
            try:
                g = judge.guide(
                    batches_i, iteration=i, n_directions=n_batches, prior_memo=memo,
                    auroc_before=baseline, auroc_after=new_baseline if retrained else None,
                    min_gain=min_gain, exhausted_gain=exhausted_gain,
                )
            except JudgeRefusalError:
                runlog.log("guidance_refused", iteration=i)
                raise
            except OpenRouterOutageError:
                raise
            except Exception as e:  # noqa: BLE001 — a judge hiccup must not lose the retrain
                print(f"  Judge failed ({type(e).__name__}: {e}); the next iteration will re-ask.")
                runlog.log("guidance_error", iteration=i, error=f"{type(e).__name__}: {e}")
                g = None
            if g is not None:
                next_directions = _fill_directions(generator, g.directions, n_batches, g.memo, i + 1)
                guidance_store.append(
                    GuidanceRecord(
                        run_id=config.output.run_id, iteration=i + 1, memo=g.memo,
                        directions=next_directions, source="judge", baseline_auroc=dict(new_baseline),
                    )
                )
                print("  Memo:\n    " + g.memo.replace("\n", "\n    ")[:2000])

        current_probe_path = new_probe_path
        baseline = new_baseline
        _maybe_eval(f"iter{i + 1}", current_probe_path)
        runlog.log("iteration_end", iteration=i)

    print(f"\nΔAUROC ledger: {history_csv}")
    if args.eval and eval_results:
        import pandas as pd

        rows = []
        for label, df in eval_results.items():
            d = df.copy()
            d.insert(0, "round", label)
            rows.append(d)
        comparison = pd.concat(rows, ignore_index=True)
        print("\n===== COMPARISON ACROSS ITERATIONS =====")
        print(comparison.to_string(index=False))
        out_csv = (
            args.comparison_csv
            or config.output.comparison_csv
            or (args.results_dir / "iter_run_comparison.csv")
        )
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(out_csv, index=False)
        print(f"Saved comparison to {out_csv}")
    return 0
