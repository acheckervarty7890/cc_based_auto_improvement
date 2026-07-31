"""Round-level resume: progress sidecar + memo restore + the run_redteam skip loop.

Run:  .venv_claude/bin/python scratchpad/test_round_resume.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from agentic_redteam.persistence import (
    RoundProgress,
    RoundProgressStore,
    RoundSummary,
    SummaryStore,
)


def test_progress_store_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "log.rounds_done.jsonl"
        store = RoundProgressStore(path=path)
        assert not store.is_done(0, "false_positive", 3)
        for r in (0, 1, 2):
            store.mark_done(
                RoundProgress(round=r, iteration=0, error_type="false_positive",
                              n_attempts=7, n_successes=2)
            )
        # A different iteration / error type must not collide.
        store.mark_done(
            RoundProgress(round=0, iteration=1, error_type="false_negative",
                          n_attempts=1, n_successes=0)
        )

        reloaded = RoundProgressStore(path=path)
        assert reloaded.done_rounds(0, "false_positive") == [0, 1, 2]
        assert reloaded.done_rounds(1, "false_negative") == [0]
        assert reloaded.done_rounds(1, "false_positive") == []
        assert reloaded.is_done(0, "false_positive", 2)
        assert not reloaded.is_done(0, "false_positive", 3)
        assert not reloaded.is_done(0, "false_negative", 0)
    print("ok  progress store roundtrip")


def test_progress_store_tolerates_torn_line() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "log.rounds_done.jsonl"
        store = RoundProgressStore(path=path)
        store.mark_done(RoundProgress(round=0, iteration=0, error_type="fp",
                                      n_attempts=1, n_successes=1))
        with path.open("a", encoding="utf-8") as f:
            f.write('{"round": 1, "iteration": 0, "error_ty')  # killed mid-write
        assert RoundProgressStore(path=path).done_rounds(0, "fp") == [0]
    print("ok  torn sidecar line skipped")


def test_summary_store_resume() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "log.summaries.jsonl"
        store = SummaryStore(path=path)
        for r, text in ((0, "memo after round 0"), (1, "memo after round 1")):
            store.update(RoundSummary(round=r, iteration=0, error_type="fp",
                                      text=text, n_attempts=5, n_successes=1))
        store.update(RoundSummary(round=10, iteration=1, error_type="fp",
                                  text="iter-1 memo", n_attempts=5, n_successes=1))

        # Default: sidecar is diagnostics-only, memo starts empty.
        assert SummaryStore(path=path).current == ""
        # Resume: newest memo for THIS iteration, not the newest row overall.
        resumed = SummaryStore(path=path, iteration=0, error_type="fp", resume=True)
        assert resumed.current == "memo after round 1", resumed.current
        assert "memo after round 1" in resumed.render()
        # A later iteration doesn't inherit an earlier one's memo (per-iteration reset).
        assert SummaryStore(path=path, iteration=2, error_type="fp", resume=True).current == ""
        # Wrong error type is ignored too.
        assert SummaryStore(path=path, iteration=0, error_type="fn", resume=True).current == ""
    print("ok  summary store resume")


def test_run_redteam_skips_finished_rounds() -> None:
    """Drive the real run_redteam loop with everything heavy stubbed out."""
    import agentic_redteam.attacker as A

    ran: list[int] = []
    summarized: list[int] = []

    class _FakeProbe:
        pos_class_label, neg_class_label, true_class_label = "pos", "neg", "neg"

        @classmethod
        def load(cls, *a, **k):
            return cls()

        def warmup(self): ...
        def release(self): ...

    class _FakeJudge:
        def __init__(self, *a, **k): ...
        def warmup(self): ...

    async def _fake_run_one_model(*, round_num, **kw):
        ran.append(round_num)
        return A.ModelRunSummary(model="m", provider="openrouter", new_successes=1,
                                 total_messages=2, stop_reason="done")

    async def _fake_summarize_round(*, summary_store, round_num, iteration, error_type, **kw):
        summarized.append(round_num)
        summary_store.update(RoundSummary(round=round_num, iteration=iteration,
                                          error_type=error_type, text=f"memo r{round_num}",
                                          n_attempts=1, n_successes=1))

    async def _fake_write_iteration_memo(**kw): ...

    orig = (A.ProbeJudge, A.LLMJudge, A.run_one_model,
            A._summarize_round, A._write_iteration_memo)
    A.ProbeJudge, A.LLMJudge = _FakeProbe, _FakeJudge
    A.run_one_model = _fake_run_one_model
    A._summarize_round = _fake_summarize_round
    A._write_iteration_memo = _fake_write_iteration_memo

    try:
        with tempfile.TemporaryDirectory() as d:
            cfg = _make_config(Path(d) / "log.jsonl", rounds=5)

            # ---- Run 1: dies after round 2 (simulated by a raising round 3). ----
            calls = {"n": 0}
            real_fake = A.run_one_model

            async def _die_on_round_3(*, round_num, **kw):
                if round_num == 3:
                    raise RuntimeError("simulated crash")
                return await real_fake(round_num=round_num, **kw)

            A.run_one_model = _die_on_round_3
            try:
                asyncio.run(A.run_redteam(cfg, error_type="false_positive",
                                          jsonl_path=cfg.output.jsonl_path))
            except RuntimeError:
                pass
            A.run_one_model = real_fake
            assert ran == [0, 1, 2], ran  # round 3 raised before recording
            assert summarized == [0, 1, 2], summarized

            done = RoundProgressStore(path=cfg.output.jsonl_path.with_suffix(".rounds_done.jsonl"))
            assert done.done_rounds(0, "false_positive") == [0, 1, 2], done.done_rounds(0, "false_positive")

            # ---- Run 2 with resume: only rounds 3,4 re-run. ----
            ran.clear()
            summarized.clear()
            asyncio.run(A.run_redteam(cfg, error_type="false_positive",
                                      jsonl_path=cfg.output.jsonl_path, resume=True))
            assert ran == [3, 4], ran
            assert summarized == [3], summarized  # round 4 is last → not summarized

            # ---- Run 3 without resume: everything re-runs. ----
            ran.clear()
            asyncio.run(A.run_redteam(cfg, error_type="false_positive",
                                      jsonl_path=cfg.output.jsonl_path, resume=False))
            assert ran == [0, 1, 2, 3, 4], ran
    finally:
        (A.ProbeJudge, A.LLMJudge, A.run_one_model,
         A._summarize_round, A._write_iteration_memo) = orig
    print("ok  run_redteam skips finished rounds")


def test_resumed_round_sees_restored_memo() -> None:
    """The first resumed round must get the memo from before the crash, not "" ."""
    import agentic_redteam.attacker as A

    seen_memos: list[str] = []

    class _FakeProbe:
        pos_class_label, neg_class_label, true_class_label = "pos", "neg", "neg"

        @classmethod
        def load(cls, *a, **k):
            return cls()

        def warmup(self): ...
        def release(self): ...

    class _FakeJudge:
        def __init__(self, *a, **k): ...
        def warmup(self): ...

    async def _record_memo(*, round_num, summary_store, **kw):
        seen_memos.append(summary_store.render() if summary_store else "")
        return A.ModelRunSummary(model="m", provider="openrouter", new_successes=0,
                                 total_messages=1, stop_reason="done")

    async def _fake_summarize_round(*, summary_store, round_num, iteration, error_type, **kw):
        summary_store.update(RoundSummary(round=round_num, iteration=iteration,
                                          error_type=error_type, text=f"memo r{round_num}",
                                          n_attempts=1, n_successes=0))

    async def _noop(**kw): ...

    orig = (A.ProbeJudge, A.LLMJudge, A.run_one_model,
            A._summarize_round, A._write_iteration_memo)
    A.ProbeJudge, A.LLMJudge = _FakeProbe, _FakeJudge
    A.run_one_model = _record_memo
    A._summarize_round = _fake_summarize_round
    A._write_iteration_memo = _noop
    try:
        with tempfile.TemporaryDirectory() as d:
            cfg = _make_config(Path(d) / "log.jsonl", rounds=3)

            # Run 1 dies entering round 2, after rounds 0-1 were summarized.
            async def _die_on_round_2(*, round_num, **kw):
                if round_num == 2:
                    raise RuntimeError("simulated crash")
                return await _record_memo(round_num=round_num, **kw)

            A.run_one_model = _die_on_round_2
            try:
                asyncio.run(A.run_redteam(cfg, error_type="false_positive",
                                          jsonl_path=cfg.output.jsonl_path))
            except RuntimeError:
                pass
            A.run_one_model = _record_memo
            assert seen_memos[0] == ""  # round 0 has no memo yet
            assert "memo r0" in seen_memos[1], seen_memos[1]

            seen_memos.clear()
            # Resume: round 2 is the only one left, and must open with the memo
            # written after round 1 — not an empty one.
            asyncio.run(A.run_redteam(cfg, error_type="false_positive",
                                      jsonl_path=cfg.output.jsonl_path, resume=True))
            assert len(seen_memos) == 1, seen_memos
            assert "memo r1" in seen_memos[0], seen_memos[0]
    finally:
        (A.ProbeJudge, A.LLMJudge, A.run_one_model,
         A._summarize_round, A._write_iteration_memo) = orig
    print("ok  resumed round sees restored memo")


def test_legacy_concurrent_path_checkpoints_rounds() -> None:
    """round_summaries: false launches all rounds at once but still checkpoints them."""
    import agentic_redteam.attacker as A

    ran: list[int] = []

    class _FakeProbe:
        pos_class_label, neg_class_label, true_class_label = "pos", "neg", "neg"

        @classmethod
        def load(cls, *a, **k):
            return cls()

        def warmup(self): ...
        def release(self): ...

    class _FakeJudge:
        def __init__(self, *a, **k): ...
        def warmup(self): ...

    async def _fake_run_one_model(*, round_num, **kw):
        await asyncio.sleep(0)  # let siblings interleave, as they really would
        ran.append(round_num)
        return A.ModelRunSummary(model="m", provider="openrouter", new_successes=0,
                                 total_messages=1, stop_reason="done")

    async def _noop(**kw): ...

    orig = (A.ProbeJudge, A.LLMJudge, A.run_one_model, A._write_iteration_memo)
    A.ProbeJudge, A.LLMJudge = _FakeProbe, _FakeJudge
    A.run_one_model = _fake_run_one_model
    A._write_iteration_memo = _noop
    try:
        with tempfile.TemporaryDirectory() as d:
            cfg = _make_config(Path(d) / "log.jsonl", rounds=4)
            cfg.attacker.round_summaries = False
            asyncio.run(A.run_redteam(cfg, error_type="false_positive",
                                      jsonl_path=cfg.output.jsonl_path))
            assert sorted(ran) == [0, 1, 2, 3], ran
            done = RoundProgressStore(
                path=cfg.output.jsonl_path.with_suffix(".rounds_done.jsonl")
            )
            assert done.done_rounds(0, "false_positive") == [0, 1, 2, 3]

            ran.clear()
            asyncio.run(A.run_redteam(cfg, error_type="false_positive",
                                      jsonl_path=cfg.output.jsonl_path, resume=True))
            assert ran == [], ran  # nothing left to do
    finally:
        (A.ProbeJudge, A.LLMJudge, A.run_one_model, A._write_iteration_memo) = orig
    print("ok  legacy concurrent path checkpoints rounds")


def _make_config(jsonl_path: Path, *, rounds: int):
    from agentic_redteam.config import (
        AttackerConfig, AttackerModel, EvalConfig, JudgeConfig,
        OutputConfig, ProbeConfig, RedteamConfig,
    )

    return RedteamConfig(
        attacker=AttackerConfig(
            models=[AttackerModel(name="fake", provider="openrouter")],
            rounds=rounds, concurrency=4, sessions_per_model=1,
            round_summaries=True, cross_iteration_memos=False,
            system_prompt="sys",
        ),
        judge=JudgeConfig(provider="openrouter", model="fake-judge", system_prompt="sys"),
        probe=ProbeConfig(path=Path("unused.pkl"), threshold=0.5,
                          error_types=["false_positive"]),
        output=OutputConfig(jsonl_path=jsonl_path, run_id="test"),
        source_path=jsonl_path.parent / "config.md",
        eval=EvalConfig(),
    )


if __name__ == "__main__":
    test_progress_store_roundtrip()
    test_progress_store_tolerates_torn_line()
    test_summary_store_resume()
    test_run_redteam_skips_finished_rounds()
    test_resumed_round_sees_restored_memo()
    test_legacy_concurrent_path_checkpoints_rounds()
    print("\nall round-resume tests passed")
