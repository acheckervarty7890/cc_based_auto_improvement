"""Verbose single-example dry run.

Loads `configs/example_config.md`, points it at `data/probe_llama1b.pkl`, runs the
attacker for a tightly capped number of turns, and prints every input/output:

  - Probe metadata (what `get_probe_info` would return)
  - The full attacker system prompt
  - The attacker kickoff user message
  - For each candidate the attacker submits:
      - The conversation (what the probe and judge see)
      - The probe's raw score, threshold, predicted class
      - The exact judge system prompt and user message
      - The judge's raw model output
      - The parsed judge verdict (label / reason / confidence)
      - The persisted JSONL row

Uses the real probe and the real judge — costs API tokens.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

from agentic_redteam.attacker import (
    _DISALLOWED_TOOLS,
    _build_full_system_prompt,
)
from agentic_redteam.config import load_config
from agentic_redteam.llm_judge import (
    JudgeVerdict,
    LLMJudge,
    _build_judge_request,
    _normalize_label,
    _parse_judge_json,
)
from agentic_redteam.persistence import JsonlStore
from agentic_redteam.probe_judge import ProbeJudge
from agentic_redteam.tools import ToolContext, allowed_tool_names, build_mcp_server


SEP = "=" * 80


def hr(title: str) -> None:
    print()
    print(SEP)
    print(title)
    print(SEP)


def patch_judge_with_tracing() -> None:
    counter = {"n": 0}

    def traced(self: LLMJudge, conversation) -> JudgeVerdict:
        counter["n"] += 1
        idx = counter["n"]

        messages, system = _build_judge_request(
            conversation,
            self.system_prompt,
            self.pos_class_label,
            self.neg_class_label,
        )

        hr(f"JUDGE CALL #{idx} — SYSTEM PROMPT (sent verbatim)")
        print(system)

        hr(f"JUDGE CALL #{idx} — MESSAGES ARRAY (sent verbatim, multi-turn)")
        for i, m in enumerate(messages):
            print(f"--- messages[{i}] role={m['role']!r} ---")
            print(m["content"])

        client = self._client_or_init()
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )
        raw_text = "".join(b.text for b in response.content if b.type == "text")

        hr(f"JUDGE CALL #{idx} — RAW MODEL OUTPUT")
        print(raw_text)

        raw_label, reason, confidence = _parse_judge_json(raw_text)
        normalized = _normalize_label(raw_label, self.pos_class_label, self.neg_class_label)
        verdict = JudgeVerdict(
            label=normalized,
            reason=reason or raw_text.strip()[:500],
            confidence=confidence,
        )

        hr(f"JUDGE CALL #{idx} — PARSED VERDICT")
        print(f"label:      {verdict.label!r}")
        print(f"reason:     {verdict.reason!r}")
        print(f"confidence: {verdict.confidence}")
        return verdict

    LLMJudge.evaluate = traced  # type: ignore[assignment]


def patch_probe_with_tracing() -> None:
    original_score = ProbeJudge.score
    counter = {"n": 0}

    def traced_score(self: ProbeJudge, conversation) -> float:
        counter["n"] += 1
        idx = counter["n"]
        hr(f"PROBE CALL #{idx} — CONVERSATION SENT TO PROBE")
        for m in conversation.messages:
            print(f"[{m.role}] {m.content}")
        s = original_score(self, conversation)
        predicts_pos = s >= self.threshold
        predicted_label = self.pos_class_label if predicts_pos else self.neg_class_label
        hr(f"PROBE CALL #{idx} — SCORE & PREDICTION")
        print(f"raw_score:           {s:.6f}")
        print(f"threshold:           {self.threshold}")
        print(f"predicts_positive:   {predicts_pos}")
        print(f"predicted_label:     {predicted_label!r}")
        return s

    ProbeJudge.score = traced_score  # type: ignore[assignment]


async def main(config_path: str) -> None:
    config = load_config(config_path)

    # Tight cap so we observe ~1 candidate end-to-end.
    config.attacker.batch_target = 1
    config.attacker.max_turns = 6

    probe = ProbeJudge.load(
        config.probe.path,
        threshold=config.probe.threshold,
        error_type=config.probe.error_type,
    )
    judge = LLMJudge(
        model=config.judge.model,
        system_prompt=config.judge.system_prompt,
        pos_class_label=probe.pos_class_label,
        neg_class_label=probe.neg_class_label,
        max_tokens=config.judge.max_tokens,
    )

    hr("CONFIG SUMMARY")
    print(f"Probe path:                  {config.probe.path}")
    print(f"Probe threshold:             {config.probe.threshold}")
    print(f"Error type:                  {config.probe.error_type}")
    print(f"Attacker models:             {config.attacker.models}")
    print(f"Attacker max_turns (capped): {config.attacker.max_turns}")
    print(f"Attacker batch_target:       {config.attacker.batch_target}")
    print(f"Judge model:                 {config.judge.model}")
    print(f"Judge confidence_threshold:  {config.judge.confidence_threshold}")

    hr("PROBE METADATA (what get_probe_info returns to the attacker)")
    info = {
        "model_name": probe.model_name,
        "layer": probe.layer,
        "pos_class_label": probe.pos_class_label,
        "neg_class_label": probe.neg_class_label,
        "description": probe.description,
        "error_type": probe.error_type,
        "threshold": probe.threshold,
        "true_class_label_for_success": probe.true_class_label,
    }
    print(json.dumps(info, indent=2))

    model_name = config.attacker.models[0]
    full_system_prompt = _build_full_system_prompt(config, probe)

    hr("ATTACKER SYSTEM PROMPT (sent verbatim)")
    print(full_system_prompt)
    hr("ATTACKER TRIGGER USER MESSAGE (sent verbatim)")
    print("Begin.")

    patch_judge_with_tracing()
    patch_probe_with_tracing()

    dry_run_jsonl = Path(__file__).resolve().parent.parent / "results" / "dry_run.jsonl"
    if dry_run_jsonl.exists():
        dry_run_jsonl.unlink()
    store = JsonlStore(path=dry_run_jsonl)

    ctx = ToolContext(
        probe=probe,
        judge=judge,
        store=store,
        run_id="dry-run",
        confidence_threshold=config.judge.confidence_threshold,
    )
    ctx.set_round(0)
    ctx.set_attacker_model(model_name)

    server = build_mcp_server(ctx)

    options = ClaudeAgentOptions(
        system_prompt=full_system_prompt,
        model=model_name,
        max_turns=config.attacker.max_turns,
        mcp_servers={"redteam": server},
        allowed_tools=allowed_tool_names(),
        disallowed_tools=_DISALLOWED_TOOLS,
        permission_mode="bypassPermissions",
        setting_sources=[],
    )

    hr("STARTING LIVE ATTACKER LOOP")
    async with ClaudeSDKClient(options=options) as client:
        await client.query("Begin.")
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text = block.text.strip()
                        if text:
                            hr("ATTACKER ASSISTANT TEXT")
                            print(text)
                    elif isinstance(block, ToolUseBlock):
                        hr(f"ATTACKER TOOL CALL: {block.name}")
                        print(json.dumps(block.input, indent=2, ensure_ascii=False))
            elif isinstance(message, ResultMessage):
                hr("RESULT MESSAGE")
                print(f"stop_reason: {getattr(message, 'stop_reason', None)}")
                break

    hr("PERSISTED JSONL ROWS")
    for rec in store.iter_all():
        print(rec.to_jsonl_row())


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/example_config.md"
    asyncio.run(main(config_path))
