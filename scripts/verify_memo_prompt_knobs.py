#!/usr/bin/env python
"""Pin the contracts of the two optional judge-memo prompt knobs.

`eval.data_description` (what the eval splits hold) and
`attacker.cross_iteration_memo_word_budget` (how long the cross-iteration memo may be)
are both optional, and both must be inert at their defaults.

The knob is optional prompt material for the judge's two summarizers, and the whole
point is that it is *inert* when unset — a run configured without it must send exactly
the prompts it sent before the knob existed, so every memo written by an older run
stays comparable to a new one. That is easy to break by accident: the coverage
paragraphs are concatenated into prompts that are themselves built by string
concatenation, and a stray space or a paragraph appended in the wrong place would not
show up in any output the run prints.

So this renders all six judge prompts (round summarizer system + user, cross-iteration
summarizer system + user, classification system + messages) both ways and asserts:

  1. UNSET  — every prompt is byte-identical to the pre-knob text, which is
     reconstructed here from the same building blocks the module exports.
  2. SET    — exactly four things change in the summarizers: one extra
     `## Task context` bullet in each user prompt, the coverage paragraph in each
     system prompt, question 6 in the round user prompt and the qualifier on question 3
     of the iteration one. With `judge.eval_scope_check` OFF the CLASSIFICATION prompt
     must not move at all — that is the setting in which the labeller stays blind to
     the eval data.
  3. SCOPE  — with `eval_scope_check` ON *and* a description set, the classification
     prompt gains the constraint block and the three scope fields in its JSON contract,
     and the round memo gains the rejected-samples paragraph and its question. Both are
     inert with either half missing, and the contrastive generator's prompt and cache
     key move with the description in exactly the same way.

No network, no GPU, no probe: it only builds prompt strings.

    python scripts/verify_memo_prompt_knobs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import agentic_redteam.llm_judge as L  # noqa: E402
import agentic_redteam.preprocessing as P  # noqa: E402
from agentic_redteam.llm_judge import LLMJudge  # noqa: E402
from agentic_redteam.persistence import AttemptRecord, Conversation, Message  # noqa: E402

PROBE_DESC = "A probe detecting whether the assistant's turn is harmful to humans."
# ONE kind, mirroring the shape the configs use and the shape the coverage paragraphs are
# written for: a title line, what the label is, an example on each side, and a closing
# sentence naming the surface cue that runs with the label. Multi-line, so the indentation
# of continuation lines under the `## Task context` bullet is exercised too.
EVAL_DESC = """A HARMFUL REQUEST, ANSWERED OR REFUSED.

WHAT THE LABEL IS. A property of the assistant's turn.
HARMFUL, in this data. The request is answered with operational detail.
NOT HARMFUL, in this data. The request is turned down in a single line.

Refusal is doing all the work here: on the non-harmful side almost every reply is a
bare refusal and nothing more."""


def _record(i: int, *, success: bool, positive: bool) -> AttemptRecord:
    return AttemptRecord(
        sample=Conversation(
            messages=(Message("user", f"u{i}"), Message("assistant", f"a{i}"))
        ),
        probe_score=0.1 * i,
        probe_predicts_positive=positive,
        judge_label="neg" if positive else "pos",
        judge_reason=f"reason {i}",
        judge_confidence=8,
        success=success,
        attacker_model="model",
        run_id="run",
        round=1,
        iteration=0,
        error_type="false_positive",
        pos_class_label="pos",
        neg_class_label="neg",
    )


RECORDS = [
    _record(1, success=True, positive=True),
    _record(2, success=False, positive=False),
    _record(3, success=False, positive=True),
]


def render(eval_description: str, *, scope_check: bool = False) -> dict[str, str]:
    """Every prompt this judge would send, keyed by name."""
    judge = LLMJudge(
        model="m",
        system_prompt="JUDGE PROMPT",
        pos_class_label="pos",
        neg_class_label="neg",
        provider="openrouter",
        max_tokens=1024,
        probe_description=PROBE_DESC,
        eval_data_description=eval_description,
        eval_scope_check=scope_check,
    )
    out: dict[str, str] = {}
    real = LLMJudge._summarization_call

    def capture(self, system, user_content, *, what):  # noqa: ANN001
        out[f"{what} / system"] = system
        out[f"{what} / user"] = user_content
        return "memo"

    LLMJudge._summarization_call = capture
    try:
        judge.summarize_round(
            RECORDS,
            round_num=1,
            error_type="false_positive",
            true_class_label="neg",
            prior_summary="PRIOR",
        )
        judge.summarize_iteration(
            RECORDS[:1],
            iteration=2,
            error_type="false_positive",
            true_class_label="neg",
            round_memo="ROUND MEMO",
            prior_memo="PRIOR MEMO",
            n_attempts=3,
        )
    finally:
        LLMJudge._summarization_call = real

    msgs, system = L._build_judge_request(
        Conversation(messages=(Message("user", "hi"), Message("assistant", "yo"))),
        "JUDGE PROMPT",
        "pos",
        "neg",
        PROBE_DESC,
        eval_description if scope_check else "",
    )
    out["classification / system"] = system
    out["classification / user"] = msgs[-1]["content"]
    return out


def main() -> int:
    off = render("")
    on = render(EVAL_DESC)
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(f"  {'ok  ' if cond else 'FAIL'}  {msg}")
        if not cond:
            failures.append(msg)

    print("unset — the knob must be inert:")
    # The pre-knob iteration system prompt is the constant itself; the pre-knob round
    # system prompt is the same function with no description. Both are reconstructed
    # from the module rather than pasted here, so this cannot rot into a stale copy.
    check(
        off["cross-cycle insights / system"] == L._ITERATION_SUMMARY_SYSTEM,
        "iteration system prompt == _ITERATION_SUMMARY_SYSTEM",
    )
    check(
        off["per-round insights / system"] == L._summary_system(1024),
        "round system prompt == _summary_system(max_tokens)",
    )
    for name, text in off.items():
        # Sentinel present in BOTH coverage paragraphs and in neither pre-knob prompt.
        # (The user prompts always carry a `## Task context` HEADER, which is why the
        # sentinel is the paragraph's opening clause and not the bare phrase.)
        check(
            "The Task context describes" not in text,
            f"no coverage text leaked into: {name}",
        )
    check(
        "\n5." not in off["per-round insights / user"],
        "round user prompt still ends at question 4",
    )
    check(
        L._eval_data_context_line("") == ""
        and L._eval_data_context_line("   \n ") == "",
        "a blank description renders as nothing (not an empty bullet)",
    )

    print("\nset — exactly the four summarizer insertions:")
    check(
        on["classification / system"] == off["classification / system"],
        "classification prompt UNCHANGED (the labeller never hears about the eval set)",
    )
    check(
        "How much of this round's evidence" in on["per-round insights / user"],
        "round user prompt gains the coverage question",
    )
    check(
        "in particular, what within the kind of conversation"
        in on["cross-cycle insights / user"],
        "iteration user prompt's question 3 gains the coverage qualifier",
    )
    # One marker per paragraph, since the round and iteration versions differ: both are
    # single-kind, but the iteration one hangs off section (3) of the hand-off memo.
    markers = {
        "per-round insights": "it is the yardstick for everything in the round",
        "cross-cycle insights": "section (3) is about what remains unexamined",
    }
    for what in ("per-round insights", "cross-cycle insights"):
        check(
            markers[what] in on[f"{what} / system"],
            f"{what} system prompt gains the coverage paragraph",
        )
        # Both paragraphs ask for the surface-cue reading, which is the clause the
        # configs' closing sentence is written to fire.
        check(
            "surface cue that runs with the label" in on[f"{what} / system"],
            f"{what} system prompt asks for the surface-cue reading",
        )
        bullet = "- The kinds of conversation the classifier is scored on:"
        check(bullet in on[f"{what} / user"], f"{what} user prompt gains the bullet")
        check(
            "\n  WHAT THE LABEL IS." in on[f"{what} / user"],
            f"{what}: continuation lines are indented under the bullet",
        )
    check(
        on["cross-cycle insights / system"].endswith(L._iteration_summary_tail()),
        "iteration coverage paragraph goes BEFORE the word budget, not after it",
    )

    print("\nscope check — the eval description as a CONSTRAINT (judge.eval_scope_check):")
    scope_off = render(EVAL_DESC, scope_check=False)
    scope_on = render(EVAL_DESC, scope_check=True)
    check(
        scope_off["classification / system"] == off["classification / system"]
        and scope_off["classification / user"] == off["classification / user"],
        "check OFF: classification prompt UNCHANGED (the labeller stays blind)",
    )
    check(
        L._scope_block("") == "" and L._scope_block("  \n ") == "",
        "no description ⇒ no constraint block, whatever the knob says",
    )
    check(
        "## The data this classifier is evaluated on" in scope_on["classification / system"]
        and EVAL_DESC in scope_on["classification / system"],
        "check ON: classification system prompt gains the constraint block",
    )
    check(
        all(
            k in scope_on["classification / user"]
            for k in ('"in_scope"', '"violated_constraint"', '"scope_reason"')
        )
        and not any(
            k in scope_off["classification / user"]
            for k in ("in_scope", "violated_constraint", "scope_reason")
        ),
        "check ON: the JSON contract gains the three scope fields, and only then",
    )
    check(
        "made after you have classified it" in scope_on["classification / system"]
        and "The label and confidence above are unaffected"
        in scope_on["classification / user"],
        "the scope verdict is asked for separately from, and after, the label",
    )
    check(
        "\n5. Which samples were REJECTED" in scope_on["per-round insights / user"]
        and "\n5. Which samples were REJECTED" not in scope_off["per-round insights / user"],
        "check ON: round user prompt gains the rejected-samples question",
    )
    check(
        "inside them by construction" in scope_on["per-round insights / system"]
        and "inside them by construction" not in scope_off["per-round insights / system"],
        "check ON: round system prompt gains the discourage-the-rejected-kind paragraph",
    )
    check(
        scope_on["cross-cycle insights / system"] == scope_off["cross-cycle insights / system"],
        "the cross-iteration memo prompt is untouched by the scope check",
    )

    print("\ncontrastive generator — the same description as a generation constraint:")
    gen_off = P._generation_system_prompt("pos", "neg")
    check(
        P._generation_system_prompt("pos", "neg", eval_data_description="") == gen_off
        and P._generation_system_prompt("pos", "neg", eval_data_description="  \n") == gen_off,
        "unset ⇒ generation prompt byte-identical to the pre-knob one",
    )
    gen_on = P._generation_system_prompt("pos", "neg", eval_data_description=EVAL_DESC)
    check(
        EVAL_DESC in gen_on
        and "must stay within what that description actually constrains" in gen_on,
        "set ⇒ the description and the generation constraint reach the prompt",
    )
    # The loosening is itself a contract: neither the scope ask nor the generation ask
    # may reintroduce a shape requirement the description does not state.
    check(
        "Do not treat details of its examples as requirements." in gen_on,
        "the generator is told the description's examples are not requirements",
    )
    # The loosening is a contract in its own right: the ask must reject only against what
    # the description STATES, and must say outright that an unstated shape is not grounds.
    check(
        "Reject it ONLY when it is not that task at all" in scope_on["classification / user"]
        and "Everything the description leaves open is in scope"
        in scope_on["classification / user"]
        and "a different layout" in scope_on["classification / user"],
        "the scope ask rejects only STATED constraints, never an inferred shape",
    )
    check(
        P._guidance_fingerprint("", "neg", None) == ""
        and P._guidance_fingerprint("", "neg", None, "") == "",
        "unset ⇒ empty fingerprint, so existing contrastive caches still hit",
    )
    check(
        P._guidance_fingerprint("cd", "neg", {"neg": "g"})
        != P._guidance_fingerprint("cd", "neg", {"neg": "g"}, EVAL_DESC),
        "set ⇒ a NEW cache key, so pairs are not reused across a changed constraint",
    )

    print("\nword budget — the memo's length knob:")
    check(
        L._iteration_summary_system() == L._ITERATION_SUMMARY_SYSTEM,
        "both knobs at their defaults reproduce _ITERATION_SUMMARY_SYSTEM exactly",
    )
    check(
        f"at most {L.DEFAULT_ITERATION_MEMO_WORD_BUDGET} words"
        in L._iteration_summary_system(),
        "the default budget is the one stated in the default prompt",
    )
    check(
        "at most 150 words" in L._iteration_summary_system("", 150),
        "a set budget reaches the prompt",
    )
    # A budget the judge cannot meet is worse than none: past the tight threshold the
    # instruction has to become "drop notes", not "compress everything".
    check(
        "That budget is tight" in L._iteration_summary_tail(L._ITERATION_MEMO_TIGHT_BUDGET)
        and "That budget is tight"
        not in L._iteration_summary_tail(L._ITERATION_MEMO_TIGHT_BUDGET + 1),
        f"the drop-notes clause switches on at <= {L._ITERATION_MEMO_TIGHT_BUDGET} words",
    )
    check(
        L._iteration_summary_system(EVAL_DESC, 150).endswith(
            L._iteration_summary_tail(150)
        ),
        "the two knobs compose: coverage paragraph before the tightened budget",
    )

    print()
    if failures:
        print(f"FAILED ({len(failures)} check(s))")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
