"""Forgiving JSON extraction from LLM replies.

Models asked for JSON routinely wrap it in prose or code fences, drop a closing
bracket, or get guillotined by ``max_tokens`` mid-array. The helpers here recover
what can be recovered: every fenced block is parsed independently (so a truncated
last block still yields the complete ones before it), balanced ``[...]`` / ``{...}``
spans are tried when there are no fences, and a fragment that fails to parse is
retried with the closers it is missing (:func:`json_repairs`). Every candidate still
has to pass the caller's shape check, so an over-eager repair fails closed rather
than inventing content.

The generator (``generator.py``) and the judge (``llm_judge.py``) both parse their
replies through :func:`extract_json_values`.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Iterator

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def iter_balanced(text: str, open_ch: str, close_ch: str):
    """Yield each top-level balanced ``open_ch..close_ch`` substring of ``text``.

    String-aware (delimiters inside JSON string literals don't count), so it
    correctly captures a whole nested structure. Unbalanced/truncated trailing
    fragments are simply not yielded.
    """
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                yield text[start : i + 1]
                start = -1



def _insert_missing_closers(fragment: str) -> str | None:
    """Insert closers for containers a *later* closer implies were finished.

    Handles ``[{...}, {"role": "assistant", "content": "…"\\n]`` — the last object never
    got its ``}``, so the ``]`` arrives while an object is still open. Appending closers
    at the end (:func:`_json_repairs`) cannot fix that: the delimiter is missing in the
    middle. Whenever a closer doesn't match the innermost open container, the closers for
    everything above it are emitted first.

    Returns None if nothing was inserted, so callers can skip a no-op candidate.
    """
    out: list[str] = []
    stack: list[str] = []
    in_str = esc = False
    changed = False
    for ch in fragment:
        out.append(ch)
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append(ch)
        elif ch in "]}":
            want = "[" if ch == "]" else "{"
            out.pop()  # re-emit after any implied closers
            while stack and stack[-1] != want:
                out.append("]" if stack[-1] == "[" else "}")
                stack.pop()
                changed = True
            if stack:
                stack.pop()
            out.append(ch)
    return "".join(out) if changed else None




def json_repairs(fragment: str) -> list[str]:
    """Repaired variants of a JSON fragment with missing closing delimiters.

    Models drop closing delimiters remarkably often, and arm C6's round 2 lost a whole
    session to it. All five of that reply's conversations were complete and well-formed;
    the first call ended every block ``…"}`` with **no ``]``**, and the two top-up calls
    re-emitted the same five with the final ``}`` of each missing instead. ``json.loads``
    rejected all of them, the balanced-bracket fallback found nothing to balance, and the
    session produced nothing from three calls. Repairing one character per block recovers
    all five.

    Candidates, in decreasing order of fidelity (``[]`` when the fragment is already
    balanced and there is nothing to repair):

    1. mid-fragment closers inserted (:func:`_insert_missing_closers`);
    2. missing trailing closers appended — the whole fragment survives;
    3. both of the above;
    4. cut back to the last complete top-level element, then closed — for a genuine
       mid-element truncation, where the trailing partial object is unusable.

    Scanning is string-aware throughout, so delimiters inside string literals don't
    count. Every candidate still has to parse *and* coerce to a message list before any
    caller uses it, so an over-eager repair fails closed rather than inventing content.
    """

    def _tail(text: str) -> tuple[str, int | None]:
        """(closers needed at the end, index just past the last complete top-level element)."""
        stack: list[str] = []
        in_str = esc = False
        last_complete: int | None = None
        for i, ch in enumerate(text):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "[{":
                stack.append(ch)
            elif ch in "]}":
                if stack:
                    stack.pop()
                if len(stack) == 1:
                    last_complete = i + 1
        return "".join("]" if c == "[" else "}" for c in reversed(stack)), last_complete

    closers, last_complete = _tail(fragment)
    inserted = _insert_missing_closers(fragment)

    out: list[str] = []
    if inserted is not None:
        out.append(inserted)
    if closers:
        out.append(fragment + closers)
    if inserted is not None:
        ins_closers, _ = _tail(inserted)
        if ins_closers:
            out.append(inserted + ins_closers)
    if closers and last_complete is not None:
        out.append(fragment[:last_complete] + closers[-1])
    # Preserve order, drop duplicates and any no-op.
    seen = {fragment}
    return [c for c in out if not (c in seen or seen.add(c))]




def loads_forgiving(fragment: str):
    """``json.loads``, retried on the repairs from :func:`_json_repairs`.

    Raises ``ValueError`` if neither the fragment nor any repair parses, so callers can
    keep treating a bad block as simply unparseable.
    """
    try:
        return json.loads(fragment)
    except (json.JSONDecodeError, ValueError):
        pass
    for repaired in json_repairs(fragment):
        try:
            return json.loads(repaired)
        except (json.JSONDecodeError, ValueError):
            continue
    raise ValueError("unparseable JSON fragment")



def extract_json_values(text: str | None, accept: Callable[[Any], Any | None]) -> list[Any]:
    """Every value in ``text`` that ``accept`` turns into something non-None.

    ``accept`` is the caller's shape check and normalizer: it is handed each parsed
    JSON value (a fenced block's contents, a balanced bracket span, or the whole
    text) and returns the normalized item, or None to reject it. Candidates are tried
    in decreasing order of reliability — fenced blocks first, each parsed on its own;
    then every balanced ``[...]`` and ``{...}`` span, only if no fence yielded
    anything; then the whole stripped text as a last resort — and parsing stops at the
    first tier that produced anything, so a fenced reply is never double-counted
    through its own bracket spans. Within the span tier, a span nested inside one that
    was already accepted is skipped for the same reason: the array of messages inside
    an accepted sample object is not a second payload.
    """
    if not text:
        return []
    out: list[Any] = []

    def _try(fragment: str) -> bool:
        try:
            value = loads_forgiving(fragment)
        except (json.JSONDecodeError, ValueError):
            return False
        item = accept(value)
        if item is None:
            return False
        out.append(item)
        return True

    for match in _FENCE_RE.finditer(text):
        _try(match.group(1).strip())
    if out:
        return out
    taken: list[tuple[int, int]] = []
    for start, span in _iter_spans(text):
        end = start + len(span)
        if any(a <= start and end <= b for a, b in taken):
            continue
        if _try(span):
            taken.append((start, end))
    if out:
        return out
    _try(text.strip())
    return out


def _iter_spans(text: str) -> Iterator[tuple[int, str]]:
    """``(offset, span)`` for every balanced ``[...]`` / ``{...}`` span, in document order.

    ``iter_balanced`` only yields spans that are top-level with respect to its own
    delimiter pair, so an object nested in an array (or vice versa) is still yielded;
    the caller decides what nesting means.
    """
    spans: list[tuple[int, str]] = []
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        pos = 0
        for s in iter_balanced(text, open_ch, close_ch):
            at = text.find(s, pos)
            spans.append((at, s))
            pos = at + 1
    seen: set[tuple[int, str]] = set()
    for item in sorted(spans, key=lambda t: (t[0], -len(t[1]))):
        if item not in seen:
            seen.add(item)
            yield item


def extract_string_list(
    text: str | None, keys: tuple[str, ...] = ("directions", "batches", "items")
) -> list[str]:
    """Every non-empty string in a JSON array in ``text`` (or under one of ``keys`` in a dict).

    Dict items are flattened to their string values joined by an em dash, so a
    ``[{"title": ..., "brief": ...}]`` shape still yields one string per item."""

    def _accept(value: Any):
        if isinstance(value, dict):
            for key in keys:
                if isinstance(value.get(key), list):
                    value = value[key]
                    break
            else:
                return None
        if not isinstance(value, list):
            return None
        strings = []
        for item in value:
            if isinstance(item, str) and item.strip():
                strings.append(item.strip())
            elif isinstance(item, dict):
                # {"direction": "..."} / {"title": ..., "brief": ...} shapes.
                text_parts = [str(v).strip() for v in item.values() if isinstance(v, str) and v.strip()]
                if text_parts:
                    strings.append(" — ".join(text_parts))
        return strings if strings else None

    out: list[str] = []
    for group in extract_json_values(text, _accept):
        out.extend(group)
    return out
