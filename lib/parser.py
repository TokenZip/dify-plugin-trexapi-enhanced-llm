"""Two-step TZP marker scanner.

Implements TZP v1.0 Specification Section 4.4.
Adapted from rag_interceptor/parser.py — no changes needed.
"""

from __future__ import annotations

import re

from .models import TagMatch

_TZP_MARKER_RE = re.compile(r"\[TZP:\s*(tx_[a-z]{2}_[a-zA-Z0-9]{9})\]")
_TZP_ESCAPED_RE = re.compile(r"\\(\[TZP:\s*tx_[a-z]{2}_[a-zA-Z0-9]{9}\])")
_FENCED_CODE_RE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


def _build_exclusion_intervals(prompt: str) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for pat in (_FENCED_CODE_RE, _INLINE_CODE_RE, _TZP_ESCAPED_RE):
        for m in pat.finditer(prompt):
            intervals.append((m.start(), m.end()))
    if not intervals:
        return []
    intervals.sort()
    merged: list[tuple[int, int]] = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _in_excluded(pos: int, intervals: list[tuple[int, int]]) -> bool:
    lo, hi = 0, len(intervals) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s, e = intervals[mid]
        if pos < s:
            hi = mid - 1
        elif pos >= e:
            lo = mid + 1
        else:
            return True
    return False


def extract_tzp_tags(prompt: str) -> list[TagMatch]:
    exclusions = _build_exclusion_intervals(prompt)
    tags: list[TagMatch] = []
    for m in _TZP_MARKER_RE.finditer(prompt):
        if exclusions and _in_excluded(m.start(), exclusions):
            continue
        if m.start() > 0 and prompt[m.start() - 1] == "\\":
            continue
        tags.append(TagMatch(trex_id=m.group(1), start=m.start(), end=m.end()))
    return tags


def strip_escaped_markers(prompt: str) -> str:
    return _TZP_ESCAPED_RE.sub(r"\1", prompt)


def replace_tzp_tags(
    prompt: str, tags: list[TagMatch], replacements: dict[str, str]
) -> str:
    for tag in sorted(tags, key=lambda t: t.start, reverse=True):
        replacement = replacements.get(tag.trex_id, "")
        prompt = prompt[: tag.start] + replacement + prompt[tag.end :]
    return prompt
