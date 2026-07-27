"""Deterministic scorers for Phase 1: tool-selection and task-success.

Phase 3 adds an LLM judge for hallucination and fuzzy answer quality; Phase 2
adds a dedicated escalation-accuracy scorer once the policy gate exists.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ToolScore:
    precision: float
    recall: float
    f1: float
    exact: bool


def score_tool_selection(expected: list[str], actual: list[str]) -> ToolScore:
    """Set-based P/R/F1 over tool names, plus an exact-set match flag."""
    exp, act = set(expected), set(actual)
    if not exp and not act:
        return ToolScore(1.0, 1.0, 1.0, True)
    tp = len(exp & act)
    fp = len(act - exp)
    fn = len(exp - act)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return ToolScore(precision, recall, f1, exp == act)


def score_task_success(
    *,
    expect_escalate: bool | None,
    answer_keywords: list[str],
    answer_text: str,
    did_escalate: bool,
) -> bool:
    """Success = escalation matches expectation AND all required keywords present."""
    escalate_ok = expect_escalate is None or (did_escalate == expect_escalate)
    text = answer_text.lower()
    keywords_ok = all(k.lower() in text for k in answer_keywords)
    return escalate_ok and keywords_ok


def score_escalation_accuracy(labeled: list[tuple[bool, bool]]) -> float:
    """Accuracy of the escalate/don't-escalate decision over labeled scenarios."""
    if not labeled:
        return 1.0
    return sum(1 for expected, actual in labeled if expected == actual) / len(labeled)


_MARKER_RE = re.compile(r"\[([A-Za-z]+-\d+)\]")


def extract_markers(text: str) -> set[str]:
    """Pull citation markers like DOC-101 / GH-102 out of tool-result text."""
    return set(_MARKER_RE.findall(text or ""))


def score_citation_grounding(cited: list[str], retrieved: set[str]) -> bool:
    """True if every cited marker was actually retrieved this run (or none cited).

    A deterministic hallucination proxy: the agent may only cite what a tool
    returned. An LLM judge (Phase 3.5) can replace this when a key is available.
    """
    if not cited:
        return True
    return all(c in retrieved for c in cited)
