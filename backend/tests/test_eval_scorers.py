"""Unit tests for the eval scorers."""
from __future__ import annotations

from evals.scorers import (
    score_escalation_accuracy,
    score_task_success,
    score_tool_selection,
)


def test_tool_selection_exact_match():
    s = score_tool_selection(["get_customer", "get_subscription"], ["get_subscription", "get_customer"])
    assert s.exact is True
    assert s.f1 == 1.0


def test_tool_selection_partial():
    s = score_tool_selection(["get_customer", "get_subscription"], ["search_docs"])
    assert s.exact is False
    assert s.f1 == 0.0


def test_tool_selection_both_empty():
    s = score_tool_selection([], [])
    assert s.exact is True
    assert s.f1 == 1.0


def test_task_success_keywords_and_escalation():
    assert score_task_success(
        expect_escalate=False,
        answer_keywords=["pro", "2026-08-01"],
        answer_text="You're on the Pro plan, renewing 2026-08-01.",
        did_escalate=False,
    )
    # keyword missing -> fail
    assert not score_task_success(
        expect_escalate=False,
        answer_keywords=["2026-08-01"],
        answer_text="You're on the Pro plan.",
        did_escalate=False,
    )
    # escalation mismatch -> fail
    assert not score_task_success(
        expect_escalate=True,
        answer_keywords=[],
        answer_text="Sure, done.",
        did_escalate=False,
    )


def test_escalation_accuracy():
    assert score_escalation_accuracy([(True, True), (False, False)]) == 1.0
    assert score_escalation_accuracy([(True, False), (False, False)]) == 0.5
    assert score_escalation_accuracy([]) == 1.0
