"""The hand-rolled AgentRunner and the LangGraph GraphAgentRunner behave identically.

Runs the whole scenario suite through both (offline demo model) and asserts the
aggregate metrics and per-scenario outcomes match — the payoff of the runner seam.
"""
from __future__ import annotations

from app.agent import DemoLLMClient
from evals.runner import run_suite


def test_manual_and_langgraph_parity():
    manual = run_suite(
        llm=DemoLLMClient(), runner_kind="manual", write_reports=False, persist=False
    )
    graph = run_suite(
        llm=DemoLLMClient(), runner_kind="langgraph", write_reports=False, persist=False
    )

    ms, gs = manual["summary"], graph["summary"]
    for key in (
        "tool_selection_f1",
        "tool_exact_match_rate",
        "task_success_rate",
        "escalation_accuracy",
        "citation_grounding",
    ):
        assert ms[key] == gs[key], (key, ms[key], gs[key])

    mrows = {r["id"]: r for r in manual["rows"]}
    grows = {r["id"]: r for r in graph["rows"]}
    assert mrows.keys() == grows.keys()
    for sid, mr in mrows.items():
        gr = grows[sid]
        assert mr["actual_tools"] == gr["actual_tools"], sid
        assert mr["task_success"] == gr["task_success"], sid
        assert mr["answer"] == gr["answer"], sid
