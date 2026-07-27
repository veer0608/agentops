"""The eval suite runs end-to-end and produces a scorecard (offline demo model)."""
from __future__ import annotations

from app.agent import DemoLLMClient
from evals.runner import run_suite


def test_run_suite_offline():
    result = run_suite(llm=DemoLLMClient(), write_reports=False, persist=False)
    summary = result["summary"]

    assert summary["n"] >= 8
    assert summary["model"] == "demo(offline)"
    for key in ("escalation_accuracy", "citation_grounding", "total_cost_usd", "latency_p50_ms"):
        assert key in summary
    # Doc scenarios should pass; account + escalation scenarios should not — so the
    # baseline lands strictly between 0 and 1, proving the scorers discriminate.
    assert 0.0 < summary["task_success_rate"] < 1.0
    assert 0.0 < summary["tool_selection_f1"] <= 1.0

    by_id = {r["id"]: r for r in result["rows"]}
    assert by_id["password-reset"]["task_success"] is True
    assert by_id["whats-my-plan"]["task_success"] is False  # demo can't read the account
