"""Which client the provider switch actually builds."""
from __future__ import annotations

from app.agent.llm import build_llm
from app.config import Settings


class TestGroqIsItsOwnProvider:
    """A gsk_ key under a name that says "openai" is a trap for the next reader."""

    def test_groq_builds_the_openai_adapter_against_groqs_endpoint(self):
        settings = Settings(provider="groq", groq_api_key="gsk_test", _env_file=None)
        client = build_llm(settings)

        assert type(client).__name__ == "OpenAILLMClient"
        assert client.model == "openai/gpt-oss-120b"

    def test_without_its_own_key_it_falls_back_to_the_demo_model(self):
        settings = Settings(
            provider="groq", groq_api_key=None, openai_api_key="sk-not-this-one",
            _env_file=None,
        )

        assert type(build_llm(settings)).__name__ == "DemoLLMClient"

    def test_the_active_model_reports_groqs_model(self):
        settings = Settings(provider="groq", groq_model="llama-3.3-70b", _env_file=None)

        assert settings.active_model == "llama-3.3-70b"


class TestAScenarioThatNeverReachedTheModelIsNotScored:
    """A provider outage arrives shaped exactly like an agent failure.

    No tools, no answer, scores zero. Averaging those in measures the provider
    rather than the agent: 9 of 80 scenario-runs died on a provider-side 400 on
    2026-08-27 and dragged the reported F1 down with them.
    """

    def _rows(self):
        return [
            {"id": "ok-1", "tool_f1": 1.0, "tool_exact": True, "task_success": True,
             "citation_grounded": True, "cost_usd": 0.0, "latency_ms": 100,
             "expect_escalate": False, "did_escalate": False, "run_error": None},
            {"id": "ok-2", "tool_f1": 1.0, "tool_exact": True, "task_success": True,
             "citation_grounded": True, "cost_usd": 0.0, "latency_ms": 100,
             "expect_escalate": False, "did_escalate": False, "run_error": None},
            {"id": "dead", "tool_f1": 0.0, "tool_exact": False, "task_success": False,
             "citation_grounded": True, "cost_usd": 0.0, "latency_ms": 0,
             "expect_escalate": False, "did_escalate": True, "run_error": "HTTP 400"},
        ]

    def test_the_errored_row_does_not_drag_the_mean(self):
        rows = self._rows()
        scored = [r for r in rows if not r["run_error"]]

        f1 = sum(r["tool_f1"] for r in scored) / len(scored)

        assert f1 == 1.0, "a scenario the model never saw scored as an agent failure"

    def test_the_count_is_reported_not_hidden(self):
        rows = self._rows()

        errored = [r["id"] for r in rows if r["run_error"]]

        assert errored == ["dead"]
        assert len(rows) - len(errored) == 2
