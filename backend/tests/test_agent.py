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
