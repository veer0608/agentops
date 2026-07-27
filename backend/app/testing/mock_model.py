"""Scripted LLM client for unit tests: returns a fixed sequence of ModelResponses."""
from __future__ import annotations

from app.agent.llm import ModelResponse, make_final_response, make_tool_use_response
from app.agent.schema import AgentAnswer

__all__ = ["MockLLMClient", "make_tool_use_response", "make_final_response", "AgentAnswer"]


class MockLLMClient:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self._i = 0

    def create(self, *, system, messages, tools) -> ModelResponse:
        if self._i >= len(self._responses):
            # Safety net so a mis-scripted test can't loop forever.
            return make_final_response(
                AgentAnswer(answer="(mock responses exhausted)", confidence=0.5)
            )
        resp = self._responses[self._i]
        self._i += 1
        return resp
