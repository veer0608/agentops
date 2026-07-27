"""Agent core: the tool-use loop, model seam, and structured answer."""
from __future__ import annotations

from app.agent.llm import (
    AnthropicLLMClient,
    DemoLLMClient,
    LLMClient,
    ModelResponse,
    ToolUse,
    build_llm,
    make_final_response,
    make_tool_use_response,
)
from app.agent.runner import AgentRunner
from app.agent.schema import FINAL_TOOL_NAME, AgentAnswer


def build_runner(llm, registry, docsearch, max_steps=8, policy=None, kind="manual"):
    """Return the hand-rolled AgentRunner or the LangGraph GraphAgentRunner.

    Same interface either way — the seam that makes the two swappable is the point.
    GraphAgentRunner is imported lazily so LangGraph is only needed when used.
    """
    if kind == "langgraph":
        from app.agent.graph_runner import GraphAgentRunner

        return GraphAgentRunner(llm, registry, docsearch, max_steps, policy)
    return AgentRunner(llm, registry, docsearch, max_steps, policy)


__all__ = [
    "AgentRunner",
    "build_runner",
    "AgentAnswer",
    "FINAL_TOOL_NAME",
    "LLMClient",
    "ModelResponse",
    "ToolUse",
    "AnthropicLLMClient",
    "DemoLLMClient",
    "build_llm",
    "make_tool_use_response",
    "make_final_response",
]
