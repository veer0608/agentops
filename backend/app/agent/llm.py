"""The model seam.

`LLMClient` is the boundary the agent loop is written against. Three impls:
- AnthropicLLMClient — the real model (stable Messages API + tool use).
- DemoLLMClient — deterministic offline stand-in when no API key is set.
- MockLLMClient (in app/testing) — scripted, for unit tests.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Protocol

from app.agent.schema import FINAL_TOOL_NAME, AgentAnswer


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict


@dataclass
class ModelResponse:
    text: str = ""
    tool_uses: list[ToolUse] = field(default_factory=list)
    stop_reason: str = "end_turn"
    # The assistant content blocks to append verbatim as history for the next turn.
    assistant_content: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    model: str = ""


class LLMClient(Protocol):
    def create(
        self, *, system: str, messages: list[dict], tools: list[dict]
    ) -> ModelResponse: ...


def _tid() -> str:
    return f"toolu_{uuid.uuid4().hex[:12]}"


def make_tool_use_response(
    calls: list[tuple[str, dict]], text: str = "", model: str = "mock"
) -> ModelResponse:
    content: list[dict] = []
    tool_uses: list[ToolUse] = []
    if text:
        content.append({"type": "text", "text": text})
    for name, inp in calls:
        tid = _tid()
        tool_uses.append(ToolUse(id=tid, name=name, input=inp))
        content.append({"type": "tool_use", "id": tid, "name": name, "input": inp})
    return ModelResponse(
        text=text,
        tool_uses=tool_uses,
        stop_reason="tool_use",
        assistant_content=content,
        model=model,
    )


def make_final_response(answer: AgentAnswer, model: str = "mock") -> ModelResponse:
    inp = answer.model_dump()
    tid = _tid()
    content = [{"type": "tool_use", "id": tid, "name": FINAL_TOOL_NAME, "input": inp}]
    return ModelResponse(
        text="",
        tool_uses=[ToolUse(id=tid, name=FINAL_TOOL_NAME, input=inp)],
        stop_reason="tool_use",
        assistant_content=content,
        model=model,
    )


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


class AnthropicLLMClient:
    """Real model via the stable Messages API + tool use."""

    def __init__(self, api_key: str, model: str) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def create(self, *, system, messages, tools) -> ModelResponse:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            # cache the stable system prompt + tool prefix to cut cost on repeat turns
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            tools=tools,
        )
        text_parts: list[str] = []
        tool_uses: list[ToolUse] = []
        assistant_content: list[dict] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
                assistant_content.append({"type": "text", "text": block.text})
            elif btype == "tool_use":
                inp = dict(block.input) if block.input else {}
                tool_uses.append(ToolUse(id=block.id, name=block.name, input=inp))
                assistant_content.append(
                    {"type": "tool_use", "id": block.id, "name": block.name, "input": inp}
                )
        usage = getattr(resp, "usage", None)
        return ModelResponse(
            text="".join(text_parts),
            tool_uses=tool_uses,
            stop_reason=resp.stop_reason or "end_turn",
            assistant_content=assistant_content,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            model=getattr(resp, "model", self.model),
        )


class DemoLLMClient:
    """Offline, deterministic stand-in: search_docs -> respond_to_customer.

    Stateless — decides its move from the shape of the last message, so it works
    as a singleton across requests. Lets the HTTP endpoint demonstrate the full
    loop (a real tool call + a grounded, structured answer + a trace) with no key.
    """

    def create(self, *, system, messages, tools) -> ModelResponse:
        last = messages[-1] if messages else {"content": ""}
        content = last.get("content")
        is_tool_result_turn = isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        )
        if is_tool_result_turn:
            results = [
                b.get("content", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "tool_result"
            ]
            joined = "\n".join(r for r in results if isinstance(r, str))
            cites = list(dict.fromkeys(re.findall(r"\[([A-Za-z0-9\-]+)\]", joined)))
            answer = AgentAnswer(
                answer=(
                    "Here's what our help center says:\n" + joined
                    if joined.strip()
                    else "I couldn't find anything relevant in the help center."
                ),
                citations=cites,
                confidence=0.4,
            )
            return make_final_response(answer, model="demo")
        query = _text_of(content) or "help"
        return make_tool_use_response(
            [("search_docs", {"query": query, "top_k": 3})], model="demo"
        )


def build_llm(settings) -> LLMClient:
    """Live model for the configured provider if its key is set, else offline demo."""
    if settings.provider == "openai":
        if settings.openai_api_key:
            from app.agent.openai_llm import OpenAILLMClient

            return OpenAILLMClient(api_key=settings.openai_api_key, model=settings.openai_model)
        return DemoLLMClient()
    if settings.provider == "google":
        if settings.google_api_key:
            from app.agent.google_llm import GoogleLLMClient

            return GoogleLLMClient(api_key=settings.google_api_key, model=settings.google_model)
        return DemoLLMClient()
    if settings.anthropic_api_key:
        return AnthropicLLMClient(
            api_key=settings.anthropic_api_key, model=settings.anthropic_model
        )
    return DemoLLMClient()
