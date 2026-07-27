"""OpenAI implementation of the LLMClient seam (Chat Completions + tools).

The agent loop speaks a canonical Anthropic-style block dialect (text / tool_use
/ tool_result). This client is an adapter: it converts that dialect to OpenAI's
message + tool-call format on the way out, and converts OpenAI's response back
into the same ModelResponse shape (with `assistant_content` as canonical blocks),
so the runner, tools, and eval harness stay provider-agnostic.
"""
from __future__ import annotations

import json

from app.agent.llm import ModelResponse, ToolUse


def _to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "assistant":
            if isinstance(content, str):
                out.append({"role": "assistant", "content": content})
                continue
            text = "".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
            tool_calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}))},
                }
                for b in content
                if isinstance(b, dict) and b.get("type") == "tool_use"
            ]
            msg: dict = {"role": "assistant", "content": text or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
        elif role == "user":
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
                continue
            tool_results = [
                b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"
            ]
            if tool_results:
                for b in tool_results:
                    c = b.get("content", "")
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": b["tool_use_id"],
                            "content": c if isinstance(c, str) else json.dumps(c),
                        }
                    )
            else:
                text = " ".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
                out.append({"role": "user", "content": text})
        else:
            out.append(
                {"role": role, "content": content if isinstance(content, str) else json.dumps(content)}
            )
    return out


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


class OpenAILLMClient:
    """Real model via OpenAI Chat Completions + tool calling."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        from openai import OpenAI

        self._client = (
            OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        )
        self.model = model

    def create(self, *, system, messages, tools) -> ModelResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=_to_openai_messages(system, messages),
            tools=_to_openai_tools(tools),
            tool_choice="auto",
        )
        choice = resp.choices[0]
        msg = choice.message

        text = msg.content or ""
        tool_uses: list[ToolUse] = []
        assistant_content: list[dict] = []
        if text:
            assistant_content.append({"type": "text", "text": text})
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_uses.append(ToolUse(id=tc.id, name=tc.function.name, input=args))
            assistant_content.append(
                {"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": args}
            )

        if choice.finish_reason == "content_filter":
            stop_reason = "refusal"
        elif tool_uses:
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"

        usage = getattr(resp, "usage", None)
        return ModelResponse(
            text=text,
            tool_uses=tool_uses,
            stop_reason=stop_reason,
            assistant_content=assistant_content,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cache_read_tokens=0,
            model=getattr(resp, "model", self.model),
        )
