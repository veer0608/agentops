"""Hermetic tests for the OpenAI message/tool adapter (no network)."""
from __future__ import annotations

import json

from app.agent.openai_llm import _to_openai_messages, _to_openai_tools


def test_to_openai_messages_translates_blocks():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "let me check"},
                {"type": "tool_use", "id": "call_1", "name": "get_subscription", "input": {"customer_id": "c1"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": '{"plan":"Pro"}', "is_error": False}
            ],
        },
    ]
    out = _to_openai_messages("sys", messages)

    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1] == {"role": "user", "content": "hi"}

    asst = out[2]
    assert asst["role"] == "assistant"
    assert asst["tool_calls"][0]["id"] == "call_1"
    assert asst["tool_calls"][0]["function"]["name"] == "get_subscription"
    assert json.loads(asst["tool_calls"][0]["function"]["arguments"]) == {"customer_id": "c1"}

    tool_msg = out[3]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_1"
    assert "Pro" in tool_msg["content"]


def test_google_client_uses_gemini_endpoint():
    from app.agent.google_llm import GoogleLLMClient

    client = GoogleLLMClient(api_key="x", model="gemini-2.0-flash")
    assert client.model == "gemini-2.0-flash"
    assert "generativelanguage" in str(client._client.base_url)


def test_to_openai_tools_shape():
    tools = [
        {
            "name": "get_customer",
            "description": "Look up a customer",
            "input_schema": {"type": "object", "properties": {"email": {"type": "string"}}},
        }
    ]
    out = _to_openai_tools(tools)
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "get_customer"
    assert out[0]["function"]["parameters"]["type"] == "object"
