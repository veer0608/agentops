"""End-to-end HTTP test using the offline DemoLLMClient (no credentials)."""
from __future__ import annotations


def test_message_endpoint_offline(client):
    created = client.post("/conversations", json={})
    assert created.status_code == 200
    conversation_id = created.json()["id"]

    resp = client.post(
        f"/conversations/{conversation_id}/messages",
        json={"content": "How do I reset my password?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert body["steps"] >= 2

    detail = client.get(f"/conversations/{conversation_id}")
    assert detail.status_code == 200
    steps = detail.json()["steps"]
    assert any(s["tool_name"] == "search_docs" for s in steps)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
