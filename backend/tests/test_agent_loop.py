"""The agent loop executes tools and writes a trace — verified with a scripted model."""
from __future__ import annotations

from sqlalchemy import func, select

from app.agent import AgentRunner
from app.agent.schema import AgentAnswer
from app.models import AgentStep, Conversation, Message
from app.testing.mock_model import MockLLMClient, make_final_response, make_tool_use_response
from app.tools import StubDocSearch, build_registry


def test_agent_loop_runs_tools_and_traces(db_session, seeded_customer_id):
    conv = Conversation(customer_id=seeded_customer_id)
    db_session.add(conv)
    db_session.flush()
    db_session.add(
        Message(conversation_id=conv.id, role="customer", content="What plan am I on?")
    )
    db_session.flush()

    scripted = [
        make_tool_use_response([("get_subscription", {"customer_id": seeded_customer_id})]),
        make_final_response(
            AgentAnswer(answer="You're on the Pro plan.", citations=[], confidence=0.9)
        ),
    ]
    runner = AgentRunner(
        MockLLMClient(scripted), build_registry(), StubDocSearch(), max_steps=5
    )
    answer = runner.run(db_session, conv)
    db_session.commit()

    assert isinstance(answer, AgentAnswer)
    assert "Pro" in answer.answer
    assert answer.should_escalate is False

    step_count = db_session.scalar(
        select(func.count()).select_from(AgentStep).where(
            AgentStep.conversation_id == conv.id
        )
    )
    assert step_count >= 2  # model turn + tool turn (+ final model turn)

    tool_steps = db_session.scalars(
        select(AgentStep).where(
            AgentStep.conversation_id == conv.id, AgentStep.kind == "tool"
        )
    ).all()
    assert any(s.tool_name == "get_subscription" for s in tool_steps)


def test_agent_escalates_on_max_steps(db_session, seeded_customer_id):
    conv = Conversation(customer_id=seeded_customer_id)
    db_session.add(conv)
    db_session.flush()
    db_session.add(Message(conversation_id=conv.id, role="customer", content="loop forever"))
    db_session.flush()

    # Model that never finalizes -> loop should hit max_steps and escalate.
    always_tool = [
        make_tool_use_response([("search_docs", {"query": "x", "top_k": 1})])
        for _ in range(10)
    ]
    runner = AgentRunner(
        MockLLMClient(always_tool), build_registry(), StubDocSearch(), max_steps=3
    )
    answer = runner.run(db_session, conv)
    assert answer.should_escalate is True
    assert answer.escalation_reason == "max_steps_exhausted"
