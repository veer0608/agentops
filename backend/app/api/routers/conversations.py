"""Conversation endpoints: start a conversation, send a message, read the trace."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agent import AgentRunner, build_llm, build_runner
from app.config import settings
from app.db import get_session
from app.models import AgentStep, Conversation, Message
from app.tools import build_registry, make_docsearch

router = APIRouter(prefix="/conversations", tags=["conversations"])

# Shared, stateless agent dependencies (built once). No network at import time:
# build_llm returns the offline DemoLLMClient unless a key is configured.
_llm = build_llm(settings)
_registry = build_registry()
_docsearch = make_docsearch()


def get_runner():
    return build_runner(
        _llm, _registry, _docsearch, max_steps=settings.max_agent_steps, kind=settings.runner
    )


class CreateConversation(BaseModel):
    customer_id: str | None = None


class ConversationOut(BaseModel):
    id: str
    customer_id: str | None
    status: str


class PostMessage(BaseModel):
    content: str


class AnswerOut(BaseModel):
    conversation_id: str
    answer: str
    citations: list[str]
    confidence: float
    should_escalate: bool
    escalation_reason: str | None
    steps: int


class MessageOut(BaseModel):
    role: str
    content: str


class StepOut(BaseModel):
    step_no: int
    kind: str
    tool_name: str | None
    tool_result: str | None
    latency_ms: int | None
    model: str | None


class ConversationDetail(BaseModel):
    id: str
    customer_id: str | None
    status: str
    messages: list[MessageOut]
    steps: list[StepOut]


@router.post("", response_model=ConversationOut)
def create_conversation(
    body: CreateConversation, session: Session = Depends(get_session)
) -> ConversationOut:
    conv = Conversation(customer_id=body.customer_id)
    session.add(conv)
    session.commit()
    session.refresh(conv)
    return ConversationOut(id=conv.id, customer_id=conv.customer_id, status=conv.status)


@router.post("/{conversation_id}/messages", response_model=AnswerOut)
def post_message(
    conversation_id: str,
    body: PostMessage,
    session: Session = Depends(get_session),
    runner: AgentRunner = Depends(get_runner),
) -> AnswerOut:
    conv = session.get(Conversation, conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    session.add(Message(conversation_id=conv.id, role="customer", content=body.content))
    session.flush()

    answer = runner.run(session, conv)

    session.add(Message(conversation_id=conv.id, role="agent", content=answer.answer))
    conv.status = "escalated" if answer.should_escalate else "answered"
    session.commit()

    step_count = session.scalar(
        select(func.count()).select_from(AgentStep).where(
            AgentStep.conversation_id == conv.id
        )
    )
    return AnswerOut(
        conversation_id=conv.id,
        answer=answer.answer,
        citations=answer.citations,
        confidence=answer.confidence,
        should_escalate=answer.should_escalate,
        escalation_reason=answer.escalation_reason,
        steps=int(step_count or 0),
    )


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: str, session: Session = Depends(get_session)
) -> ConversationDetail:
    conv = session.get(Conversation, conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return ConversationDetail(
        id=conv.id,
        customer_id=conv.customer_id,
        status=conv.status,
        messages=[MessageOut(role=m.role, content=m.content) for m in conv.messages],
        steps=[
            StepOut(
                step_no=s.step_no,
                kind=s.kind,
                tool_name=s.tool_name,
                tool_result=s.tool_result,
                latency_ms=s.latency_ms,
                model=s.model,
            )
            for s in conv.steps
        ],
    )
