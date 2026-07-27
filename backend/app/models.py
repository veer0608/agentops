"""ORM models — Phase 0 subset: mock CRM + conversation trace.

Phase 2 adds tickets / review_queue / audit_log; Phase 1 adds eval_runs.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    tier: Mapped[str] = mapped_column(String, nullable=False, default="free")  # free|pro|enterprise
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    customer_id: Mapped[str] = mapped_column(
        ForeignKey("customers.id"), index=True, nullable=False
    )
    plan: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")  # active|past_due|canceled
    renews_at: Mapped[str | None] = mapped_column(String, nullable=True)  # ISO date
    mrr_cents: Mapped[int] = mapped_column(Integer, default=0)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    customer_id: Mapped[str | None] = mapped_column(
        ForeignKey("customers.id"), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(String, nullable=False, default="open")  # open|answered|escalated
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
    steps: Mapped[list["AgentStep"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="AgentStep.step_no",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id"), index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(String, nullable=False)  # customer|agent
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class AgentStep(Base):
    """One row per model turn or tool execution — the trace + eval data source."""

    __tablename__ = "agent_steps"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id"), index=True, nullable=False
    )
    step_no: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False)  # model|tool
    thought: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_name: Mapped[str | None] = mapped_column(String, nullable=True)
    tool_args: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    tool_result: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cache_read_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    conversation: Mapped["Conversation"] = relationship(back_populates="steps")


class EvalRun(Base):
    """One row per eval suite run — the before/after tracking surface."""

    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    model: Mapped[str] = mapped_column(String, nullable=False)
    doc_search_mode: Mapped[str] = mapped_column(String, nullable=False)
    n_scenarios: Mapped[int] = mapped_column(Integer, nullable=False)
    tool_selection_f1: Mapped[float] = mapped_column(Float, nullable=False)
    tool_exact_match_rate: Mapped[float] = mapped_column(Float, nullable=False)
    task_success_rate: Mapped[float] = mapped_column(Float, nullable=False)
    escalation_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    citation_grounding: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_p50_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    report_path: Mapped[str | None] = mapped_column(String, nullable=True)


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    customer_id: Mapped[str | None] = mapped_column(
        ForeignKey("customers.id"), nullable=True, index=True
    )
    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey("conversations.id"), nullable=True, index=True
    )
    subject: Mapped[str] = mapped_column(String, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False, default="")
    priority: Mapped[str] = mapped_column(String, nullable=False, default="normal")
    status: Mapped[str] = mapped_column(String, nullable=False, default="open")
    idempotency_key: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class ReviewQueue(Base):
    __tablename__ = "review_queue"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey("conversations.id"), nullable=True, index=True
    )
    proposed_action: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")  # pending|approved|rejected
    resolution_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AuditLog(Base):
    """Append-only record of every side-effecting decision and action."""

    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    conversation_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    actor: Mapped[str] = mapped_column(String, nullable=False)  # agent|policy|human
    action: Mapped[str] = mapped_column(String, nullable=False)
    decision: Mapped[str | None] = mapped_column(String, nullable=True)  # auto|review|block|executed|rejected
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    target: Mapped[str | None] = mapped_column(String, nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class IdempotencyKey(Base):
    __tablename__ = "idempotency_keys"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    tool_name: Mapped[str] = mapped_column(String, nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
