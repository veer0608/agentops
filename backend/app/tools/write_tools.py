"""Phase 2 write tools: create_ticket, update_ticket, escalate_to_human.

Writes are idempotent (an idempotency_key replays the stored result) and are
classified WRITE so the policy gate evaluates them before execution.
"""
from __future__ import annotations

import json
from collections.abc import Callable

from pydantic import BaseModel, Field

from app.models import IdempotencyKey, Ticket
from app.tools.base import SideEffect, Tool, ToolContext, ToolResult


def _idempotent(
    ctx: ToolContext, key: str | None, tool_name: str, produce: Callable[[], ToolResult]
) -> ToolResult:
    if not key:
        return produce()
    existing = ctx.session.get(IdempotencyKey, key)
    if existing is not None:
        return ToolResult(content=existing.response, data={"idempotent_replay": True})
    result = produce()
    ctx.session.add(
        IdempotencyKey(key=key, tool_name=tool_name, response=result.content)
    )
    ctx.session.flush()
    return result


# --- create_ticket ---------------------------------------------------------
class CreateTicketArgs(BaseModel):
    subject: str = Field(description="Short ticket title")
    body: str = Field(default="", description="Details of the issue")
    priority: str = Field(default="normal", description="low | normal | high | urgent")
    idempotency_key: str | None = Field(
        default=None, description="Repeat-safe key; a repeat returns the same ticket"
    )
    confidence: float | None = Field(
        default=None,
        description="0-1 confidence this action is correct/authorized; low routes to review",
    )


def _create_ticket(ctx: ToolContext, args: CreateTicketArgs) -> ToolResult:
    def produce() -> ToolResult:
        ticket = Ticket(
            customer_id=ctx.customer_id,
            conversation_id=ctx.conversation_id,
            subject=args.subject,
            body=args.body,
            priority=args.priority,
            status="open",
            idempotency_key=args.idempotency_key,
        )
        ctx.session.add(ticket)
        ctx.session.flush()
        return ToolResult(
            content=json.dumps(
                {"ticket_id": ticket.id, "status": ticket.status, "priority": ticket.priority}
            ),
            data={"ticket_id": ticket.id},
        )

    return _idempotent(ctx, args.idempotency_key, "create_ticket", produce)


# --- update_ticket ---------------------------------------------------------
class UpdateTicketArgs(BaseModel):
    ticket_id: str = Field(description="Ticket id to update")
    status: str | None = Field(default=None, description="open | pending | resolved")
    priority: str | None = Field(default=None, description="low | normal | high | urgent")
    note: str | None = Field(default=None, description="Optional note appended to the body")
    idempotency_key: str | None = Field(default=None)
    confidence: float | None = Field(default=None, description="0-1 confidence")


def _update_ticket(ctx: ToolContext, args: UpdateTicketArgs) -> ToolResult:
    def produce() -> ToolResult:
        ticket = ctx.session.get(Ticket, args.ticket_id)
        if ticket is None:
            return ToolResult(content=f"No ticket {args.ticket_id}.", is_error=True)
        if args.status:
            ticket.status = args.status
        if args.priority:
            ticket.priority = args.priority
        if args.note:
            ticket.body = (ticket.body + "\n" + args.note).strip()
        ctx.session.flush()
        return ToolResult(
            content=json.dumps(
                {"ticket_id": ticket.id, "status": ticket.status, "priority": ticket.priority}
            ),
            data={"ticket_id": ticket.id},
        )

    return _idempotent(ctx, args.idempotency_key, "update_ticket", produce)


# --- escalate_to_human -----------------------------------------------------
class EscalateArgs(BaseModel):
    reason: str = Field(description="Why this needs a human")
    summary: str = Field(default="", description="Short summary of the conversation")


def _escalate(ctx: ToolContext, args: EscalateArgs) -> ToolResult:
    from app.policy.service import enqueue_review  # lazy: avoid import cycle

    review = enqueue_review(
        ctx.session,
        ctx.conversation_id,
        "escalate_to_human",
        {"reason": args.reason, "summary": args.summary},
        args.reason,
    )
    return ToolResult(
        content=f"Escalated to a human (review_id={review.id}).",
        data={"review_id": review.id},
    )


CREATE_TICKET = Tool(
    name="create_ticket",
    description="Open a support ticket for the customer.",
    args_model=CreateTicketArgs,
    side_effect=SideEffect.WRITE,
    handler=_create_ticket,
)

UPDATE_TICKET = Tool(
    name="update_ticket",
    description="Update an existing support ticket's status, priority, or notes.",
    args_model=UpdateTicketArgs,
    side_effect=SideEffect.WRITE,
    handler=_update_ticket,
)

ESCALATE_TO_HUMAN = Tool(
    name="escalate_to_human",
    description="Hand the conversation to a human agent when you cannot safely resolve it.",
    args_model=EscalateArgs,
    side_effect=SideEffect.WRITE,
    handler=_escalate,
)

WRITE_TOOLS: list[Tool] = [CREATE_TICKET, UPDATE_TICKET, ESCALATE_TO_HUMAN]
