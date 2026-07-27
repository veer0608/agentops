"""Review-queue + audit-log operations. Kept free of tool imports (registry is
passed in) so it doesn't create an import cycle with app.tools."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AuditLog, Conversation, ReviewQueue


def _now() -> datetime:
    return datetime.now(timezone.utc)


def audit(
    session: Session,
    *,
    conversation_id: str | None,
    actor: str,
    action: str,
    decision: str | None = None,
    reason: str | None = None,
    payload: dict | None = None,
    target: str | None = None,
) -> None:
    session.add(
        AuditLog(
            conversation_id=conversation_id,
            actor=actor,
            action=action,
            decision=decision,
            reason=reason,
            payload=payload,
            target=target,
        )
    )
    session.flush()


def enqueue_review(
    session: Session,
    conversation_id: str | None,
    tool_name: str,
    args: dict,
    reason: str,
) -> ReviewQueue:
    review = ReviewQueue(
        conversation_id=conversation_id,
        proposed_action={"tool": tool_name, "args": args},
        reason=reason,
        status="pending",
    )
    session.add(review)
    session.flush()
    return review


def list_pending(session: Session) -> list[ReviewQueue]:
    return list(
        session.scalars(
            select(ReviewQueue)
            .where(ReviewQueue.status == "pending")
            .order_by(ReviewQueue.created_at)
        )
    )


def approve_review(
    session: Session,
    review_id: str,
    registry: dict,
    docsearch,
    note: str | None = None,
) -> tuple[ReviewQueue | None, str | None]:
    """Execute the queued action as a human (bypassing the gate), mark approved."""
    from app.tools import ToolContext  # lazy: avoid import cycle

    review = session.get(ReviewQueue, review_id)
    if review is None:
        return None, "not_found"
    if review.status != "pending":
        return review, "not_pending"

    action = review.proposed_action or {}
    tool = registry.get(action.get("tool"))
    conv = (
        session.get(Conversation, review.conversation_id)
        if review.conversation_id
        else None
    )
    ctx = ToolContext(
        session=session,
        docsearch=docsearch,
        conversation_id=review.conversation_id,
        customer_id=conv.customer_id if conv else None,
    )

    result_content: str | None = None
    if tool is not None:
        args = tool.args_model(**action.get("args", {}))
        result = tool.handler(ctx, args)
        result_content = result.content

    review.status = "approved"
    review.resolved_at = _now()
    review.resolution_note = note
    audit(
        session,
        conversation_id=review.conversation_id,
        actor="human",
        action=action.get("tool", "?"),
        decision="executed",
        reason=note,
        payload=action.get("args"),
    )
    session.commit()
    return review, result_content


def reject_review(
    session: Session, review_id: str, note: str | None = None
) -> ReviewQueue | None:
    review = session.get(ReviewQueue, review_id)
    if review is None:
        return None
    review.status = "rejected"
    review.resolved_at = _now()
    review.resolution_note = note
    audit(
        session,
        conversation_id=review.conversation_id,
        actor="human",
        action=(review.proposed_action or {}).get("tool", "?"),
        decision="rejected",
        reason=note,
    )
    session.commit()
    return review
