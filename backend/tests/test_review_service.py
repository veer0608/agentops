"""Review queue: approve executes the queued action; reject does not."""
from __future__ import annotations

from sqlalchemy import func, select

from app.models import Conversation, Ticket
from app.policy import approve_review, enqueue_review, list_pending, reject_review
from app.tools import StubDocSearch, build_registry


def _conv(db_session, customer_id):
    conv = Conversation(customer_id=customer_id)
    db_session.add(conv)
    db_session.flush()
    return conv


def test_approve_executes_queued_action(db_session, seeded_customer_id):
    conv = _conv(db_session, seeded_customer_id)
    rq = enqueue_review(
        db_session, conv.id, "create_ticket", {"subject": "Refund help", "priority": "normal"}, "review"
    )
    db_session.commit()
    assert len(list_pending(db_session)) == 1

    review, _ = approve_review(db_session, rq.id, build_registry(), StubDocSearch(), note="ok")
    assert review.status == "approved"
    assert db_session.scalar(select(func.count()).select_from(Ticket)) == 1
    assert len(list_pending(db_session)) == 0


def test_reject_does_not_execute(db_session, seeded_customer_id):
    conv = _conv(db_session, seeded_customer_id)
    rq = enqueue_review(db_session, conv.id, "create_ticket", {"subject": "x"}, "review")
    db_session.commit()

    review = reject_review(db_session, rq.id, note="no")
    assert review.status == "rejected"
    assert db_session.scalar(select(func.count()).select_from(Ticket)) == 0
