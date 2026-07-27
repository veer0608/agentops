"""HTTP path for the review queue (offline)."""
from __future__ import annotations

from app.models import Conversation
from app.policy import enqueue_review


def test_reviews_list_and_approve(client, db_session, seeded_customer_id):
    conv = Conversation(customer_id=seeded_customer_id)
    db_session.add(conv)
    db_session.flush()
    rq = enqueue_review(
        db_session, conv.id, "create_ticket", {"subject": "API review", "priority": "normal"}, "review"
    )
    db_session.commit()

    listed = client.get("/reviews")
    assert listed.status_code == 200
    assert rq.id in [r["id"] for r in listed.json()]

    approved = client.post(f"/reviews/{rq.id}/approve", json={"note": "ok"})
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"
