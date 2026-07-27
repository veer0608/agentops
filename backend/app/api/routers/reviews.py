"""Human review queue: list pending actions, approve (executes) or reject them."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_session
from app.policy import approve_review, list_pending, reject_review
from app.tools import build_registry, make_docsearch

router = APIRouter(prefix="/reviews", tags=["reviews"])

_registry = build_registry()
_docsearch = make_docsearch()


class ReviewOut(BaseModel):
    id: str
    conversation_id: str | None
    proposed_action: dict | None
    reason: str | None
    status: str


class ResolveBody(BaseModel):
    note: str | None = None


class ApproveResult(BaseModel):
    id: str
    status: str
    result: str | None


@router.get("", response_model=list[ReviewOut])
def get_reviews(session: Session = Depends(get_session)) -> list[ReviewOut]:
    return [
        ReviewOut(
            id=r.id,
            conversation_id=r.conversation_id,
            proposed_action=r.proposed_action,
            reason=r.reason,
            status=r.status,
        )
        for r in list_pending(session)
    ]


@router.post("/{review_id}/approve", response_model=ApproveResult)
def approve(
    review_id: str,
    body: ResolveBody | None = None,
    session: Session = Depends(get_session),
) -> ApproveResult:
    note = body.note if body else None
    review, result = approve_review(session, review_id, _registry, _docsearch, note=note)
    if review is None:
        raise HTTPException(status_code=404, detail="review not found")
    if result == "not_pending":
        raise HTTPException(status_code=409, detail="review already resolved")
    return ApproveResult(id=review.id, status=review.status, result=result)


@router.post("/{review_id}/reject", response_model=ReviewOut)
def reject(
    review_id: str,
    body: ResolveBody | None = None,
    session: Session = Depends(get_session),
) -> ReviewOut:
    note = body.note if body else None
    review = reject_review(session, review_id, note=note)
    if review is None:
        raise HTTPException(status_code=404, detail="review not found")
    return ReviewOut(
        id=review.id,
        conversation_id=review.conversation_id,
        proposed_action=review.proposed_action,
        reason=review.reason,
        status=review.status,
    )
