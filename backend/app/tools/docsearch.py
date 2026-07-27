"""Doc search behind an interface so the agent doesn't hard-depend on CiteRAG.

- StubDocSearch: deterministic offline corpus (used by tests/evals).
- CiteRagDocSearch: HTTP POST to the CiteRAG /query service next door; uses the
  returned `chunks` + scores as grounding (contract from
  citerag/backend/app/routers/query.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx

from app.config import settings


@dataclass
class DocPassage:
    doc_id: str
    title: str
    content: str
    score: float


class DocSearch(Protocol):
    def search(self, query: str, top_k: int) -> list[DocPassage]: ...


class StubDocSearch:
    """Token-overlap retrieval over a tiny fixed corpus — no network, deterministic."""

    CORPUS: list[dict] = [
        {
            "id": "DOC-101",
            "title": "Resetting your password",
            "content": (
                "To reset your password, open Settings > Security and click Reset "
                "Password. A reset link is emailed to you and expires in 30 minutes."
            ),
        },
        {
            "id": "DOC-102",
            "title": "Refund policy",
            "content": (
                "Refunds are available within 14 days of purchase for annual plans. "
                "Monthly plans are non-refundable. Contact support to request a refund."
            ),
        },
        {
            "id": "DOC-103",
            "title": "Plan limits",
            "content": (
                "Free plans include 1 seat and 1,000 API calls per month. Pro includes "
                "5 seats and 50,000 calls. Enterprise limits are custom."
            ),
        },
        {
            "id": "DOC-104",
            "title": "API rate limits",
            "content": (
                "The API allows 60 requests per minute on Pro and 600 per minute on "
                "Enterprise. Exceeding the limit returns HTTP 429."
            ),
        },
        {
            "id": "DOC-105",
            "title": "Canceling your subscription",
            "content": (
                "You can cancel anytime from Billing > Subscription. Access continues "
                "until the end of the current billing period."
            ),
        },
    ]

    def search(self, query: str, top_k: int) -> list[DocPassage]:
        terms = {t for t in query.lower().split() if len(t) > 2}
        scored: list[tuple[int, dict]] = []
        for doc in self.CORPUS:
            haystack = (doc["title"] + " " + doc["content"]).lower()
            overlap = sum(1 for t in terms if t in haystack)
            if overlap:
                scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            DocPassage(doc_id=d["id"], title=d["title"], content=d["content"], score=float(s))
            for s, d in scored[: max(top_k, 0)]
        ]


class CiteRagDocSearch:
    """Wraps CiteRAG's /query endpoint; we use its retrieved chunks as grounding."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def search(self, query: str, top_k: int) -> list[DocPassage]:
        resp = httpx.post(
            f"{self.base_url}/query",
            json={"question": query, "top_k": top_k},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            DocPassage(
                doc_id=str(c["document_id"]),
                title=str(c.get("document_id", "")),
                content=c["content"],
                score=float(c.get("score", 0.0)),
            )
            for c in data.get("chunks", [])
        ]


def make_docsearch() -> DocSearch:
    if settings.doc_search_mode == "citerag":
        return CiteRagDocSearch(settings.citerag_url)
    return StubDocSearch()
