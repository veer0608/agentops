"""Unit tests for the read tools."""
from __future__ import annotations

from app.tools import StubDocSearch, ToolContext
from app.tools.builtin import (
    GET_CUSTOMER,
    GET_SUBSCRIPTION,
    SEARCH_DOCS,
    GetCustomerArgs,
    GetSubscriptionArgs,
    SearchDocsArgs,
)


def test_get_customer_by_id(db_session, seeded_customer_id):
    ctx = ToolContext(session=db_session, docsearch=StubDocSearch(), customer_id=seeded_customer_id)
    result = GET_CUSTOMER.handler(ctx, GetCustomerArgs(customer_id=seeded_customer_id))
    assert not result.is_error
    assert "ada@example.com" in result.content


def test_get_customer_by_email(db_session, seeded_customer_id):
    ctx = ToolContext(session=db_session, docsearch=StubDocSearch())
    result = GET_CUSTOMER.handler(ctx, GetCustomerArgs(email="ada@example.com"))
    assert result.data is not None
    assert result.data["tier"] == "pro"


def test_get_subscription(db_session, seeded_customer_id):
    ctx = ToolContext(session=db_session, docsearch=StubDocSearch(), customer_id=seeded_customer_id)
    result = GET_SUBSCRIPTION.handler(ctx, GetSubscriptionArgs(customer_id=seeded_customer_id))
    assert "Pro" in result.content


def test_search_docs_finds_password_article():
    ctx = ToolContext(session=None, docsearch=StubDocSearch())  # type: ignore[arg-type]
    result = SEARCH_DOCS.handler(ctx, SearchDocsArgs(query="how do I reset my password", top_k=3))
    assert "DOC-101" in result.content
