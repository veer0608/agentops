"""Phase 0 read tools: search_docs, get_customer, get_subscription."""
from __future__ import annotations

import json

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.models import Customer, Subscription
from app.tools.base import SideEffect, Tool, ToolContext, ToolResult


# --- search_docs -----------------------------------------------------------
class SearchDocsArgs(BaseModel):
    query: str = Field(description="What to search the help center for")
    top_k: int = Field(default=5, description="Maximum number of passages to return")


def _search_docs(ctx: ToolContext, args: SearchDocsArgs) -> ToolResult:
    passages = ctx.docsearch.search(args.query, args.top_k)
    if not passages:
        return ToolResult(content="No matching help-center articles found.")
    lines = [f"[{p.doc_id}] {p.title}: {p.content}" for p in passages]
    return ToolResult(
        content="\n".join(lines),
        data={"passages": [p.__dict__ for p in passages]},
    )


# --- get_customer ----------------------------------------------------------
class GetCustomerArgs(BaseModel):
    customer_id: str | None = Field(default=None, description="Customer id")
    email: str | None = Field(default=None, description="Customer email")


def _get_customer(ctx: ToolContext, args: GetCustomerArgs) -> ToolResult:
    if not args.customer_id and not args.email:
        return ToolResult(content="Provide customer_id or email.", is_error=True)
    stmt = select(Customer)
    if args.customer_id:
        stmt = stmt.where(Customer.id == args.customer_id)
    else:
        stmt = stmt.where(Customer.email == args.email)
    customer = ctx.session.scalars(stmt).first()
    if customer is None:
        return ToolResult(content="No customer found.")
    payload = {
        "id": customer.id,
        "name": customer.name,
        "email": customer.email,
        "tier": customer.tier,
    }
    return ToolResult(content=json.dumps(payload), data=payload)


# --- get_subscription ------------------------------------------------------
class GetSubscriptionArgs(BaseModel):
    customer_id: str = Field(description="Customer id to look up the subscription for")


def _get_subscription(ctx: ToolContext, args: GetSubscriptionArgs) -> ToolResult:
    stmt = select(Subscription).where(Subscription.customer_id == args.customer_id)
    sub = ctx.session.scalars(stmt).first()
    if sub is None:
        return ToolResult(content="No subscription found for that customer.")
    payload = {
        "plan": sub.plan,
        "status": sub.status,
        "renews_at": sub.renews_at,
        "mrr_cents": sub.mrr_cents,
    }
    return ToolResult(content=json.dumps(payload), data=payload)


SEARCH_DOCS = Tool(
    name="search_docs",
    description=(
        "Search the help center for relevant articles. Returns passages prefixed "
        "with [doc_id] markers — cite those markers in your answer."
    ),
    args_model=SearchDocsArgs,
    side_effect=SideEffect.READ,
    handler=_search_docs,
)

GET_CUSTOMER = Tool(
    name="get_customer",
    description="Look up a customer account by id or email.",
    args_model=GetCustomerArgs,
    side_effect=SideEffect.READ,
    handler=_get_customer,
)

GET_SUBSCRIPTION = Tool(
    name="get_subscription",
    description="Get the plan, status, and billing details for a customer id.",
    args_model=GetSubscriptionArgs,
    side_effect=SideEffect.READ,
    handler=_get_subscription,
)


# --- search_github_issues --------------------------------------------------
_GITHUB_ISSUES: list[dict] = [
    {"id": "GH-101", "title": "Dashboard shows 500 error on load", "state": "open"},
    {"id": "GH-102", "title": "API returns 429 under normal load", "state": "open"},
    {"id": "GH-103", "title": "Password reset email sometimes not received", "state": "closed"},
    {"id": "GH-104", "title": "Billing page slow for enterprise accounts", "state": "open"},
]


class SearchGithubIssuesArgs(BaseModel):
    query: str = Field(description="Keywords to search known engineering issues")
    state: str = Field(default="open", description="open | closed | all")


def _search_github_issues(ctx: ToolContext, args: SearchGithubIssuesArgs) -> ToolResult:
    terms = {t for t in args.query.lower().split() if len(t) > 2}
    matches = [
        iss
        for iss in _GITHUB_ISSUES
        if (args.state == "all" or iss["state"] == args.state)
        and any(t in iss["title"].lower() for t in terms)
    ]
    if not matches:
        return ToolResult(content="No matching known issues found.")
    lines = [f"[{i['id']}] ({i['state']}) {i['title']}" for i in matches]
    return ToolResult(content="\n".join(lines), data={"issues": matches})


SEARCH_GITHUB_ISSUES = Tool(
    name="search_github_issues",
    description=(
        "Search known engineering issues (GitHub) for a bug the customer may be "
        "hitting. Results are prefixed with [GH-###] markers — cite them."
    ),
    args_model=SearchGithubIssuesArgs,
    side_effect=SideEffect.READ,
    handler=_search_github_issues,
)

READ_TOOLS: list[Tool] = [SEARCH_DOCS, GET_CUSTOMER, GET_SUBSCRIPTION, SEARCH_GITHUB_ISSUES]
