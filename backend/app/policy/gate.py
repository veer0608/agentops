"""The policy/confidence gate — the differentiating layer.

Deterministic, unit-testable rules over (tool, args, customer context) that decide
whether a write executes automatically, needs human review, or is blocked. Read
tools always auto-run; escalation is always allowed.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.models import Customer
from app.tools.base import SideEffect, Tool, ToolContext

DEFAULT_CONFIDENCE_THRESHOLD = 0.6
ALWAYS_ALLOW_WRITES = {"escalate_to_human"}
REVIEW_PRIORITIES = {"urgent", "p0", "critical"}


class PolicyAction(str, Enum):
    AUTO = "auto"
    REVIEW = "review"
    BLOCK = "block"


@dataclass
class PolicyDecision:
    action: PolicyAction
    reason: str


class PolicyGate:
    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        self.confidence_threshold = confidence_threshold

    def evaluate(self, tool: Tool, args: dict, ctx: ToolContext) -> PolicyDecision:
        if tool.side_effect == SideEffect.READ:
            return PolicyDecision(PolicyAction.AUTO, "read-only")
        if tool.name in ALWAYS_ALLOW_WRITES:
            return PolicyDecision(PolicyAction.AUTO, "escalation is always allowed")

        # Business rules (highest priority), then the confidence threshold.
        tier = self._customer_tier(ctx)
        if tier == "enterprise":
            return PolicyDecision(
                PolicyAction.REVIEW, "enterprise account: writes require human review"
            )
        priority = str(args.get("priority", "")).lower()
        if priority in REVIEW_PRIORITIES:
            return PolicyDecision(
                PolicyAction.REVIEW, f"{priority} priority requires human review"
            )
        confidence = args.get("confidence")
        if isinstance(confidence, (int, float)) and confidence < self.confidence_threshold:
            return PolicyDecision(
                PolicyAction.REVIEW, f"low confidence ({float(confidence):.2f})"
            )
        return PolicyDecision(PolicyAction.AUTO, "within policy")

    def _customer_tier(self, ctx: ToolContext) -> str | None:
        if not ctx.customer_id:
            return None
        customer = ctx.session.get(Customer, ctx.customer_id)
        return customer.tier if customer else None
