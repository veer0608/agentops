"""Scenario model + YAML loader for the eval harness."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"


@dataclass
class SeedCustomer:
    name: str
    email: str
    tier: str = "free"


@dataclass
class SeedSubscription:
    email: str  # links to a SeedCustomer by email
    plan: str
    status: str = "active"
    renews_at: str | None = None
    mrr_cents: int = 0


@dataclass
class Scenario:
    id: str
    message: str
    expected_tools: list[str] = field(default_factory=list)
    answer_keywords: list[str] = field(default_factory=list)
    expect_escalate: bool | None = None
    customers: list[SeedCustomer] = field(default_factory=list)
    subscriptions: list[SeedSubscription] = field(default_factory=list)
    conversation_customer_email: str | None = None


def _parse(data: dict) -> Scenario:
    return Scenario(
        id=data["id"],
        message=data["message"],
        expected_tools=list(data.get("expected_tools", [])),
        answer_keywords=[str(k) for k in data.get("answer_keywords", [])],
        expect_escalate=data.get("expect_escalate"),
        customers=[SeedCustomer(**c) for c in data.get("customers", [])],
        subscriptions=[SeedSubscription(**s) for s in data.get("subscriptions", [])],
        conversation_customer_email=data.get("conversation_customer_email"),
    )


def load_scenarios() -> list[Scenario]:
    scenarios = []
    for path in sorted(SCENARIO_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        scenarios.append(_parse(data))
    return scenarios
