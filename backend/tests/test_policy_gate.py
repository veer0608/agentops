"""Policy gate decision rules."""
from __future__ import annotations

from app.models import Customer
from app.policy import PolicyAction, PolicyGate
from app.tools import StubDocSearch, ToolContext
from app.tools.builtin import SEARCH_DOCS
from app.tools.write_tools import CREATE_TICKET, ESCALATE_TO_HUMAN


def _ctx(session, customer_id=None):
    return ToolContext(session=session, docsearch=StubDocSearch(), customer_id=customer_id)


def test_read_tool_auto(db_session):
    d = PolicyGate().evaluate(SEARCH_DOCS, {"query": "x"}, _ctx(db_session))
    assert d.action == PolicyAction.AUTO


def test_escalate_always_auto(db_session):
    d = PolicyGate().evaluate(ESCALATE_TO_HUMAN, {"reason": "x"}, _ctx(db_session))
    assert d.action == PolicyAction.AUTO


def test_normal_write_auto(db_session, seeded_customer_id):  # seeded customer is "pro"
    d = PolicyGate().evaluate(
        CREATE_TICKET,
        {"subject": "x", "priority": "normal", "confidence": 0.9},
        _ctx(db_session, seeded_customer_id),
    )
    assert d.action == PolicyAction.AUTO


def test_enterprise_write_requires_review(db_session):
    cust = Customer(name="Big Co", email="big@co.com", tier="enterprise")
    db_session.add(cust)
    db_session.flush()
    d = PolicyGate().evaluate(CREATE_TICKET, {"subject": "x"}, _ctx(db_session, cust.id))
    assert d.action == PolicyAction.REVIEW


def test_urgent_priority_requires_review(db_session, seeded_customer_id):
    d = PolicyGate().evaluate(
        CREATE_TICKET, {"subject": "x", "priority": "urgent"}, _ctx(db_session, seeded_customer_id)
    )
    assert d.action == PolicyAction.REVIEW


def test_low_confidence_requires_review(db_session, seeded_customer_id):
    d = PolicyGate().evaluate(
        CREATE_TICKET, {"subject": "x", "confidence": 0.2}, _ctx(db_session, seeded_customer_id)
    )
    assert d.action == PolicyAction.REVIEW
