"""Write-tool behavior, including idempotency."""
from __future__ import annotations

import json

from sqlalchemy import func, select

from app.models import Ticket
from app.tools import StubDocSearch, ToolContext
from app.tools.write_tools import (
    CREATE_TICKET,
    UPDATE_TICKET,
    CreateTicketArgs,
    UpdateTicketArgs,
)


def _ctx(session, customer_id=None):
    return ToolContext(session=session, docsearch=StubDocSearch(), customer_id=customer_id)


def test_create_ticket(db_session, seeded_customer_id):
    res = CREATE_TICKET.handler(
        _ctx(db_session, seeded_customer_id),
        CreateTicketArgs(subject="Dashboard broken", body="500 errors"),
    )
    assert not res.is_error
    assert db_session.get(Ticket, res.data["ticket_id"]) is not None


def test_create_ticket_idempotent(db_session, seeded_customer_id):
    ctx = _ctx(db_session, seeded_customer_id)
    first = CREATE_TICKET.handler(ctx, CreateTicketArgs(subject="X", idempotency_key="k1"))
    second = CREATE_TICKET.handler(ctx, CreateTicketArgs(subject="X again", idempotency_key="k1"))
    assert first.data["ticket_id"] == json.loads(second.content)["ticket_id"]
    count = db_session.scalar(select(func.count()).select_from(Ticket))
    assert count == 1  # the repeat did not create a second ticket


def test_update_ticket(db_session, seeded_customer_id):
    ctx = _ctx(db_session, seeded_customer_id)
    created = CREATE_TICKET.handler(ctx, CreateTicketArgs(subject="X"))
    res = UPDATE_TICKET.handler(
        ctx, UpdateTicketArgs(ticket_id=created.data["ticket_id"], status="resolved")
    )
    assert json.loads(res.content)["status"] == "resolved"
