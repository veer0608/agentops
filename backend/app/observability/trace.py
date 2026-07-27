"""Trace helpers: record each agent step to the DB and expose a logger."""
from __future__ import annotations

import structlog
from sqlalchemy.orm import Session

from app.models import AgentStep

log = structlog.get_logger()


def record_step(
    session: Session,
    *,
    conversation_id: str,
    step_no: int,
    kind: str,
    thought: str | None = None,
    tool_name: str | None = None,
    tool_args: dict | None = None,
    tool_result: str | None = None,
    latency_ms: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_read_tokens: int | None = None,
    model: str | None = None,
) -> AgentStep:
    step = AgentStep(
        conversation_id=conversation_id,
        step_no=step_no,
        kind=kind,
        thought=thought,
        tool_name=tool_name,
        tool_args=tool_args,
        tool_result=tool_result,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        model=model,
    )
    session.add(step)
    session.flush()
    return step
