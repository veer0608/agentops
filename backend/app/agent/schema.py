"""The agent's structured final answer + the terminal tool derived from it."""
from __future__ import annotations

from pydantic import BaseModel, Field

FINAL_TOOL_NAME = "respond_to_customer"


class AgentAnswer(BaseModel):
    answer: str = Field(description="The reply to send to the customer")
    citations: list[str] = Field(
        default_factory=list, description="doc_id markers (e.g. DOC-101) backing the answer"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Calibrated confidence, 0-1"
    )
    should_escalate: bool = Field(
        default=False, description="True if a human should handle this instead"
    )
    escalation_reason: str | None = Field(default=None)


def final_tool_schema() -> dict:
    """Anthropic tool schema for the terminal 'respond_to_customer' action."""
    schema = AgentAnswer.model_json_schema()
    schema.pop("title", None)
    schema["type"] = "object"
    for prop in schema.get("properties", {}).values():
        if isinstance(prop, dict):
            prop.pop("title", None)
    return {
        "name": FINAL_TOOL_NAME,
        "description": (
            "Send the final structured answer to the customer. Call this exactly "
            "once, when you have gathered enough information to respond."
        ),
        "input_schema": schema,
    }
