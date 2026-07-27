"""Tool abstraction: a validated, side-effect-classified callable the agent can use.

The `side_effect` class (read/write) is what the Phase 2 policy gate keys on.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.tools.docsearch import DocSearch


class SideEffect(str, Enum):
    READ = "read"
    WRITE = "write"


@dataclass
class ToolContext:
    """Everything a tool handler may need, injected per run."""

    session: Session
    docsearch: DocSearch
    customer_id: str | None = None
    conversation_id: str | None = None


@dataclass
class ToolResult:
    content: str  # text returned to the model
    is_error: bool = False
    data: dict | None = None  # structured payload (grounding / citations)


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    side_effect: SideEffect
    handler: Callable[["ToolContext", BaseModel], "ToolResult"]

    def input_schema(self) -> dict[str, Any]:
        return _clean_schema(self.args_model.model_json_schema())

    def anthropic_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema(),
        }


def _clean_schema(schema: dict) -> dict:
    """Strip pydantic 'title' noise; ensure a top-level object type for the API."""
    schema = dict(schema)
    schema.pop("title", None)
    schema.setdefault("type", "object")
    props = schema.get("properties")
    if isinstance(props, dict):
        for prop in props.values():
            if isinstance(prop, dict):
                prop.pop("title", None)
    return schema
