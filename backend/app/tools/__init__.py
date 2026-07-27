"""Tool registry and public tool API."""
from __future__ import annotations

from app.tools.base import SideEffect, Tool, ToolContext, ToolResult
from app.tools.builtin import READ_TOOLS
from app.tools.docsearch import DocPassage, DocSearch, StubDocSearch, make_docsearch
from app.tools.write_tools import WRITE_TOOLS


def build_registry() -> dict[str, Tool]:
    """Name -> Tool: read tools + write tools."""
    return {tool.name: tool for tool in [*READ_TOOLS, *WRITE_TOOLS]}


__all__ = [
    "SideEffect",
    "Tool",
    "ToolContext",
    "ToolResult",
    "READ_TOOLS",
    "WRITE_TOOLS",
    "DocPassage",
    "DocSearch",
    "StubDocSearch",
    "make_docsearch",
    "build_registry",
]
