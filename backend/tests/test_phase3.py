"""Phase 3: pricing, citation-grounding, and the search_github_issues tool."""
from __future__ import annotations

from app.tools import StubDocSearch, ToolContext
from app.tools.builtin import SEARCH_GITHUB_ISSUES, SearchGithubIssuesArgs
from evals.pricing import cost_usd
from evals.scorers import extract_markers, score_citation_grounding


def test_cost_usd():
    assert cost_usd("demo(offline)", 1000, 1000) == 0.0
    assert cost_usd("gpt-4o", 1_000_000, 1_000_000) == round(2.50 + 10.00, 6)


def test_extract_markers_and_grounding():
    retrieved = extract_markers("see [DOC-101] and [GH-102] for details")
    assert retrieved == {"DOC-101", "GH-102"}
    assert score_citation_grounding(["DOC-101"], retrieved) is True
    assert score_citation_grounding(["DOC-999"], retrieved) is False  # hallucinated
    assert score_citation_grounding([], retrieved) is True


def test_search_github_issues():
    ctx = ToolContext(session=None, docsearch=StubDocSearch())  # type: ignore[arg-type]
    res = SEARCH_GITHUB_ISSUES.handler(
        ctx, SearchGithubIssuesArgs(query="dashboard 500 error")
    )
    assert "GH-101" in res.content
