"""Run the eval suite: per-scenario fresh DB -> run agent -> score -> scorecard.

Usage: `python -m evals.runner`
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.agent import AgentAnswer, build_llm, build_runner
from app.config import settings
from app.db import Base
from app.models import AgentStep, Conversation, Customer, Message, Subscription
from app.tools import build_registry, make_docsearch
from evals.pricing import cost_usd
from evals.scenarios import Scenario, load_scenarios
from evals.scorers import (
    extract_markers,
    score_citation_grounding,
    score_escalation_accuracy,
    score_task_success,
    score_tool_selection,
)

REPORT_DIR = Path(__file__).resolve().parent / "reports"
_RETRIEVAL_TOOLS = {"search_docs", "search_github_issues"}


def _setup_scenario_db(scenario: Scenario) -> tuple[Session, Conversation]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, future=True)()

    email_to_id: dict[str, str] = {}
    for c in scenario.customers:
        customer = Customer(name=c.name, email=c.email, tier=c.tier)
        session.add(customer)
        session.flush()
        email_to_id[c.email] = customer.id
    for s in scenario.subscriptions:
        session.add(
            Subscription(
                customer_id=email_to_id[s.email],
                plan=s.plan,
                status=s.status,
                renews_at=s.renews_at,
                mrr_cents=s.mrr_cents,
            )
        )

    customer_id = (
        email_to_id.get(scenario.conversation_customer_email)
        if scenario.conversation_customer_email
        else None
    )
    conv = Conversation(customer_id=customer_id)
    session.add(conv)
    session.flush()
    session.add(Message(conversation_id=conv.id, role="customer", content=scenario.message))
    session.commit()
    return session, conv


def _scenario_metrics(session: Session, conversation_id: str) -> dict:
    """Pull tool usage, retrieved markers, latency, and tokens from the trace."""
    steps = session.scalars(
        select(AgentStep)
        .where(AgentStep.conversation_id == conversation_id)
        .order_by(AgentStep.step_no)
    ).all()
    tools: list[str] = []
    retrieved: set[str] = set()
    for s in steps:
        if s.kind == "tool" and s.tool_name:
            if s.tool_name not in tools:
                tools.append(s.tool_name)
            if s.tool_name in _RETRIEVAL_TOOLS and s.tool_result:
                retrieved |= extract_markers(s.tool_result)
    model = next((s.model for s in steps if s.model), "")
    return {
        "tools": tools,
        "retrieved": retrieved,
        "latency_ms": sum(s.latency_ms or 0 for s in steps),
        "input_tokens": sum(s.input_tokens or 0 for s in steps),
        "output_tokens": sum(s.output_tokens or 0 for s in steps),
        "model": model,
    }


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo))


def run_suite(
    *, llm=None, runner_kind: str | None = None, write_reports: bool = True, persist: bool = True
) -> dict:
    scenarios = load_scenarios()
    llm = llm or build_llm(settings)
    kind = runner_kind or settings.runner
    runner = build_runner(
        llm, build_registry(), make_docsearch(), max_steps=settings.max_agent_steps, kind=kind
    )

    rows: list[dict] = []
    for sc in scenarios:
        session, conv = _setup_scenario_db(sc)
        try:
            answer = runner.run(session, conv)
            session.commit()
            m = _scenario_metrics(session, conv.id)
        except Exception as exc:  # a live API hiccup shouldn't kill the whole suite
            answer = AgentAnswer(
                answer=f"[run error: {exc}]",
                should_escalate=True,
                escalation_reason="run_error",
            )
            m = {
                "tools": [],
                "retrieved": set(),
                "latency_ms": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "model": getattr(llm, "model", ""),
            }
        finally:
            session.close()

        ts = score_tool_selection(sc.expected_tools, m["tools"])
        success = score_task_success(
            expect_escalate=sc.expect_escalate,
            answer_keywords=sc.answer_keywords,
            answer_text=answer.answer,
            did_escalate=answer.should_escalate,
        )
        grounded = score_citation_grounding(answer.citations, m["retrieved"])
        cost = cost_usd(m["model"], m["input_tokens"], m["output_tokens"])
        rows.append(
            {
                "id": sc.id,
                "expected_tools": sc.expected_tools,
                "actual_tools": m["tools"],
                "tool_precision": round(ts.precision, 3),
                "tool_recall": round(ts.recall, 3),
                "tool_f1": round(ts.f1, 3),
                "tool_exact": ts.exact,
                "expect_escalate": sc.expect_escalate,
                "did_escalate": answer.should_escalate,
                "task_success": success,
                "citation_grounded": grounded,
                "latency_ms": m["latency_ms"],
                "cost_usd": cost,
                "answer": answer.answer,
            }
        )

    n = len(rows) or 1
    labeled = [
        (r["expect_escalate"], r["did_escalate"])
        for r in rows
        if r["expect_escalate"] is not None
    ]
    latencies = [float(r["latency_ms"]) for r in rows]
    summary = {
        "n": len(rows),
        "model": getattr(llm, "model", "demo(offline)"),
        "runner": kind,
        "doc_search_mode": settings.doc_search_mode,
        "tool_selection_f1": round(sum(r["tool_f1"] for r in rows) / n, 3),
        "tool_exact_match_rate": round(sum(1 for r in rows if r["tool_exact"]) / n, 3),
        "task_success_rate": round(sum(1 for r in rows if r["task_success"]) / n, 3),
        "escalation_accuracy": round(score_escalation_accuracy(labeled), 3),
        "citation_grounding": round(sum(1 for r in rows if r["citation_grounded"]) / n, 3),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
        "latency_p50_ms": round(_pct(latencies, 0.5), 1),
        "latency_p95_ms": round(_pct(latencies, 0.95), 1),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    report_path: str | None = None
    if write_reports:
        report_path = _write_reports(summary, rows)
    if persist:
        _persist(summary, report_path)

    _print_summary(summary, rows)
    return {"summary": summary, "rows": rows}


def _render_markdown(summary: dict, rows: list[dict]) -> str:
    lines = [
        "# AgentOps eval scorecard",
        "",
        f"- Model: `{summary['model']}`",
        f"- Runner: `{summary['runner']}`",
        f"- Doc search: `{summary['doc_search_mode']}`",
        f"- Scenarios: {summary['n']}",
        f"- Tool-selection F1: **{summary['tool_selection_f1']}**",
        f"- Tool exact-match rate: **{summary['tool_exact_match_rate']}**",
        f"- Task-success rate: **{summary['task_success_rate']}**",
        f"- Escalation accuracy: **{summary['escalation_accuracy']}**",
        f"- Citation grounding: **{summary['citation_grounding']}**",
        f"- Total cost: **${summary['total_cost_usd']}**",
        f"- Latency p50/p95: **{summary['latency_p50_ms']} / {summary['latency_p95_ms']} ms**",
        f"- Run at: {summary['created_at']}",
        "",
        "| scenario | expected tools | actual tools | tool F1 | task | grounded |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {', '.join(r['expected_tools']) or '—'} | "
            f"{', '.join(r['actual_tools']) or '—'} | {r['tool_f1']:.2f} | "
            f"{'PASS' if r['task_success'] else 'FAIL'} | "
            f"{'ok' if r['citation_grounded'] else 'HALLUC'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_reports(summary: dict, rows: list[dict]) -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "scorecard-latest.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )
    md_path = REPORT_DIR / "scorecard-latest.md"
    md_path.write_text(_render_markdown(summary, rows), encoding="utf-8")
    return str(md_path)


def _persist(summary: dict, report_path: str | None) -> None:
    from app.db import SessionLocal
    from app.models import EvalRun

    with SessionLocal() as session:
        session.add(
            EvalRun(
                model=summary["model"],
                doc_search_mode=summary["doc_search_mode"],
                n_scenarios=summary["n"],
                tool_selection_f1=summary["tool_selection_f1"],
                tool_exact_match_rate=summary["tool_exact_match_rate"],
                task_success_rate=summary["task_success_rate"],
                escalation_accuracy=summary["escalation_accuracy"],
                citation_grounding=summary["citation_grounding"],
                total_cost_usd=summary["total_cost_usd"],
                latency_p50_ms=summary["latency_p50_ms"],
                report_path=report_path,
            )
        )
        session.commit()


def _print_summary(summary: dict, rows: list[dict]) -> None:
    print("=" * 60)
    print(f"AgentOps eval - model={summary['model']} runner={summary['runner']} docs={summary['doc_search_mode']}")
    print(f"  scenarios          : {summary['n']}")
    print(f"  tool-selection F1  : {summary['tool_selection_f1']}")
    print(f"  tool exact-match   : {summary['tool_exact_match_rate']}")
    print(f"  task-success rate  : {summary['task_success_rate']}")
    print(f"  escalation accuracy: {summary['escalation_accuracy']}")
    print(f"  citation grounding : {summary['citation_grounding']}")
    print(f"  total cost (USD)   : {summary['total_cost_usd']}")
    print(f"  latency p50/p95 ms : {summary['latency_p50_ms']} / {summary['latency_p95_ms']}")
    print("-" * 60)
    for r in rows:
        flag = "PASS" if r["task_success"] else "FAIL"
        print(f"  [{flag}] {r['id']:<24} f1={r['tool_f1']:.2f} tools={r['actual_tools']}")
    print("=" * 60)


if __name__ == "__main__":
    run_suite()
