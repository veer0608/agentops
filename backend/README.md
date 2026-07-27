# AgentOps — AI Support Engineer (backend)

An AI support **agent** that answers customer messages and takes actions
(account lookups, doc search, tickets) behind a confidence/policy gate, with an
**eval harness** that scores every run. Architecture: [../ARCHITECTURE.md](../ARCHITECTURE.md).

Built phase by phase (see the build plan). **Phases 0–3** are done — the agent
spine, the eval harness, the write tools + policy/escalation gate, and the Phase 3
metrics (cost/latency, citation-grounding), the `search_github_issues` tool, and the
[error-analysis writeup](../EVAL_WRITEUP.md). **Phase 4** adds a second agent
implementation on **LangGraph** behind the same interface, with the eval harness
proving the two score identically.

## Stack
FastAPI · SQLAlchemy + Alembic (SQLite dev / Postgres later) · **OpenAI, Anthropic,
or Google (Gemini)** behind a swappable `LLMClient` seam (hand-rolled or LangGraph loop) · pytest.
`search_docs` wraps the sibling **CiteRAG** service behind an interface (offline stub).

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows; use .venv/bin on POSIX

# create the schema + demo data
.venv/Scripts/python -m alembic upgrade head
.venv/Scripts/python -m app.seed

# run the API (offline demo model until a key is set)
.venv/Scripts/python -m uvicorn app.api.main:app --port 8010

# tests
.venv/Scripts/python -m pytest
```

Try it:
```bash
curl -X POST localhost:8010/conversations -H "content-type: application/json" -d "{}"
curl -X POST localhost:8010/conversations/<id>/messages -H "content-type: application/json" \
  -d "{\"content\":\"How do I reset my password?\"}"
```

### Live model
The provider is a config switch (`PROVIDER=openai` default, or `anthropic`). Set the
matching key in `.env` and the agent uses that provider instead of the offline demo:

- `PROVIDER=openai` + `OPENAI_API_KEY=...` → `OPENAI_MODEL` (default `gpt-4o`)
- `PROVIDER=anthropic` + `ANTHROPIC_API_KEY=...` → `ANTHROPIC_MODEL` (default `claude-opus-5`)
- `PROVIDER=google` + `GOOGLE_API_KEY=...` → `GOOGLE_MODEL` (default `gemini-2.0-flash`, via Google's OpenAI-compatible endpoint)

`DOC_SEARCH_MODE` (`stub` | `citerag`) is also configurable — see `.env.example`.
Because the harness is provider-agnostic, the same scenarios benchmark either model.

### Database
SQLite by default (`data/agentops.db`, managed by Alembic). For Postgres, set
`DATABASE_URL=postgresql+psycopg://user:pass@host/agentops` and
`pip install "psycopg[binary]"` — a connection-string swap, no model changes.

### Runners
Two interchangeable agent implementations behind one `AgentRunner` interface: the
hand-rolled loop (`manual`, default) and a LangGraph `StateGraph` (`langgraph`).
Switch with `RUNNER=langgraph`. Both reuse the same tools, policy gate, and tracing;
the eval harness proves they score identically (`tests/test_runner_parity.py`).

## Actions & policy gate

Write tools (`create_ticket`, `update_ticket`, `escalate_to_human`) are idempotent and
run through a **policy gate** before executing: reads auto-run; writes are auto-executed,
routed to a **human review queue**, or blocked, based on rules (enterprise account,
urgent priority, low confidence). Every decision and action lands in an append-only
`audit_log`. Review endpoints: `GET /reviews`, `POST /reviews/{id}/approve` (executes the
queued action as a human), `POST /reviews/{id}/reject`.

## Evals

```bash
.venv/Scripts/python -m evals.runner   # runs all scenarios, writes evals/reports/scorecard-latest.md
```

Scenarios live in `evals/scenarios/*.yaml`. Metrics: tool-selection (precision/
recall/F1 + exact-set match) and task-success (escalation + keyword rubric). Each
run is persisted to the `eval_runs` table.

### Baseline (10 scenarios, offline demo model)

| Model | Tool-sel. F1 | Task-success | Escalation acc. | Grounding |
|---|---|---|---|---|
| demo (offline) | 0.60 | 0.50 | 0.90 | 1.00 |
| `gpt-4o` | _set `OPENAI_API_KEY` and re-run_ | _—_ | _—_ | _—_ |
| `claude-opus-5` | _set `PROVIDER=anthropic` + `ANTHROPIC_API_KEY`_ | _—_ | _—_ | _—_ |

All 5 failures are tool-selection / escalation routing (account lookups, known-issue
lookups, one missed escalation) — not retrieval or hallucination (grounding is 1.0).
Cost and p50/p95 latency are ~0 offline and populate on live runs. Full analysis:
[EVAL_WRITEUP.md](../EVAL_WRITEUP.md).
