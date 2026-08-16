# AgentOps — AI Support Engineer

[![ci](https://github.com/veer0608/agentops/actions/workflows/ci.yml/badge.svg)](https://github.com/veer0608/agentops/actions/workflows/ci.yml)

An AI support **agent** (not a chatbot) that resolves customer messages by taking real
actions — looking up accounts, searching docs and known issues, opening tickets — behind
a **confidence/policy gate** that decides when to act autonomously and when to hand off to
a human. Every decision is logged, replayable, and scored by a built-in **eval harness**.

Built to demonstrate agent design *and* evaluation rigor — the `measure → diagnose →
improve` loop — not just a demo.

## Highlights

- **Acts, but gated** — read tools auto-run; writes (`create_ticket`, `update_ticket`) are
  auto-executed, routed to a human **review queue**, or blocked by rules (enterprise
  account, urgent priority, low confidence). Idempotent writes, an append-only **audit
  log**, and `escalate_to_human`.
- **Eval-first** — a scenario suite scores six dimensions: tool-selection (P/R/F1),
  task-success, escalation accuracy, citation-grounding (a hallucination proxy), cost, and
  latency. Every run is a diffable scorecard persisted to `eval_runs`.
- **Provider-agnostic** — OpenAI, Anthropic, and Google (Gemini) behind one `LLMClient`
  seam; swap with a single config value.
- **Two agent implementations** — a hand-rolled tool-use loop and a **LangGraph**
  `StateGraph`, behind the same interface, with the eval harness **proving they score
  identically**.
- **Backend-grade** — FastAPI + SQLAlchemy + Alembic (SQLite dev / Postgres via URL swap),
  structured tracing, **33 tests**.

## Offline baseline (10 scenarios, deterministic demo model)

| Tool-sel. F1 | Task-success | Escalation acc. | Citation grounding |
|---|---|---|---|
| 0.60 | 0.50 | 0.90 | 1.00 |

Every failure is a tool-selection / escalation routing decision (account lookups,
known-issue lookups, one missed escalation) — not retrieval or hallucination (grounding is
1.0). That diagnosis, and how a real model plus the policy gate close it, is in the
[error-analysis writeup](EVAL_WRITEUP.md).

## Docs

- **[Architecture](ARCHITECTURE.md)** — components, data flow, design decisions
- **[Eval writeup](EVAL_WRITEUP.md)** — metrics, baseline, failure analysis, the "X%" story
- **[Backend README](backend/README.md)** — run instructions, config, endpoints

## Quickstart

```bash
cd backend
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows; use .venv/bin on POSIX
.venv/Scripts/python -m alembic upgrade head
.venv/Scripts/python -m app.seed
.venv/Scripts/python -m pytest                            # 33 tests
.venv/Scripts/python -m evals.runner                      # scorecard
.venv/Scripts/python -m uvicorn app.api.main:app --port 8011   # then open /docs
```

Runs against a free, deterministic offline demo model by default; set a funded provider key
in `backend/.env` (`PROVIDER` = `openai` | `anthropic` | `google`) for a live run and the
same eval produces the live scorecard. Full details in the
[backend README](backend/README.md).

## Status

Phases 0–4 complete: agent spine → eval harness → policy gate + review/audit → cost/latency
& grounding metrics + `search_github_issues` → LangGraph parity.

## License

MIT — see [LICENSE](LICENSE).
