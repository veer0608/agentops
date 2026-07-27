# AgentOps — Eval writeup (measure → diagnose → improve)

The point of AgentOps isn't the agent; it's the **loop that makes the agent better**.
This is the writeup that loop produces. It is regenerated from the eval harness
(`backend/evals/`), which runs every scenario against the configured model, scores
the resulting trace, and writes `backend/evals/reports/scorecard-latest.md`.

## What we measure

| Metric | Definition |
|---|---|
| **Tool-selection F1** | Set-based F1 of the tools the agent actually called vs. the scenario's expected tools (+ exact-set match). |
| **Task-success** | Did it reach the right outcome? Deterministic: required escalation matches **and** all answer keywords present. |
| **Escalation accuracy** | Accuracy of the escalate / don't-escalate decision over labeled scenarios. |
| **Citation grounding** | Fraction of answers whose every cited `[DOC-###]` / `[GH-###]` marker was actually retrieved this run — a deterministic **hallucination proxy** (an LLM judge slots in when a key is present). |
| **Cost / latency** | Per-run USD (from token usage × a price table) and p50/p95 latency, both read from the `agent_steps` trace. |

## Baseline (10 scenarios, offline demo model)

The offline demo model is a fixed heuristic — *always `search_docs`, then answer* —
run with no API key so the harness itself is verifiable end to end.

| Tool-sel. F1 | Task-success | Escalation acc. | Citation grounding | Cost | Latency p50/p95 |
|---|---|---|---|---|---|
| 0.60 | 0.50 | 0.90 | 1.00 | $0.00 | ~0 / ~0 ms |

## Diagnosis — where it fails, and why

5 of 10 scenarios fail. Grouping the failures is the whole game:

| Failing scenario | Expected | What the demo did | Failure class |
|---|---|---|---|
| `whats-my-plan` | `get_customer`, `get_subscription` | `search_docs` | tool selection (account) |
| `subscription-status` | `get_customer`, `get_subscription` | `search_docs` | tool selection (account) |
| `known-issue-dashboard` | `search_github_issues` | `search_docs` | tool selection (known-issue) |
| `known-issue-rate-limit` | `search_github_issues` | `search_docs` | tool selection (known-issue) |
| `refund-request-action` | escalate | answered instead | escalation |

**100% of failures are routing decisions** — picking the wrong tool, or not choosing
to escalate. Critically, they are **not** retrieval or hallucination failures:
citation grounding is **1.0**, so the demo never fabricates a source; it just answers
the wrong kind of question with the only tool it knows. That separation is the
diagnosis: the gap is *tool selection and escalation*, not knowledge or grounding.

## Improvement hypotheses (what to change, and what should move)

1. **A real model** (`gpt-4o` or `claude-opus-5`) that actually selects
   `get_customer` / `get_subscription` / `search_github_issues` → should lift
   tool-selection F1 and recover the 4 routing failures.
2. **The policy gate + escalation guidance** (Phase 2) → the agent withholds the
   refund action and escalates → escalation accuracy → ~1.0.

Each hypothesis maps to a specific failure class above, so the next scorecard tells
us directly whether the change worked. That's the loop.

## The story (fill in from a live run)

> "My support agent resolved **50%** of scenarios on the baseline. Error analysis
> showed **every** failure was a tool-selection or escalation decision — not
> retrieval or hallucination (grounding was 1.0). I switched to a real model and
> enforced a policy gate for risky actions, which targeted exactly those failure
> modes; task-success rose to **X%**, with cost and p95 latency tracked per run."

Run it live to fill in **X%**:

```bash
printf 'PROVIDER=openai\nOPENAI_API_KEY=sk-...\n' >> backend/.env
backend/.venv/Scripts/python -m evals.runner
```

The harness is provider-agnostic, so the same command with `PROVIDER=anthropic`
benchmarks `claude-opus-5` on identical scenarios — a direct gpt-4o vs. Claude
comparison on tool-selection, task-success, cost, and latency.

## Honest limitations

- **10 scenarios**, not 50–100 yet. Scaling is continued authoring (same discipline
  as CiteRAG's golden set); the harness and scorers don't change.
- **Grounding is a deterministic proxy.** It catches fabricated citations, not subtly
  wrong claims grounded in the wrong passage — that needs the LLM judge (Phase 3.5).
- **Offline latency/cost are ~0** because the demo model does no network I/O; these
  columns are meaningful only on live runs.
