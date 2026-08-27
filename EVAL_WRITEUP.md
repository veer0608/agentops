# AgentOps: eval writeup (measure → diagnose → improve)

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
| **Citation grounding** | Fraction of answers whose every cited `[DOC-###]` / `[GH-###]` marker was actually retrieved this run: a deterministic **hallucination proxy** (an LLM judge slots in when a key is present). |
| **Cost / latency** | Per-run USD (from token usage × a price table) and p50/p95 latency, both read from the `agent_steps` trace. |

## Baseline (10 scenarios, offline demo model)

The offline demo model is a fixed heuristic, *always `search_docs`, then answer*,
run with no API key so the harness itself is verifiable end to end.

| Tool-sel. F1 | Task-success | Escalation acc. | Citation grounding | Cost | Latency p50/p95 |
|---|---|---|---|---|---|
| 0.60 | 0.50 | 0.90 | 1.00 | $0.00 | ~0 / ~0 ms |

## Diagnosis: where it fails, and why

5 of 10 scenarios fail. Grouping the failures is the whole game:

| Failing scenario | Expected | What the demo did | Failure class |
|---|---|---|---|
| `whats-my-plan` | `get_customer`, `get_subscription` | `search_docs` | tool selection (account) |
| `subscription-status` | `get_customer`, `get_subscription` | `search_docs` | tool selection (account) |
| `known-issue-dashboard` | `search_github_issues` | `search_docs` | tool selection (known-issue) |
| `known-issue-rate-limit` | `search_github_issues` | `search_docs` | tool selection (known-issue) |
| `refund-request-action` | escalate | answered instead | escalation |

**100% of failures are routing decisions**: picking the wrong tool, or not choosing
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

## Live run: 8 passes, and why one pass was never going to do

Run against `openai/gpt-oss-120b` on Groq, reached through the OpenAI-compatible
`OPENAI_BASE_URL` seam. Eight consecutive passes over the same 10 scenarios, same
config, minutes apart.

| Metric | Baseline (demo) | Live, mean of 8 | sd | Range |
|---|---|---|---|---|
| Tool-selection F1 | 0.60 | **0.780** | 0.106 | 0.580 - 0.967 |
| Tool exact-match | n/a | 0.637 | 0.122 | 0.500 - 0.900 |
| Task-success | 0.50 | 0.475 | 0.139 | 0.400 - 0.800 |
| Escalation accuracy | 0.90 | 0.725 | 0.120 | 0.600 - 0.900 |
| Citation grounding | 1.00 | **1.000** | 0.000 | 1.000 - 1.000 |

**The first run scored F1 0.88. The fifth scored 0.58.** Nothing changed between
them. A standard deviation of 0.106 on a 10-scenario suite is wider than the
effect being measured, so a single pass is a sample, not a result, and reporting
one would have meant publishing whichever number the dice produced. This is the
finding, and it applies to every agent benchmark of this size.

What survives the variance:

- **Tool selection genuinely improved.** Seven of eight runs beat the 0.60
  baseline, and the mean is +0.18. This is the one claim the spread supports.
- **Grounding is 1.000 with zero deviation** across 8 runs and 80 scenario-runs.
  A hallucination proxy that never once fired is a stronger statement than any
  single-run figure here.
- **Task-success did not move** (0.475 against 0.50), and **escalation accuracy
  got worse** (0.725 against 0.90). The baseline's heuristic never escalates
  when it should not; a real model sometimes does. The diagnosis therefore moves
  from "every failure is tool selection" to "tool selection is largely solved and
  the escalation decision is now the bottleneck".

### 11% of scenario-runs never reached the agent

Nine of the 80 scenario-runs died on a provider-side 400: the model emitted a
call to `response_to_customer`, a tool that does not exist in this repo, and the
API rejected the whole request rather than returning the bad call for the agent
to recover from.

It hit five different scenarios and appeared in six of the eight runs, so it is
not one broken scenario. It is a random tax that scores as an agent failure while
saying nothing about the agent, and it depresses both F1 and task-success above.
Two consecutive runs happened to hit the same scenario, which is exactly long
enough to mistake it for a reproducible bug.

The harness records these in the row's answer as `[run error: ...]` rather than
hiding them, which is the only reason the pattern was visible at all. Not scoring
them at all is the obvious next change.

### Reproducing it

```bash
printf 'PROVIDER=openai
OPENAI_API_KEY=gsk_...
OPENAI_MODEL=openai/gpt-oss-120b
OPENAI_BASE_URL=https://api.groq.com/openai/v1
' >> backend/.env
backend/.venv/Scripts/python -m evals.runner
```

`OPENAI_BASE_URL` points the tested OpenAI adapter at any compatible endpoint
(Groq, Together, OpenRouter, a local server) without touching code. Leave it unset
for OpenAI itself.

## Honest limitations

- **10 scenarios**, not 50–100 yet. Scaling is continued authoring (same discipline
  as CiteRAG's golden set); the harness and scorers don't change.
- **Grounding is a deterministic proxy.** It catches fabricated citations, not subtly
  wrong claims grounded in the wrong passage, which needs the LLM judge (Phase 3.5).
- **Offline latency/cost are ~0** because the demo model does no network I/O; these
  columns are meaningful only on live runs.
- **One pass is not a measurement.** At 10 scenarios the run-to-run sd is 0.11 on
  F1 and 0.14 on task-success, so any figure quoted from a single run is inside
  the noise. Quote a mean with its spread and the number of runs, or quote
  nothing. Enlarging the suite is what actually narrows this.
- **Provider-agnostic has one live gap.** Gemini 3.x rejects replayed function
  calls that carry no `thought_signature`, which the OpenAI-compatible adapter
  does not emit, so multi-turn tool calling fails on every currently available
  Gemini model. `gemini-2.0-flash`, which predates the requirement, is retired.
  OpenAI, Anthropic and any OpenAI-compatible endpoint are unaffected.
