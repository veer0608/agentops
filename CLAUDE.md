# AgentOps

An AI support agent that takes real actions behind a policy/escalation gate, with a
provider-agnostic LLM seam and a built-in eval harness. README.md covers the design;
this file is how to work in it.

## Layout

The repo root holds the docs (`README.md`, `ARCHITECTURE.md`, `EVAL_WRITEUP.md`).
**The Python project is `backend/`** — everything below runs from there.

- `backend/app/` — `agent/` (runner, LLM seam), `observability/`, API, services
- `backend/app/testing/` — `mock_model.py`, the deterministic client used by tests
- `backend/evals/` — `runner.py`, `scorers.py`, `scenarios/`, `pricing.py`
- `backend/tests/` — 33 tests
- `backend/alembic/` — migrations

## Commands

Run from `backend/`. Python 3.11 (the floor in `pyproject.toml`).

```bash
pip install -r requirements.txt      # deps are here, NOT in pyproject.toml
python -m pytest -q                  # 33 tests, no key needed
```

`pyproject.toml` sets `pythonpath = ["."]` and `testpaths = ["tests"]`, so pytest
must be invoked from `backend/` or collection fails on `app` imports.

## CI

`.github/workflows/ci.yml` runs the tests on 3.11 only. That is deliberate: 3.11 is
the verified version, and a red badge on arrival says less than no badge. Widen to a
matrix once 3.12 has been seen green.

No CI step runs the eval suite — it needs a model, and a job that needs a key goes
red for anyone who forks the repo. `test_eval_runner.py`, `test_eval_scorers.py` and
`test_runner_parity.py` cover that logic instead.

## What the tests protect

- **`test_runner_parity.py` is the important one.** Two agent implementations exist —
  the in-house runner and a LangGraph one behind the same interface — and this is what
  would notice them quietly disagreeing. If it fails, do not "fix" it by loosening the
  assertion.
- **`test_policy_gate.py`** covers the gate that decides auto-execute vs human review
  vs block. That gate is the project's whole claim; treat changes to it as changes to
  the thesis.

## Environment

- Shell is PowerShell; POSIX flags like `ls -la` fail there.
- Docker cannot run on this machine (Win11 Home / VBS). Do not propose it.
- The only live key is Groq free tier, and its real limit is tokens-per-day, which
  appears in no response header. Anything that calls a model should assume a hard
  daily ceiling and fail gracefully rather than mid-run.
