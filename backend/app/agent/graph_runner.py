"""GraphAgentRunner: the agent loop expressed as a LangGraph StateGraph.

Same `.run(session, conversation) -> AgentAnswer` interface and same behavior as
the hand-rolled AgentRunner. It reuses the tested primitives — tool execution
(policy gate + audit via `_execute`), tool schemas, history, trace recording — and
only re-expresses the control flow as a graph:

    START -> call_model --(final/answer)--> END
                 |  ^
             (tool calls) |
                 v  |
              run_tools ---

Eval parity across the two runners is the proof (tests/test_runner_parity.py).
"""
from __future__ import annotations

import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from app.agent.prompt import SYSTEM_PROMPT
from app.agent.runner import AgentRunner
from app.agent.schema import FINAL_TOOL_NAME, AgentAnswer
from app.models import Conversation
from app.observability.trace import record_step
from app.tools import ToolContext, ToolResult

_MAX_STEPS_ANSWER = AgentAnswer(
    answer="This needs a closer look — escalating to a human.",
    should_escalate=True,
    escalation_reason="max_steps_exhausted",
)


class _State(TypedDict, total=False):
    messages: list
    step_no: int
    turns: int
    answer: object
    pending: object


class GraphAgentRunner:
    def __init__(self, llm, registry, docsearch, max_steps: int = 8, policy=None) -> None:
        # Reuse the hand-rolled runner's tested helpers so only orchestration differs.
        self._base = AgentRunner(llm, registry, docsearch, max_steps, policy)
        self.max_steps = max_steps

    def run(self, session: Session, conversation: Conversation) -> AgentAnswer:
        base = self._base
        ctx = ToolContext(
            session=session,
            docsearch=base.docsearch,
            customer_id=conversation.customer_id,
            conversation_id=conversation.id,
        )
        history = base._history(session, conversation)
        if not history:
            return AgentAnswer(
                answer="There is no customer message to respond to.",
                should_escalate=True,
                escalation_reason="empty_conversation",
            )
        tool_schemas = base._tool_schemas()
        conv_id = conversation.id

        def call_model(state: _State) -> dict:
            turns = state.get("turns", 0) + 1
            if turns > self.max_steps:
                return {"turns": turns, "answer": _MAX_STEPS_ANSWER}
            step_no = state["step_no"] + 1
            t0 = time.perf_counter()
            resp = base.llm.create(
                system=SYSTEM_PROMPT, messages=state["messages"], tools=tool_schemas
            )
            latency = int((time.perf_counter() - t0) * 1000)
            record_step(
                session,
                conversation_id=conv_id,
                step_no=step_no,
                kind="model",
                thought=resp.text or None,
                latency_ms=latency,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                cache_read_tokens=resp.cache_read_tokens,
                model=resp.model,
            )
            if resp.stop_reason == "refusal":
                return {
                    "turns": turns,
                    "step_no": step_no,
                    "answer": AgentAnswer(
                        answer="I'm sorry, I can't help with that request.",
                        should_escalate=True,
                        escalation_reason="model_refusal",
                    ),
                }
            if not resp.tool_uses:
                return {
                    "turns": turns,
                    "step_no": step_no,
                    "answer": AgentAnswer(answer=resp.text or "", confidence=0.4),
                }
            messages = [*state["messages"], {"role": "assistant", "content": resp.assistant_content}]
            final = next((tu for tu in resp.tool_uses if tu.name == FINAL_TOOL_NAME), None)
            if final is not None:
                try:
                    answer = AgentAnswer(**final.input)
                except Exception as exc:
                    answer = AgentAnswer(
                        answer=str(final.input.get("answer", "")),
                        should_escalate=True,
                        escalation_reason=f"invalid_final:{exc}",
                    )
                return {"turns": turns, "step_no": step_no, "messages": messages, "answer": answer}
            return {"turns": turns, "step_no": step_no, "messages": messages, "pending": resp.tool_uses}

        def run_tools(state: _State) -> dict:
            step_no = state["step_no"]
            blocks: list[dict] = []
            for tu in state["pending"]:
                step_no += 1
                t0 = time.perf_counter()
                tool = base.registry.get(tu.name)
                if tool is None:
                    result = ToolResult(content=f"Unknown tool: {tu.name}", is_error=True)
                else:
                    try:
                        args = tool.args_model(**tu.input)
                        result = base._execute(ctx, tool, args, tu.input)
                    except Exception as exc:
                        result = ToolResult(content=f"Tool error: {exc}", is_error=True)
                latency = int((time.perf_counter() - t0) * 1000)
                record_step(
                    session,
                    conversation_id=conv_id,
                    step_no=step_no,
                    kind="tool",
                    tool_name=tu.name,
                    tool_args=tu.input,
                    tool_result=result.content,
                    latency_ms=latency,
                )
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": result.content,
                        "is_error": result.is_error,
                    }
                )
            messages = [*state["messages"], {"role": "user", "content": blocks}]
            return {"step_no": step_no, "messages": messages, "pending": None}

        def route(state: _State) -> str:
            return "end" if state.get("answer") is not None else "tools"

        builder = StateGraph(_State)
        builder.add_node("call_model", call_model)
        builder.add_node("run_tools", run_tools)
        builder.add_edge(START, "call_model")
        builder.add_conditional_edges("call_model", route, {"tools": "run_tools", "end": END})
        builder.add_edge("run_tools", "call_model")
        graph = builder.compile()

        final_state = graph.invoke(
            {"messages": history, "step_no": 0, "turns": 0, "answer": None, "pending": None},
            config={"recursion_limit": self.max_steps * 2 + 5},
        )
        answer = final_state.get("answer")
        return answer if answer is not None else _MAX_STEPS_ANSWER
