"""AgentRunner: the manual tool-use loop, written against the LLMClient seam.

Kept behind this class so a Phase-4 LangGraph / Tool Runner implementation can be
swapped in without touching tools, persistence, or the API. Every model turn and
tool execution is written to `agent_steps` (the trace + eval data source).
"""
from __future__ import annotations

import time

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.llm import LLMClient
from app.agent.prompt import SYSTEM_PROMPT
from app.agent.schema import FINAL_TOOL_NAME, AgentAnswer, final_tool_schema
from app.models import Conversation, Message
from app.observability.trace import log, record_step
from app.policy import PolicyAction, PolicyGate, audit, enqueue_review
from app.tools import DocSearch, SideEffect, Tool, ToolContext, ToolResult


class AgentRunner:
    def __init__(
        self,
        llm: LLMClient,
        registry: dict[str, Tool],
        docsearch: DocSearch,
        max_steps: int = 8,
        policy: PolicyGate | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.docsearch = docsearch
        self.max_steps = max_steps
        self.policy = policy or PolicyGate()

    def _tool_schemas(self) -> list[dict]:
        schemas = [tool.anthropic_schema() for tool in self.registry.values()]
        schemas.append(final_tool_schema())
        return schemas

    def _history(self, session: Session, conversation: Conversation) -> list[dict]:
        msgs = session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at)
        ).all()
        return [
            {"role": "user" if m.role == "customer" else "assistant", "content": m.content}
            for m in msgs
        ]

    def _execute(self, ctx: ToolContext, tool: Tool, args, raw_input: dict) -> ToolResult:
        """Read tools auto-run; writes go through the policy gate + audit log."""
        if tool.side_effect == SideEffect.READ:
            return tool.handler(ctx, args)

        decision = self.policy.evaluate(tool, raw_input, ctx)
        audit(
            ctx.session,
            conversation_id=ctx.conversation_id,
            actor="policy",
            action=tool.name,
            decision=decision.action.value,
            reason=decision.reason,
            payload=raw_input,
        )
        if decision.action == PolicyAction.AUTO:
            result = tool.handler(ctx, args)
            audit(
                ctx.session,
                conversation_id=ctx.conversation_id,
                actor="agent",
                action=tool.name,
                decision="executed",
                payload=raw_input,
            )
            return result
        if decision.action == PolicyAction.REVIEW:
            review = enqueue_review(
                ctx.session, ctx.conversation_id, tool.name, raw_input, decision.reason
            )
            return ToolResult(
                content=(
                    f"Action '{tool.name}' was queued for human review "
                    f"(review_id={review.id}; reason: {decision.reason}). It has NOT "
                    "been applied — tell the customer it is under review."
                ),
                data={"review_id": review.id},
            )
        return ToolResult(
            content=(
                f"Action '{tool.name}' is blocked by policy ({decision.reason}). "
                "Escalate to a human instead."
            )
        )

    def run(self, session: Session, conversation: Conversation) -> AgentAnswer:
        ctx = ToolContext(
            session=session,
            docsearch=self.docsearch,
            customer_id=conversation.customer_id,
            conversation_id=conversation.id,
        )
        messages = self._history(session, conversation)
        if not messages:
            return AgentAnswer(
                answer="There is no customer message to respond to.",
                should_escalate=True,
                escalation_reason="empty_conversation",
            )

        tool_schemas = self._tool_schemas()
        step_no = 0

        for _ in range(self.max_steps):
            step_no += 1
            t0 = time.perf_counter()
            resp = self.llm.create(
                system=SYSTEM_PROMPT, messages=messages, tools=tool_schemas
            )
            latency_ms = int((time.perf_counter() - t0) * 1000)
            record_step(
                session,
                conversation_id=conversation.id,
                step_no=step_no,
                kind="model",
                thought=resp.text or None,
                latency_ms=latency_ms,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                cache_read_tokens=resp.cache_read_tokens,
                model=resp.model,
            )

            if resp.stop_reason == "refusal":
                return AgentAnswer(
                    answer="I'm sorry, I can't help with that request.",
                    should_escalate=True,
                    escalation_reason="model_refusal",
                )

            if not resp.tool_uses:
                return AgentAnswer(answer=resp.text or "", confidence=0.4)

            # Record the assistant turn (with its tool_use blocks) into history.
            messages.append({"role": "assistant", "content": resp.assistant_content})

            # Terminal tool ends the loop.
            final = next(
                (tu for tu in resp.tool_uses if tu.name == FINAL_TOOL_NAME), None
            )
            if final is not None:
                try:
                    return AgentAnswer(**final.input)
                except Exception as exc:  # malformed final -> escalate rather than crash
                    log.warning("invalid_final_answer", error=str(exc))
                    return AgentAnswer(
                        answer=str(final.input.get("answer", "")),
                        should_escalate=True,
                        escalation_reason=f"invalid_final:{exc}",
                    )

            # Execute the requested read tools and feed results back.
            tool_result_blocks: list[dict] = []
            for tu in resp.tool_uses:
                step_no += 1
                t_tool = time.perf_counter()
                tool = self.registry.get(tu.name)
                if tool is None:
                    result = ToolResult(content=f"Unknown tool: {tu.name}", is_error=True)
                else:
                    try:
                        args = tool.args_model(**tu.input)
                        result = self._execute(ctx, tool, args, tu.input)
                    except Exception as exc:
                        result = ToolResult(content=f"Tool error: {exc}", is_error=True)
                tool_latency = int((time.perf_counter() - t_tool) * 1000)
                record_step(
                    session,
                    conversation_id=conversation.id,
                    step_no=step_no,
                    kind="tool",
                    tool_name=tu.name,
                    tool_args=tu.input,
                    tool_result=result.content,
                    latency_ms=tool_latency,
                )
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": result.content,
                        "is_error": result.is_error,
                    }
                )
            messages.append({"role": "user", "content": tool_result_blocks})

        return AgentAnswer(
            answer="This needs a closer look — escalating to a human.",
            should_escalate=True,
            escalation_reason="max_steps_exhausted",
        )
