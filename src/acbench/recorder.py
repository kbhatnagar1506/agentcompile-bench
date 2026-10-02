"""One conversation's bookkeeping: every model call is priced, charged to the run's budget and
written to the trace as it happens, so an interrupted run still has an exact record."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .models import Budget, ModelRef, Prices, Usage, usage_of, with_retries
from .trace import TraceWriter


class Recorder:
    def __init__(
        self,
        context: dict[str, Any],
        writer: TraceWriter,
        budget: Budget,
        prices: Prices,
        agent_model: ModelRef,
        customer_model: ModelRef | None,
    ) -> None:
        self.context = context
        self.writer = writer
        self.budget = budget
        self.prices = prices
        self.agent_model = agent_model
        self.customer_model = customer_model
        self.turn = 0
        self.replies: list[str] = []
        self.totals = {
            "agent_calls": 0,  # calls that reached the agent's model
            "compiled_calls": 0,  # calls AgentCompile answered itself
            "input_tokens": 0,
            "output_tokens": 0,
            "agent_cost_usd": 0.0,
            "customer_cost_usd": 0.0,
            "ac_cost_usd": 0.0,  # AgentCompile's own model calls, as its decisions report them
            "writes": {"agent": [0, 0], "compiled": [0, 0]},  # by -> [sent, accepted]
        }
        self._sdk_event: dict[str, Any] | None = None
        self._customer_usage = (Usage(), 0.0)

    # --- the SDK reports each wrapped call here (agentcompile.wrap(on_event=...)) ---

    def on_sdk_event(self, event: dict[str, Any]) -> None:
        self._sdk_event = event

    # --- agent side ---

    def agent_call(self, create: Callable[[], Any], *, step: int) -> tuple[Any, str]:
        """Make one agent model call; returns (response, route)."""
        self.budget.check()
        self._sdk_event = None
        started = time.perf_counter()
        response = with_retries(create)
        measured_ms = (time.perf_counter() - started) * 1000
        sdk = self._sdk_event or {}
        route = sdk.get("route") or "baseline"
        usage = Usage() if route == "compiled" else usage_of(response)
        cost = self.prices.cost(self.agent_model, usage)
        self.budget.charge(cost)
        events = sdk.get("events") or []
        ac_cost = sum(float(e.get("cost_usd") or 0) for e in events)
        if route == "compiled":
            self.totals["compiled_calls"] += 1
        else:
            self.totals["agent_calls"] += 1
        self.totals["input_tokens"] += usage.input
        self.totals["output_tokens"] += usage.output
        self.totals["agent_cost_usd"] += cost
        self.totals["ac_cost_usd"] += ac_cost
        self._write(
            "agent_call",
            step=step,
            turn=self.turn,
            route=route,
            action=sdk.get("action"),
            tool=sdk.get("tool"),
            reason=sdk.get("reason"),
            job=_job(events),
            events=events or None,
            decide_ms=sdk.get("decide_ms"),
            total_ms=round(measured_ms, 1),
            input_tokens=usage.input,
            output_tokens=usage.output,
            cost_usd=cost,
            ac_cost_usd=ac_cost or None,
        )
        return response, route

    def tool(
        self, *, step: int, name: str, args: Any, by: str, write: bool, result: str
    ) -> None:
        ok = not result.startswith(("Error", "Unknown action"))
        if write:
            sent_accepted = self.totals["writes"][by]
            sent_accepted[0] += 1
            sent_accepted[1] += int(ok)
        self._write(
            "tool", step=step, turn=self.turn, tool=name, args=args, by=by, write=write, ok=ok,
            result=result[:2000],
        )

    def reply(self, *, step: int, by: str, text: str, turn_ms: float) -> None:
        self.replies.append(text)
        self._write("reply", step=step, turn=self.turn, by=by, text=text, turn_ms=round(turn_ms, 1))

    # --- customer side ---

    def customer_chat(self, client: Any) -> Callable[[list[dict[str, Any]]], str]:
        """The `chat` a SimulatedCustomer calls; its cost is written with the customer's message."""
        model = self.customer_model
        assert model is not None

        def chat(messages: list[dict[str, Any]]) -> str:
            self.budget.check()
            response = with_retries(
                lambda: client.chat.completions.create(model=model.name, messages=messages)
            )
            usage = usage_of(response)
            cost = self.prices.cost(model, usage)
            self.budget.charge(cost)
            self._customer_usage = (usage, cost)
            return response.choices[0].message.content or ""

        return chat

    def customer(self, text: str) -> None:
        usage, cost = self._customer_usage
        self._customer_usage = (Usage(), 0.0)
        self.totals["customer_cost_usd"] += cost
        self._write(
            "customer", turn=self.turn, text=text, input_tokens=usage.input,
            output_tokens=usage.output, cost_usd=cost,
        )

    # --- end ---

    def end(
        self, *, status: str, ended: str, success: bool | None, detail: dict[str, Any] | None,
        wall_ms: float, error: str | None = None,
    ) -> None:
        self._write(
            "conversation", status=status, ended=ended, success=success, detail=detail,
            turns=self.turn, wall_ms=round(wall_ms, 1), error=error, **self.totals,
        )

    def _write(self, kind: str, **fields: Any) -> None:
        event = {"kind": kind, **self.context}
        event.update({k: v for k, v in fields.items() if v is not None or k == "success"})
        self.writer.write(event)


def _job(events: list[dict[str, Any]]) -> str | None:
    """The job AgentCompile says it is running, if its decision events name one."""
    for event in events:
        job = event.get("job")
        if isinstance(job, str):
            return job
    return None
