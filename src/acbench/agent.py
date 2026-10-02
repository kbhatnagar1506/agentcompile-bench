"""The default reference agent: a raw tool-calling loop over an OpenAI-style client, no framework.

The client may be wrapped by AgentCompile; the loop cannot tell, which is the point. It runs
every tool call in a reply, in order, and speaks to the customer whenever the model answers
with text.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .customer import STOP, Customer
from .recorder import Recorder
from .suites.base import Environment


class ReferenceAgent:
    def __init__(
        self, client: Any, model: str, *, temperature: float = 0.0, max_steps: int = 30
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_steps = max_steps

    def run(self, env: Environment, customer: Customer, rec: Recorder) -> str:
        """Run the conversation to its end; returns why it ended."""
        first = customer.start()
        rec.customer(first)
        if STOP in first:
            return "customer_stop"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": env.policy},
            {"role": "user", "content": first},
        ]
        turn_started = time.perf_counter()
        for step in range(self.max_steps):
            response, route = rec.agent_call(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=env.tools,
                    temperature=self.temperature,
                ),
                step=step,
            )
            by = "compiled" if route == "compiled" else "agent"
            message = response.choices[0].message
            calls = list(message.tool_calls or [])
            if calls:
                messages.append(_assistant_with_calls(message.content, calls))
                stop = False
                for call in calls:
                    name = call.function.name
                    try:
                        args = json.loads(call.function.arguments or "{}")
                        result = env.call(name, args) if isinstance(args, dict) else (
                            "Error: tool arguments must be a JSON object"
                        )
                    except json.JSONDecodeError as exc:
                        args, result = call.function.arguments, f"Error: bad JSON arguments: {exc}"
                    rec.tool(
                        step=step, name=name, args=args, by=by,
                        write=name in env.write_tools, result=result,
                    )
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                    stop = stop or name in env.stop_tools
                if stop:
                    return "stop_tool"
                continue
            text = message.content or ""
            turn_ms = (time.perf_counter() - turn_started) * 1000
            rec.reply(step=step, by=by, text=text, turn_ms=turn_ms)
            messages.append({"role": "assistant", "content": text})
            answer = customer.reply(text)
            rec.turn += 1
            rec.customer(answer)
            if STOP in answer:
                return "customer_stop"
            messages.append({"role": "user", "content": answer})
            turn_started = time.perf_counter()
        return "max_steps"


def _assistant_with_calls(content: str | None, calls: list[Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments},
            }
            for c in calls
        ],
    }
