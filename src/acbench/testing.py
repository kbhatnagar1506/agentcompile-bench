"""Offline stand-ins for the toy suite: an agent model and an AgentCompile decision service.

FakeAgentModel is an OpenAI-shaped client (`fake/<anything>` as the agent model) that plays a
correct toy-shop agent from the conversation alone. FakeDecisionService answers /v1/decide like
the hosted service would for a compiled `order_status` job and forwards everything else; it can
also fail in each of the ways the chaos suite will inject. Neither makes a network call.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

import httpx

ORDER_ID = re.compile(r"\b([A-Z]\d)\b")


class FakeAgentModel:
    """Use exactly like OpenAI(): client.chat.completions.create(...)."""

    def __init__(self) -> None:
        self.chat = _Chat(self)
        self.calls = 0

    def respond(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls += 1
        first = _first_user(messages)
        order = _order_id(first)
        cancel = "cancel" in first.lower()
        last = messages[-1]
        if last["role"] == "tool":
            tool = _last_tool_name(messages)
            result = last["content"]
            if result.startswith("Error"):
                return _text(f"Sorry, I couldn't do that: {result}")
            status = json.loads(result)["status"]
            if tool == "cancel_order":
                return _text(f"Done: order {order} is now cancelled.")
            if cancel and status == "pending":
                return _text(f"Order {order} is pending. I can cancel it. Shall I go ahead?")
            if cancel:
                return _text(f"Order {order} is {status}, so it can't be cancelled.")
            return _text(f"Your order {order} is {status}.")
        said = str(last.get("content") or "").lower()
        if cancel and "yes" in said and order:
            return _tool_call("cancel_order", {"order_id": order})
        if order and not any(m["role"] == "tool" for m in messages):
            return _tool_call("get_order", {"order_id": order})
        return _text("Is there anything else I can help with?")


class _Chat:
    def __init__(self, model: FakeAgentModel) -> None:
        self.completions = _Completions(model)


class _Completions:
    def __init__(self, model: FakeAgentModel) -> None:
        self._model = model

    def create(self, *, model: str, messages: list[dict[str, Any]], **_: Any) -> Any:
        from openai.types.chat import ChatCompletion

        message = self._model.respond(messages)
        prompt = sum(len(str(m.get("content") or "")) for m in messages) // 4 + 50
        return ChatCompletion.model_validate(
            {
                "id": f"chatcmpl-fake-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                    }
                ],
                "usage": {"prompt_tokens": prompt, "completion_tokens": 20,
                          "total_tokens": prompt + 20},
            }
        )


class FakeDecisionService:
    """A local /v1/decide. `fault` is one of: None, "down", "http500", "malformed", "timeout"."""

    STATUS_JOB = "order_status"
    WORDING_COST_USD = 0.0002  # what a `worded` reply would cost us; reported in the events

    def __init__(self, fault: str | None = None) -> None:
        self.fault = fault
        self.decisions = 0

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handle))

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.decisions += 1
        if self.fault == "down":
            raise httpx.ConnectError("decision service down", request=request)
        if self.fault == "timeout":
            raise httpx.ReadTimeout("decision service slow", request=request)
        if self.fault == "http500":
            return httpx.Response(500, json={"error": "boom"})
        if self.fault == "malformed":
            return httpx.Response(200, json={"action": "tool_call"})
        messages = json.loads(request.content)["request"].get("messages", [])
        return httpx.Response(200, json=self.decide(messages))

    def decide(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        first = _first_user(messages)
        order = _order_id(first)
        if not order or "cancel" in first.lower():
            return {"action": "forward", "reason": "no job matched"}
        job = [{"job": self.STATUS_JOB}]
        last = messages[-1]
        if last["role"] == "user" and not any(m["role"] == "tool" for m in messages):
            return {"action": "tool_call", "tool": "get_order", "args": {"order_id": order},
                    "events": job}
        if last["role"] == "tool" and not last["content"].startswith("Error"):
            status = json.loads(last["content"])["status"]
            return {"action": "say", "text": f"Your order {order} is {status}.",
                    "events": [{**job[0], "cost_usd": self.WORDING_COST_USD}]}
        return {"action": "forward", "reason": "job finished", "events": job}


def _first_user(messages: list[dict[str, Any]]) -> str:
    return next((str(m.get("content") or "") for m in messages if m["role"] == "user"), "")


def _order_id(text: str) -> str | None:
    match = ORDER_ID.search(text)
    return match.group(1) if match else None


def _last_tool_name(messages: list[dict[str, Any]]) -> str | None:
    for m in reversed(messages):
        if m["role"] == "assistant" and m.get("tool_calls"):
            return m["tool_calls"][-1]["function"]["name"]
    return None


def _text(content: str) -> dict[str, Any]:
    return {"role": "assistant", "content": content}


def _tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }
