"""The customer side of a conversation: a model playing a persona, or a fixed script."""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

STOP = "###STOP###"
OPENING = "Hi! How can I help you today?"

# A customer model call: messages in, the customer's next line out. The runner supplies it, so
# every customer call is traced, priced and counted against the budget.
Chat = Callable[[list[dict[str, Any]]], str]


class Customer(abc.ABC):
    @abc.abstractmethod
    def start(self) -> str:
        """The customer's first message."""

    @abc.abstractmethod
    def reply(self, agent_text: str) -> str:
        """The customer's answer to the agent; contains STOP when the customer is done."""


class SimulatedCustomer(Customer):
    def __init__(self, instruction: str, chat: Chat) -> None:
        self.chat = chat
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": _prompt(instruction)},
            {"role": "user", "content": OPENING},
        ]

    def start(self) -> str:
        return self._next()

    def reply(self, agent_text: str) -> str:
        self.messages.append({"role": "user", "content": agent_text})
        return self._next()

    def _next(self) -> str:
        text = self.chat(self.messages) or ""
        self.messages.append({"role": "assistant", "content": text})
        return text


class ScriptedCustomer(Customer):
    """Says its lines in order, whatever the agent says, then stops. For offline suites, tests."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = list(lines)
        self.said = 0

    def start(self) -> str:
        return self._next()

    def reply(self, agent_text: str) -> str:
        return self._next()

    def _next(self) -> str:
        if self.said >= len(self.lines):
            return STOP
        self.said += 1
        return self.lines[self.said - 1]


def _prompt(instruction: str) -> str:
    return f"""You are playing a customer talking to a support agent.

Your situation and goal:
{instruction}

How to play it:
- Write one short message at a time, as the customer would.
- Reveal details only when they are needed or asked for; do not state your whole goal at once.
- Never invent details your situation does not give you. If asked for something you do not
  have, say you do not know or do not have it.
- Put things in your own words rather than repeating the situation text.
- Keep to the personality described.
- When your goal is met, or clearly cannot be met, reply with only {STOP}."""
