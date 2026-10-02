"""What a suite adapter provides: its tasks, and a fresh environment and customer per task."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from ..customer import Customer, SimulatedCustomer
from ..tasks import Task


@dataclass(frozen=True)
class Outcome:
    success: bool
    detail: dict[str, Any] = field(default_factory=dict)


class Environment(abc.ABC):
    """One conversation's world: tools over a private copy of the database, and the policy the
    agent is given."""

    policy: str
    tools: list[dict[str, Any]]  # OpenAI function-tool schemas
    write_tools: frozenset[str]  # tools that change the database
    stop_tools: frozenset[str] = frozenset()  # tools that end the conversation (hand to a human)

    @abc.abstractmethod
    def call(self, name: str, args: dict[str, Any]) -> str:
        """Run one tool and return what the agent sees. Never raises: errors are observations."""

    @abc.abstractmethod
    def outcome(self, replies: list[str]) -> Outcome:
        """Score the conversation from the end state of the database and the agent's replies."""


class Suite(abc.ABC):
    name: str

    @abc.abstractmethod
    def tasks(self) -> list[Task]: ...

    @abc.abstractmethod
    def environment(self, task: Task) -> Environment: ...

    def customer(self, task: Task, chat: Any) -> Customer:
        """A model-simulated customer by default; `chat` sends one customer model call."""
        return SimulatedCustomer(task.instruction, chat)

    def select(self, split: str | None = None, ids: list[str] | None = None) -> list[Task]:
        tasks = self.tasks()
        if split:
            tasks = [t for t in tasks if t.split == split]
        if ids:
            wanted = set(ids)
            tasks = [t for t in tasks if t.id in wanted]
            missing = wanted - {t.id for t in tasks}
            if missing:
                raise ValueError(f"{self.name}: no tasks {sorted(missing)}")
        return tasks
