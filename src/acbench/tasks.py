"""The one task format every suite converts to.

The environment (tools, database, policy) is built by the task's suite; the task carries what
differs per task: the customer's persona and goal, what the right answer is, which capabilities
it feeds, and which split it belongs to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# train: what AgentCompile may learn from. dev: all tuning and fixes are chosen here.
# heldout: run once per release, never used to pick what to fix (SPEC: dev vs final).
SPLITS = ("train", "dev", "heldout")

CAPABILITIES = {
    "C1": "Repetition",
    "C2": "Correctness",
    "C3": "Generalization",
    "C4": "Knowing when not to act",
    "C5": "Consent",
    "C6": "Wrong actions",
    "C7": "Consistency",
    "C8": "Latency",
    "C9": "Fail-open",
    "C10": "Providers",
    "C11": "Learning curve",
    "C12": "Overhead",
    "C13": "Reply quality",
    "C14": "Generality",
    "C15": "Recipe authoring",
}


@dataclass(frozen=True)
class Task:
    suite: str
    id: str  # unique within the suite
    instruction: str  # the customer's persona and goal, as the simulated customer sees it
    split: str = "dev"
    job: str | None = None  # the repeated job this task is an instance of, if any
    tags: tuple[str, ...] = ()  # capability tags, C1..C15
    # Suite-specific answer key: expected writes, required outputs, expected end state.
    expected: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"{self.key}: split must be one of {SPLITS}, got {self.split!r}")
        unknown = [t for t in self.tags if t not in CAPABILITIES]
        if unknown:
            raise ValueError(f"{self.key}: unknown capability tags {unknown}")

    @property
    def key(self) -> str:
        return f"{self.suite}/{self.id}"
