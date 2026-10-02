"""The trace format: one JSON line per event, the same for every suite and arm.

Agent model calls carry the live SDK trail's fields unchanged (route, action, tool, reason,
events, decide_ms, total_ms), so a test run reads exactly like production, plus the bench's own:
which run, conversation, task, arm, trial and turn, tokens and cost. The scorer reads nothing
else, so anything a metric needs must be in the trace. Full field list: docs/TRACE.md.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

VERSION = 1

# Event kinds and the fields each must carry (on top of the conversation context fields).
KINDS: dict[str, tuple[str, ...]] = {
    "agent_call": (
        "step", "turn", "route", "input_tokens", "output_tokens", "cost_usd", "total_ms",
    ),
    "customer": ("turn", "text", "input_tokens", "output_tokens", "cost_usd"),
    "tool": ("step", "turn", "tool", "args", "by", "write", "ok"),
    "reply": ("step", "turn", "by", "text", "turn_ms"),
    "conversation": ("status", "ended", "success", "turns", "wall_ms"),
}
CONTEXT = ("run", "conversation", "suite", "task", "split", "arm", "agent_model", "trial")

# Routes of an agent model call. The SDK's own: compiled, forwarded, fail-open, shadow,
# no-conversation. The bench adds `baseline`: the agent alone, no SDK in the path.
ROUTES = ("baseline", "compiled", "forwarded", "fail-open", "shadow", "no-conversation")


class TraceWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict[str, Any]) -> None:
        validate(event)
        line = json.dumps({"v": VERSION, "ts": round(time.time(), 3), **event}, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def validate(event: dict[str, Any]) -> None:
    kind = event.get("kind")
    if kind not in KINDS:
        raise ValueError(f"unknown trace event kind {kind!r}")
    missing = [f for f in (*CONTEXT, *KINDS[kind]) if f not in event]
    if missing:
        raise ValueError(f"{kind} event missing {missing}")
    if kind == "agent_call" and event["route"] not in ROUTES:
        raise ValueError(f"unknown route {event['route']!r}")


def read(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)
