"""Arms: the same agent, tasks and window, with and without AgentCompile in the path.

baseline  the agent alone
sdk       the agent's client wrapped by agentcompile.wrap(mode="live")
shadow    wrapped with mode="shadow": AgentCompile decides, the agent's model always answers

Chaos arms (decision service down, slow, erroring, malformed) are milestone 3.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import agentcompile

ARMS = ("baseline", "sdk", "shadow")


@dataclass(frozen=True)
class SdkSettings:
    key: str | None = None  # or AGENTCOMPILE_KEY
    base_url: str | None = None  # or AGENTCOMPILE_URL, else the hosted service
    timeout: float = 2.0
    # Builds the HTTP client the SDK sends decisions with; tests use it for a local decision
    # service, chaos arms for fault injection.
    http_client: Callable[[], Any] | None = None

    def check(self) -> None:
        if not (self.key or os.environ.get("AGENTCOMPILE_KEY")) and self.http_client is None:
            raise RuntimeError(
                "the sdk and shadow arms need AGENTCOMPILE_KEY (or --ac-key); without it every "
                "call would fail open and the arm would measure nothing"
            )


def client_for(
    arm: str, client: Any, on_event: Callable[[dict[str, Any]], None], sdk: SdkSettings
) -> Any:
    if arm == "baseline":
        return client
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; known: {ARMS}")
    return agentcompile.wrap(
        client,
        key=sdk.key,
        base_url=sdk.base_url,
        mode="shadow" if arm == "shadow" else "live",
        timeout=sdk.timeout,
        trail=False,  # the bench writes its own trace, with the trail's fields
        on_event=on_event,
        http_client=sdk.http_client() if sdk.http_client else None,
    )
