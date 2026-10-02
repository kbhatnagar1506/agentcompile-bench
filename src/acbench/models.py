"""Model clients, token counts, list prices and the run's dollar cap.

A model is named `provider/model`, e.g. `gemini/gemini-2.5-flash`. Every provider here is
reached through an OpenAI-compatible client, which is also what `agentcompile.wrap` wraps.
The full provider matrix (Anthropic, Azure, Bedrock; sync, async, streaming) is milestone 4.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Provider name -> (base URL, API key variable). `None` base URL means the OpenAI default.
PROVIDERS: dict[str, tuple[str | None, str]] = {
    "openai": (None, "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
}

# USD per million tokens, (input, output). List prices change: check them against the provider's
# pricing page before quoting a cost, and pass `prices=` to override. A model missing here
# cannot run, because the budget cap needs a price for every call.
PRICES: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}

RETRY_DELAYS = (2.0, 8.0, 30.0)


@dataclass(frozen=True)
class ModelRef:
    provider: str
    name: str

    @classmethod
    def parse(cls, spec: str) -> ModelRef:
        provider, sep, name = spec.partition("/")
        if not sep or not name:
            raise ValueError(f"model must be 'provider/model', got {spec!r}")
        return cls(provider, name)

    def __str__(self) -> str:
        return f"{self.provider}/{self.name}"


def make_client(ref: ModelRef) -> Any:
    """A fresh OpenAI-compatible client for this model's provider."""
    if ref.provider == "fake":
        from .testing import FakeAgentModel

        return FakeAgentModel()
    if ref.provider not in PROVIDERS:
        raise ValueError(f"unknown provider {ref.provider!r}; known: {sorted(PROVIDERS)} or fake")
    from openai import OpenAI

    base_url, key_var = PROVIDERS[ref.provider]
    key = os.environ.get(key_var)
    if not key:
        raise RuntimeError(f"{key_var} is not set (needed for {ref})")
    return OpenAI(api_key=key, base_url=base_url, max_retries=0)


@dataclass(frozen=True)
class Usage:
    input: int = 0
    output: int = 0


def usage_of(response: Any) -> Usage:
    """Tokens billed for one chat completion. Reasoning ("thinking") tokens are billed as output;
    some OpenAI-compatible endpoints leave them out of completion_tokens but count them in
    total_tokens, so output is whichever is larger."""
    u = getattr(response, "usage", None)
    if u is None:
        return Usage()
    prompt = int(getattr(u, "prompt_tokens", 0) or 0)
    completion = int(getattr(u, "completion_tokens", 0) or 0)
    total = int(getattr(u, "total_tokens", 0) or 0)
    return Usage(prompt, max(completion, total - prompt))


class Prices:
    def __init__(self, overrides: dict[str, tuple[float, float]] | None = None) -> None:
        self._table = {**PRICES, **(overrides or {})}

    def check(self, ref: ModelRef) -> None:
        if ref.provider != "fake" and ref.name not in self._table:
            raise ValueError(f"no price for {ref}; add it to PRICES or pass prices=")

    def cost(self, ref: ModelRef, usage: Usage) -> float:
        if ref.name not in self._table:
            if ref.provider == "fake":
                return 0.0
            raise ValueError(f"no price for {ref}")
        price_in, price_out = self._table[ref.name]
        return (usage.input * price_in + usage.output * price_out) / 1_000_000


class BudgetExceeded(Exception):
    pass


class Budget:
    """The run's hard dollar cap, shared by every worker. Calls already in flight finish and are
    charged; no call starts once the cap is reached."""

    def __init__(self, cap_usd: float) -> None:
        self.cap = cap_usd
        self.spent = 0.0
        self._lock = threading.Lock()

    def check(self) -> None:
        if self.exhausted:
            raise BudgetExceeded(f"budget ${self.cap:.2f} reached (spent ${self.spent:.4f})")

    def charge(self, usd: float) -> None:
        with self._lock:
            self.spent += usd

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self.spent >= self.cap


def with_retries(call: Callable[[], Any], delays: tuple[float, ...] = RETRY_DELAYS) -> Any:
    """Retry rate limits, timeouts and provider 5xx; anything else raises at once."""
    import openai

    retryable = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
    )
    for delay in delays:
        try:
            return call()
        except retryable:
            time.sleep(delay)
    return call()
