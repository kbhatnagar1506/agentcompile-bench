"""The runner: fans out every (task, arm, trial) on a worker pool under one dollar cap.

Work is ordered trial by trial and task by task, with every arm of a task side by side, so the
arms run in the same window (latency is only comparable that way). When the cap is reached no
new conversation starts, conversations in flight end as `budget`, and the run reports what it
finished.
"""

from __future__ import annotations

import json
import platform
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import agentcompile

from . import __version__
from .agent import ReferenceAgent
from .arms import ARMS, SdkSettings, client_for
from .models import Budget, BudgetExceeded, ModelRef, Prices, make_client
from .recorder import Recorder
from .suites import get_suite
from .suites.base import Suite
from .tasks import Task
from .trace import TraceWriter


@dataclass
class RunConfig:
    suite: str
    agent_model: str
    customer_model: str | None = None  # None: the suite's customers are scripted
    arms: list[str] = field(default_factory=lambda: ["baseline", "sdk"])
    split: str | None = None
    task_ids: list[str] | None = None
    trials: int = 1
    workers: int = 4
    budget_usd: float = 5.0
    temperature: float = 0.0
    max_steps: int = 30
    out_dir: Path = Path("runs")
    run_id: str | None = None
    prices: dict[str, tuple[float, float]] | None = None


@dataclass(frozen=True)
class WorkItem:
    task: Task
    arm: str
    trial: int


def run(config: RunConfig, sdk: SdkSettings | None = None) -> Path:
    """Run everything the config names; returns the run directory (manifest + trace)."""
    sdk = sdk or SdkSettings()
    for arm in config.arms:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; known: {ARMS}")
    if any(arm != "baseline" for arm in config.arms):
        sdk.check()
    suite = get_suite(config.suite)
    tasks = suite.select(config.split, config.task_ids)
    if not tasks:
        raise ValueError("no tasks selected")
    agent_model = ModelRef.parse(config.agent_model)
    customer_model = ModelRef.parse(config.customer_model) if config.customer_model else None
    prices = Prices(config.prices)
    prices.check(agent_model)
    if customer_model:
        prices.check(customer_model)

    run_id = config.run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    run_dir = config.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    writer = TraceWriter(run_dir / "trace.jsonl")
    budget = Budget(config.budget_usd)
    items = [
        WorkItem(task, arm, trial)
        for trial in range(config.trials)
        for task in tasks
        for arm in config.arms
    ]
    manifest: dict[str, Any] = {
        "run": run_id,
        "config": {**asdict(config), "out_dir": str(config.out_dir)},
        "tasks": len(tasks),
        "conversations_planned": len(items),
        "versions": {
            "acbench": __version__,
            "agentcompile": agentcompile.__version__,
            "python": platform.python_version(),
        },
        "started": time.time(),
    }
    _write_manifest(run_dir, manifest)

    def work(item: WorkItem) -> str:
        return _conversation(
            item, suite, config, sdk, writer, budget, prices, agent_model, customer_model, run_id
        )

    with ThreadPoolExecutor(max_workers=max(1, config.workers)) as pool:
        statuses = list(pool.map(work, items))

    manifest.update(
        finished=time.time(),
        spent_usd=round(budget.spent, 6),
        budget_hit=budget.exhausted,
        conversations={s: statuses.count(s) for s in sorted(set(statuses))},
    )
    _write_manifest(run_dir, manifest)
    return run_dir


def _conversation(
    item: WorkItem,
    suite: Suite,
    config: RunConfig,
    sdk: SdkSettings,
    writer: TraceWriter,
    budget: Budget,
    prices: Prices,
    agent_model: ModelRef,
    customer_model: ModelRef | None,
    run_id: str,
) -> str:
    task = item.task
    conversation_id = f"{run_id}:{task.key}:{item.arm}:{item.trial}"
    context = {
        "run": run_id,
        "conversation": conversation_id,
        "suite": task.suite,
        "task": task.id,
        "split": task.split,
        "job": task.job,
        "tags": list(task.tags),
        "arm": item.arm,
        "agent_model": str(agent_model),
        "trial": item.trial,
    }
    rec = Recorder(context, writer, budget, prices, agent_model, customer_model)
    started = time.perf_counter()
    if budget.exhausted:
        rec.end(status="budget", ended="not_started", success=None, detail=None, wall_ms=0.0)
        return "budget"
    try:
        env = suite.environment(task)
        chat = rec.customer_chat(make_client(customer_model)) if customer_model else None
        customer = suite.customer(task, chat)
        client = client_for(item.arm, make_client(agent_model), rec.on_sdk_event, sdk)
        agent = ReferenceAgent(
            client, agent_model.name, temperature=config.temperature, max_steps=config.max_steps
        )
        with agentcompile.conversation(conversation_id):
            ended = agent.run(env, customer, rec)
        outcome = env.outcome(rec.replies)
    except BudgetExceeded:
        rec.end(
            status="budget", ended="budget", success=None, detail=None,
            wall_ms=(time.perf_counter() - started) * 1000,
        )
        return "budget"
    except Exception as exc:  # the agent or a provider broke: a failed conversation, kept
        rec.end(
            status="error", ended="error", success=False, detail=None,
            wall_ms=(time.perf_counter() - started) * 1000, error=f"{type(exc).__name__}: {exc}",
        )
        return "error"
    rec.end(
        status="done", ended=ended, success=outcome.success, detail=outcome.detail,
        wall_ms=(time.perf_counter() - started) * 1000,
    )
    return "done"


def _write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
