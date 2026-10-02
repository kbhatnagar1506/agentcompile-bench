"""τ-bench retail (Sierra, MIT), installed at a pinned commit with `pip install -e .[tau]`.

We take τ's database, tools, policy (wiki) and tasks, and run our own agent loop, customer and
scoring around them: τ's own Env starts a litellm customer call when it is built, which we
could neither wrap, price nor cache. Scoring is τ's rule, reimplemented: the conversation
succeeds when the database ends exactly where replaying the task's expected actions on a fresh
copy leaves it, and every required output appears in one of the agent's replies.

Splits. τ's 115 test tasks are split by index: 0-39 dev, 40-114 held out. τ's 500 train tasks
are `train`, for AgentCompile to learn from; they are never scored as dev or held out.
"""

from __future__ import annotations

import functools
from typing import Any

from ..customer import Customer
from ..tasks import Task
from .base import Environment, Outcome, Suite

DEV_TEST_TASKS = 40

WRITE_TOOLS = frozenset(
    {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
    }
)
STOP_TOOLS = frozenset({"transfer_to_human_agents"})

# Job label from the write a task expects; tasks with several kinds of write are `multi`.
JOBS = {
    "cancel_pending_order": "cancel",
    "exchange_delivered_order_items": "exchange",
    "modify_pending_order_address": "change_address",
    "modify_pending_order_items": "modify_items",
    "modify_pending_order_payment": "change_payment",
    "modify_user_address": "change_user_address",
    "return_delivered_order_items": "return",
}


@functools.lru_cache(maxsize=1)
def _tau() -> dict[str, Any]:
    try:
        from tau_bench.envs.retail.data import load_data
        from tau_bench.envs.retail.tasks_test import TASKS_TEST
        from tau_bench.envs.retail.tasks_train import TASKS_TRAIN
        from tau_bench.envs.retail.tools import ALL_TOOLS
        from tau_bench.envs.retail.wiki import WIKI
    except ImportError as exc:
        raise RuntimeError("τ-bench is not installed: pip install -e '.[tau]'") from exc
    return {
        "load_data": load_data,
        "test": TASKS_TEST,
        "train": TASKS_TRAIN,
        "tools": {tool.get_info()["function"]["name"]: tool for tool in ALL_TOOLS},
        "wiki": WIKI,
    }


class TauRetail(Suite):
    name = "tau-retail"

    @functools.cached_property
    def _tasks(self) -> list[Task]:
        tau = _tau()
        tasks = []
        for i, raw in enumerate(tau["test"]):
            tasks.append(self._task(f"test-{i}", "dev" if i < DEV_TEST_TASKS else "heldout", raw))
        for i, raw in enumerate(tau["train"]):
            tasks.append(self._task(f"train-{i}", "train", raw))
        return tasks

    def tasks(self) -> list[Task]:
        return self._tasks

    def environment(self, task: Task) -> Environment:
        return TauEnv(task, self._end_state(task.id))

    def customer(self, task: Task, chat: Any) -> Customer:
        if chat is None:
            raise ValueError("tau-retail needs a customer model (--customer provider/model)")
        return super().customer(task, chat)

    def _task(self, task_id: str, split: str, raw: Any) -> Task:
        actions = [{"name": a.name, "kwargs": a.kwargs} for a in raw.actions]
        writes = [a["name"] for a in actions if a["name"] in WRITE_TOOLS]
        kinds = sorted({JOBS[w] for w in writes})
        job = "lookup" if not kinds else kinds[0] if len(kinds) == 1 else "multi"
        tags = ["C1", "C2", "C7"] + (["C3"] if split == "heldout" else [])
        if writes:
            tags += ["C5", "C6"]
        return Task(
            suite=self.name,
            id=task_id,
            instruction=raw.instruction,
            split=split,
            job=job,
            tags=tuple(tags),
            expected={"actions": actions, "outputs": list(raw.outputs), "user_id": raw.user_id},
        )

    @functools.cache  # noqa: B019 - one suite object per run
    def _end_state(self, task_id: str) -> dict[str, Any]:
        """The database after replaying the task's expected actions on a fresh copy."""
        tau = _tau()
        task = next(t for t in self._tasks if t.id == task_id)
        data = tau["load_data"]()
        for action in task.expected["actions"]:
            tool = tau["tools"].get(action["name"])
            if tool is not None and action["name"] not in STOP_TOOLS:
                tool.invoke(data=data, **action["kwargs"])
        return data


class TauEnv(Environment):
    write_tools = WRITE_TOOLS
    stop_tools = STOP_TOOLS

    def __init__(self, task: Task, end_state: dict[str, Any]) -> None:
        tau = _tau()
        self.task = task
        self.end_state = end_state
        self.data = tau["load_data"]()
        self._tools = tau["tools"]
        self.tools = [tool.get_info() for tool in self._tools.values()]
        self.policy = tau["wiki"]

    def call(self, name: str, args: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Unknown action {name}"
        try:
            return str(tool.invoke(data=self.data, **args))
        except Exception as exc:
            return f"Error: {exc}"

    def outcome(self, replies: list[str]) -> Outcome:
        db_ok = self.data == self.end_state
        said = [r.lower().replace(",", "") for r in replies]
        outputs = {o: any(o.lower() in r for r in said) for o in self.task.expected["outputs"]}
        return Outcome(db_ok and all(outputs.values()), {"db": db_ok, "outputs": outputs})
