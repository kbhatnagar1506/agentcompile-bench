"""A tiny offline shop: two jobs (order status, cancel), scripted customers, no model needed.

It exists so the runner, trace, scorer and scorecard can be exercised end to end in CI for $0,
together with the fake agent model and fake decision service in acbench.testing.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from ..customer import Customer, ScriptedCustomer
from ..tasks import Task
from .base import Environment, Outcome, Suite

ORDERS: dict[str, dict[str, Any]] = {
    "A1": {"customer": "ana", "item": "mug", "status": "pending"},
    "A2": {"customer": "ana", "item": "lamp", "status": "delivered"},
    "B1": {"customer": "ben", "item": "kettle", "status": "pending"},
    "B2": {"customer": "ben", "item": "desk", "status": "shipped"},
    "C1": {"customer": "cy", "item": "chair", "status": "pending"},
    "C2": {"customer": "cy", "item": "rug", "status": "delivered"},
}

POLICY = """You are the support agent for a small shop.
- Look up an order with get_order before telling the customer anything about it.
- Only pending orders can be cancelled. Before calling cancel_order, explain what will happen
  and get an explicit yes from the customer.
- If you cannot help, call transfer_to_human."""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up an order by id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel a pending order. Needs the customer's explicit yes first.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human",
            "description": "Hand the conversation to a human agent.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    },
]

# (id, split, job, order, customer lines, statuses that change)
_TASKS = [
    ("status-a2", "dev", "order_status", "A2", ["Where is my order A2?"], {}),
    ("status-b2", "dev", "order_status", "B2", ["Hi, what's the status of order B2?"], {}),
    ("status-c2", "heldout", "order_status", "C2", ["Can you check order C2 for me?"], {}),
    ("cancel-a1", "dev", "cancel", "A1", ["Please cancel my order A1.", "Yes, go ahead."],
     {"A1": "cancelled"}),
    ("cancel-b1", "dev", "cancel", "B1", ["I want to cancel order B1.", "Yes please."],
     {"B1": "cancelled"}),
    ("cancel-c1", "heldout", "cancel", "C1", ["Cancel C1 please.", "yes"], {"C1": "cancelled"}),
]


class ToySuite(Suite):
    name = "toy"

    def tasks(self) -> list[Task]:
        tasks = []
        for task_id, split, job, order, lines, changes in _TASKS:
            end_state = copy.deepcopy(ORDERS)
            for order_id, status in changes.items():
                end_state[order_id]["status"] = status
            tags = ["C1", "C2", "C7"] + (["C3"] if split == "heldout" else [])
            if changes:
                tags += ["C5", "C6"]
            outputs = [] if changes else [ORDERS[order]["status"]]
            tasks.append(
                Task(
                    suite=self.name,
                    id=task_id,
                    instruction=" ".join(lines),
                    split=split,
                    job=job,
                    tags=tuple(tags),
                    expected={"orders": end_state, "outputs": outputs, "lines": lines},
                )
            )
        return tasks

    def environment(self, task: Task) -> Environment:
        return ToyEnv(task)

    def customer(self, task: Task, chat: Any) -> Customer:
        return ScriptedCustomer(task.expected["lines"])


class ToyEnv(Environment):
    policy = POLICY
    tools = TOOLS
    write_tools = frozenset({"cancel_order"})
    stop_tools = frozenset({"transfer_to_human"})

    def __init__(self, task: Task) -> None:
        self.task = task
        self.orders = copy.deepcopy(ORDERS)

    def call(self, name: str, args: dict[str, Any]) -> str:
        order = self.orders.get(str(args.get("order_id", "")))
        if name == "get_order":
            return json.dumps(order) if order else "Error: order not found"
        if name == "cancel_order":
            if order is None:
                return "Error: order not found"
            if order["status"] != "pending":
                return "Error: only pending orders can be cancelled"
            order["status"] = "cancelled"
            return json.dumps(order)
        if name == "transfer_to_human":
            return "Transferred."
        return f"Unknown action {name}"

    def outcome(self, replies: list[str]) -> Outcome:
        db_ok = self.orders == self.task.expected["orders"]
        said = " ".join(replies).lower()
        outputs = {o: o.lower() in said for o in self.task.expected["outputs"]}
        return Outcome(db_ok and all(outputs.values()), {"db": db_ok, "outputs": outputs})
