"""τ-bench retail adapter. Needs `pip install -e .[tau]`; makes no model calls."""

from __future__ import annotations

import pytest

pytest.importorskip("tau_bench")

from acbench.suites.tau import DEV_TEST_TASKS, TauRetail  # noqa: E402


@pytest.fixture(scope="module")
def suite():
    return TauRetail()


def test_splits(suite):
    assert len(suite.select("dev")) == DEV_TEST_TASKS
    assert len(suite.select("heldout")) == 115 - DEV_TEST_TASKS
    assert len(suite.select("train")) == 500
    assert all("C3" in t.tags for t in suite.select("heldout"))


def test_replaying_the_expected_actions_succeeds(suite):
    for task in suite.select("dev")[:10]:
        env = suite.environment(task)
        replies = []
        for action in task.expected["actions"]:
            if action["name"] == "respond":
                replies.append(action["kwargs"]["content"])
            elif action["name"] not in env.stop_tools:
                env.call(action["name"], action["kwargs"])
        replies += task.expected["outputs"]
        assert env.outcome(replies).success, task.id


def test_doing_nothing_fails_any_task_that_expects_a_write(suite):
    writers = [t for t in suite.select("dev") if "C5" in t.tags]
    assert writers
    for task in writers[:5]:
        assert not suite.environment(task).outcome([]).success, task.id


def test_tool_errors_are_observations_not_exceptions(suite):
    env = suite.environment(suite.select("dev")[0])
    assert env.call("get_order_details", {"order_id": "#nope"}).startswith("Error")
    assert env.call("get_order_details", {"bad_arg": 1}).startswith("Error")
    assert env.call("no_such_tool", {}).startswith("Unknown action")
