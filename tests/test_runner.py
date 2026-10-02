from __future__ import annotations

import json

import pytest

from acbench import trace
from acbench.runner import RunConfig, run


def events(run_dir, kind=None):
    return [e for e in trace.read(run_dir / "trace.jsonl") if kind is None or e["kind"] == kind]


def test_baseline_solves_every_toy_task(toy_run):
    run_dir, _ = toy_run(arms=["baseline"])
    convs = events(run_dir, "conversation")
    assert len(convs) == 6
    assert all(c["status"] == "done" and c["success"] for c in convs), convs
    assert {e["route"] for e in events(run_dir, "agent_call")} == {"baseline"}


def test_every_event_is_valid_and_carries_its_context(toy_run):
    run_dir, _ = toy_run(arms=["baseline", "sdk"], trials=2)
    for e in events(run_dir):
        trace.validate(e)
        assert e["v"] == trace.VERSION
    convs = events(run_dir, "conversation")
    assert len(convs) == 6 * 2 * 2
    assert len({c["conversation"] for c in convs}) == len(convs)


def test_sdk_arm_compiles_the_status_job_and_forwards_the_rest(toy_run):
    run_dir, service = toy_run(arms=["baseline", "sdk"])
    convs = {(c["task"], c["arm"]): c for c in events(run_dir, "conversation")}
    assert all(c["success"] for c in convs.values())
    for task in ("status-a2", "status-b2", "status-c2"):
        assert convs[(task, "sdk")]["agent_calls"] == 0
        assert convs[(task, "sdk")]["compiled_calls"] == 2
        assert convs[(task, "baseline")]["agent_calls"] == 2
    for task in ("cancel-a1", "cancel-b1", "cancel-c1"):
        assert convs[(task, "sdk")]["compiled_calls"] == 0
        assert convs[(task, "sdk")]["agent_calls"] == convs[(task, "baseline")]["agent_calls"]
    sdk_calls = [e for e in events(run_dir, "agent_call") if e["arm"] == "sdk"]
    assert {e["route"] for e in sdk_calls} == {"compiled", "forwarded"}
    assert all(e["job"] == "order_status" for e in sdk_calls if e["route"] == "compiled")
    assert all(e["input_tokens"] == 0 for e in sdk_calls if e["route"] == "compiled")
    assert service.decisions == len(sdk_calls)
    replies = [e for e in events(run_dir, "reply") if e["arm"] == "sdk" and e["by"] == "compiled"]
    assert len(replies) == 3


def test_agentcompile_own_cost_is_counted(toy_run):
    run_dir, _ = toy_run(arms=["sdk"])
    convs = [c for c in events(run_dir, "conversation") if c["task"].startswith("status")]
    assert all(c["ac_cost_usd"] == pytest.approx(0.0002) for c in convs)


def test_shadow_decides_but_never_answers(toy_run):
    run_dir, service = toy_run(arms=["shadow"])
    calls = events(run_dir, "agent_call")
    assert service.decisions == len(calls) > 0
    assert "compiled" not in {e["route"] for e in calls}
    assert "shadow" in {e["route"] for e in calls}
    assert all(c["success"] and c["compiled_calls"] == 0 for c in events(run_dir, "conversation"))


@pytest.mark.parametrize("fault", ["down", "timeout", "http500", "malformed"])
def test_a_broken_decision_service_fails_open(toy_run, fault):
    run_dir, _ = toy_run(arms=["sdk"], fault=fault)
    assert {e["route"] for e in events(run_dir, "agent_call")} == {"fail-open"}
    assert all(c["success"] for c in events(run_dir, "conversation"))


def test_budget_cap_stops_the_run_and_reports_it(toy_run):
    run_dir, _ = toy_run(
        arms=["baseline"], trials=3, workers=1, budget_usd=0.01,
        prices={"toy": (100.0, 100.0)},  # ~$0.01 per call: the cap is hit within a few calls
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["budget_hit"] is True
    assert manifest["conversations"]["budget"] > 0
    assert manifest["spent_usd"] < 0.01 + 0.02  # at most the call in flight goes over


def test_sdk_arm_without_a_key_refuses_to_run(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTCOMPILE_KEY", raising=False)
    with pytest.raises(RuntimeError, match="AGENTCOMPILE_KEY"):
        run(RunConfig(suite="toy", agent_model="fake/toy", arms=["sdk"], out_dir=tmp_path))


def test_unpriced_model_refuses_to_run(tmp_path):
    with pytest.raises(ValueError, match="no price"):
        run(RunConfig(suite="toy", agent_model="openai/unpriced-model", arms=["baseline"],
                      out_dir=tmp_path))
