from __future__ import annotations

import pytest

from acbench import stats
from acbench.cli import main
from acbench.scorecard import render
from acbench.scorer import failures, score


def caps(card, arm="sdk"):
    return {r["id"]: r for r in card["capabilities"] if r["arm"] in (arm, None)}


def test_bootstrap_bound_sits_below_the_point_estimate():
    units = [(1.0, 0.0)] * 30 + [(0.0, 1.0)] * 10  # with-arm wins 30 tasks, loses 10
    point = stats.mean_diff(units)
    lower = stats.lower_bound(units, stats.mean_diff, resamples=2000)
    assert point == pytest.approx(0.5)
    assert 0.2 < lower < point


def test_bootstrap_of_identical_arms_is_exactly_zero():
    units = [(1.0, 1.0), (0.0, 0.0), (1.0, 1.0)]
    assert stats.lower_bound(units, stats.mean_diff, resamples=500) == 0.0


def test_ratio_cut_pools_over_tasks():
    assert stats.ratio_cut([(1, 4), (3, 4)]) == pytest.approx(0.5)


def test_percentile():
    assert stats.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert stats.percentile([5], 95) == 5


def test_toy_scorecard_shows_parity_and_the_call_cut(toy_run):
    run_dir, _ = toy_run(arms=["baseline", "sdk"], trials=2)
    card = score(run_dir, resamples=2000)
    sdk, base = card["arms"]["sdk"], card["arms"]["baseline"]
    assert sdk["success"] == base["success"] == 1.0
    assert sdk["pass_k"] == base["pass_k"] == 1.0
    assert sdk["agent_calls"] < base["agent_calls"]
    assert sdk["compiled_share"] == pytest.approx(0.5)  # the three status tasks of six

    cut = card["comparisons"]["sdk"]["agent_calls_cut"]
    # Status tasks: 2 calls -> 0. Cancel tasks: 4 calls -> 4. Pooled: 1 - 12/18.
    assert cut["value"] == pytest.approx(1 - 12 / 18)
    assert cut["lower"] <= cut["value"]

    c = caps(card)
    assert c["C2"]["verdict"] == "pass" and c["C2"]["value"] == 0
    assert c["C3"]["verdict"] == "pass" and c["C3"]["note"] == "2 tasks"  # held out apart
    assert c["C7"]["verdict"] == "pass"
    # A third of calls cut, but on 6 tasks the lower bound is far under 30%: C1 must not pass.
    assert c["C1"]["verdict"] == "fail" and c["C1"]["lower"] < 0.30
    assert c["C12"]["verdict"] == "pass"
    assert c["C4"]["verdict"] == "not measured"
    assert {r["id"] for r in card["capabilities"]} == {f"C{i}" for i in range(1, 16)}


def test_c1_fails_when_the_cut_bound_is_under_30_percent(toy_run):
    run_dir, _ = toy_run(arms=["baseline", "sdk"], task_ids=["status-a2", "cancel-a1",
                                                              "cancel-b1", "cancel-c1"])
    card = score(run_dir, resamples=2000)
    assert caps(card)["C1"]["verdict"] == "fail"
    assert "C1" in {r["id"] for r in failures(card)}


def test_fail_open_arm_keeps_success_parity(toy_run):
    run_dir, _ = toy_run(arms=["baseline", "sdk"], fault="down")
    card = score(run_dir, resamples=1000)
    c = caps(card)
    assert c["C2"]["verdict"] == "pass"
    assert card["arms"]["sdk"]["routes"] == {"fail-open": card["arms"]["baseline"]["routes"]
                                             ["baseline"]}


def test_scorecard_renders_every_capability(toy_run):
    run_dir, _ = toy_run(arms=["baseline", "sdk"])
    text = render(score(run_dir, resamples=500))
    for i in range(1, 16):
        assert f"| C{i} |" in text
    assert "Runtime writes" in text


def test_cli_score_and_gate(toy_run, capsys):
    run_dir, _ = toy_run(arms=["baseline", "sdk"])
    assert main(["score", str(run_dir)]) == 0
    assert (run_dir / "scorecard.md").exists() and (run_dir / "scorecard.json").exists()
    assert main(["gate", str(run_dir)]) == 1  # C1's bound on 6 tasks is under 30%
    assert "FAIL C1" in capsys.readouterr().err

    baseline_only, _ = toy_run(arms=["baseline"], run_id="baseline-only")
    assert main(["gate", str(baseline_only)]) == 0
