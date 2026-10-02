"""The scorer: turns one run's trace into every metric and capability verdict.

It reads only manifest.json and trace.jsonl, so any run can be re-scored later. Every
comparison is paired by task against the baseline arm (stats.py).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import stats, trace
from .tasks import CAPABILITIES

C1_MIN_CUT = 0.30
C12_MAX_DECIDE_MS = 80.0

# What later milestones add; listed so the scorecard shows every capability, measured or not.
NOT_YET = {
    "C4": "near-miss suite (milestone 3)",
    "C5": "consent audit (milestone 3)",
    "C6": "consent audit (milestone 3)",
    "C9": "fail-open chaos arms (milestone 3)",
    "C10": "provider matrix (milestone 4)",
    "C11": "repetition world timeline (milestone 5)",
    "C13": "reply quality judge (milestone 3)",
    "C14": "second domain (milestone 6)",
    "C15": "recipe authoring suite (milestone 7)",
}


def score(run_dir: Path, *, resamples: int = stats.RESAMPLES) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    events = list(trace.read(run_dir / "trace.jsonl"))
    convs = [e for e in events if e["kind"] == "conversation" and e["status"] != "budget"]
    arms = list(manifest["config"]["arms"])
    k = int(manifest["config"]["trials"])

    by_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for c in convs:
        by_task[c["task"]][c["arm"]].append(c)
    split_of = {c["task"]: c["split"] for c in convs}

    summaries = {arm: _summary(arm, convs, events, by_task, k) for arm in arms}
    comparisons = {}
    if "baseline" in arms:
        for arm in arms:
            if arm != "baseline":
                comparisons[arm] = _compare(arm, by_task, split_of, events, k, resamples)
    return {
        "run": manifest["run"],
        "suite": manifest["config"]["suite"],
        "agent_model": manifest["config"]["agent_model"],
        "customer_model": manifest["config"]["customer_model"],
        "trials": k,
        "spent_usd": manifest.get("spent_usd"),
        "budget_usd": manifest["config"]["budget_usd"],
        "budget_hit": manifest.get("budget_hit"),
        "conversations": manifest.get("conversations"),
        "conversations_planned": manifest.get("conversations_planned"),
        "arms": summaries,
        "comparisons": comparisons,
        "capabilities": _capabilities(arms, summaries, comparisons, k),
    }


def _summary(
    arm: str,
    convs: list[dict[str, Any]],
    events: list[dict[str, Any]],
    by_task: dict[str, dict[str, list[dict[str, Any]]]],
    k: int,
) -> dict[str, Any]:
    mine = [c for c in convs if c["arm"] == arm]
    calls = [e for e in events if e["kind"] == "agent_call" and e["arm"] == arm]
    replies = [e for e in events if e["kind"] == "reply" and e["arm"] == arm]
    routes: dict[str, int] = defaultdict(int)
    for e in calls:
        routes[e["route"]] += 1
    writes = {"agent": [0, 0], "compiled": [0, 0]}
    for c in mine:
        for by, (sent, accepted) in c.get("writes", {}).items():
            writes[by][0] += sent
            writes[by][1] += accepted
    full = [runs[arm] for runs in by_task.values() if len(runs.get(arm, [])) == k]
    turn_ms = [r["turn_ms"] for r in replies]
    decide_ms = [e["decide_ms"] for e in calls if e.get("decide_ms") is not None]
    n = len(mine)

    def per_conv(field: str) -> float:
        return stats.mean([c.get(field, 0) for c in mine])

    return {
        "conversations": n,
        "errors": sum(c["status"] == "error" for c in mine),
        "success": stats.mean([bool(c["success"]) for c in mine]),
        "pass_k": stats.mean([all(c["success"] for c in runs) for runs in full]) if full else None,
        "agent_calls": per_conv("agent_calls"),
        "compiled_calls": per_conv("compiled_calls"),
        "input_tokens": per_conv("input_tokens"),
        "output_tokens": per_conv("output_tokens"),
        "agent_cost_usd": per_conv("agent_cost_usd"),
        "customer_cost_usd": per_conv("customer_cost_usd"),
        "ac_cost_usd": per_conv("ac_cost_usd"),
        "compiled_share": stats.mean(
            [c.get("agent_calls", 0) == 0 and c.get("compiled_calls", 0) > 0 for c in mine]
        ),
        "routes": dict(routes),
        "runtime_writes": {"sent": writes["compiled"][0], "accepted": writes["compiled"][1]},
        "agent_writes": {"sent": writes["agent"][0], "accepted": writes["agent"][1]},
        "turn_ms_p50": stats.percentile(turn_ms, 50),
        "turn_ms_p95": stats.percentile(turn_ms, 95),
        "decide_ms_p50": stats.percentile(decide_ms, 50) if decide_ms else None,
        "decide_ms_p95": stats.percentile(decide_ms, 95) if decide_ms else None,
    }


def _compare(
    arm: str,
    by_task: dict[str, dict[str, list[dict[str, Any]]]],
    split_of: dict[str, str],
    events: list[dict[str, Any]],
    k: int,
    resamples: int,
) -> dict[str, Any]:
    paired = [t for t, runs in by_task.items() if runs.get(arm) and runs.get("baseline")]

    def units(field: str, tasks: Sequence[str]) -> list[tuple[float, float]]:
        return [
            (
                stats.mean([float(c.get(field) or 0) for c in by_task[t][arm]]),
                stats.mean([float(c.get(field) or 0) for c in by_task[t]["baseline"]]),
            )
            for t in tasks
        ]

    def bounded(u: Sequence[Any], stat: Callable[[Sequence[Any]], float]) -> dict[str, Any]:
        if len(u) < 2:
            return {"value": stat(u) if u else None, "lower": None, "tasks": len(u)}
        return {
            "value": stat(u),
            "lower": stats.lower_bound(u, stat, resamples=resamples),
            "tasks": len(u),
        }

    learned = [t for t in paired if split_of[t] != "heldout"]
    heldout = [t for t in paired if split_of[t] == "heldout"]
    full = [t for t in paired if len(by_task[t][arm]) == k and len(by_task[t]["baseline"]) == k]
    pass_k_units = [
        (float(all(c["success"] for c in by_task[t][arm])),
         float(all(c["success"] for c in by_task[t]["baseline"])))
        for t in full
    ]
    turns: dict[tuple[str, str], list[float]] = defaultdict(list)
    for e in events:
        if e["kind"] == "reply" and e["arm"] in (arm, "baseline"):
            turns[(e["task"], e["arm"])].append(e["turn_ms"])
    latency_units = [(turns[(t, arm)], turns[(t, "baseline")]) for t in paired]

    return {
        "tasks": len(paired),
        "success_diff": bounded(units("success", learned), stats.mean_diff),
        "success_diff_heldout": bounded(units("success", heldout), stats.mean_diff),
        "pass_k_diff": bounded(pass_k_units, stats.mean_diff) if k >= 2 else None,
        "agent_calls_cut": bounded(units("agent_calls", paired), stats.ratio_cut),
        "tokens_cut": bounded(
            [(a1 + a2, b1 + b2) for (a1, b1), (a2, b2) in
             zip(units("input_tokens", paired), units("output_tokens", paired), strict=True)],
            stats.ratio_cut,
        ),
        "agent_cost_cut": bounded(units("agent_cost_usd", paired), stats.ratio_cut),
        # Positive: the arm answers faster. Bootstrapped over tasks, p50 of all turns pooled.
        "turn_p50_saved_ms": bounded(latency_units, _p50_saved),
    }


def _p50_saved(units: Sequence[tuple[list[float], list[float]]]) -> float:
    with_arm = [x for a, _ in units for x in a]
    without = [x for _, b in units for x in b]
    return stats.percentile(without, 50) - stats.percentile(with_arm, 50)


def _capabilities(
    arms: list[str],
    summaries: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def row(cap: str, arm: str | None, verdict: str, number: Any = None, lower: Any = None,
            criterion: str = "", note: str = "") -> None:
        rows.append({"id": cap, "name": CAPABILITIES[cap], "arm": arm, "verdict": verdict,
                     "value": number, "lower": lower, "criterion": criterion, "note": note})

    def parity(cap: str, arm: str, m: dict[str, Any] | None, criterion: str) -> None:
        if not m or m["tasks"] == 0:
            row(cap, arm, "n/a", criterion=criterion, note="no tasks in this split")
        elif m["lower"] is None:
            row(cap, arm, "n/a", m["value"], criterion=criterion, note="needs 2+ paired tasks")
        else:
            verdict = "pass" if m["lower"] >= stats.PARITY_MARGIN else "fail"
            row(cap, arm, verdict, m["value"], m["lower"], criterion, f"{m['tasks']} tasks")

    for arm, c in comparisons.items():
        cut = c["agent_calls_cut"]
        if arm == "sdk":
            if cut["lower"] is None:
                row("C1", arm, "n/a", cut["value"], criterion="calls cut ≥ 30% (lower bound)",
                    note="needs 2+ paired tasks")
            else:
                row("C1", arm, "pass" if cut["lower"] >= C1_MIN_CUT else "fail", cut["value"],
                    cut["lower"], "calls cut ≥ 30% (lower bound)",
                    f"tokens cut {_pct(c['tokens_cut']['value'])}, "
                    f"agent cost cut {_pct(c['agent_cost_cut']['value'])}")
        parity("C2", arm, c["success_diff"], "success diff lower bound ≥ −5 pts")
        parity("C3", arm, c["success_diff_heldout"], "same as C2, held-out tasks")
        if k >= 2:
            parity("C7", arm, c["pass_k_diff"], f"pass^{k} diff lower bound ≥ −5 pts")
        else:
            row("C7", arm, "n/a", criterion="pass^k diff lower bound ≥ −5 pts",
                note="needs --trials 2 or more")
        saved = c["turn_p50_saved_ms"]
        faster = saved["lower"] is not None and saved["lower"] > 0
        row("C8", arm, "report", saved["value"], saved["lower"],
            "claim faster only if p50 saved has lower bound > 0",
            ("faster" if faster else "faster not shown") + f"; p50/p95 "
            f"{_ms(summaries[arm]['turn_ms_p50'])}/{_ms(summaries[arm]['turn_ms_p95'])} vs "
            f"{_ms(summaries['baseline']['turn_ms_p50'])}/{_ms(summaries['baseline']['turn_ms_p95'])}")
    for arm in arms:
        if arm == "baseline":
            continue
        p50 = summaries[arm]["decide_ms_p50"]
        note = f"our cost ${summaries[arm]['ac_cost_usd']:.5f}/conv"
        if p50 is None:
            row("C12", arm, "n/a", criterion="decision p50 ≤ 80 ms", note=note)
        else:
            row("C12", arm, "pass" if p50 <= C12_MAX_DECIDE_MS else "fail", p50, None,
                "decision p50 ≤ 80 ms", note)
    for cap, why in NOT_YET.items():
        row(cap, None, "not measured", note=why)
    order = list(CAPABILITIES)
    rows.sort(key=lambda r: (order.index(r["id"]), r["arm"] or ""))
    return rows


def failures(card: dict[str, Any]) -> list[dict[str, Any]]:
    """Capabilities that failed their criterion: the CI gate fails the build on any of these."""
    return [r for r in card["capabilities"] if r["verdict"] == "fail"]


def _pct(x: float | None) -> str:
    return "–" if x is None or x != x else f"{x * 100:.1f}%"


def _ms(x: float | None) -> str:
    return "–" if x is None or x != x else f"{x:.0f} ms"
