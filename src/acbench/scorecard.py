"""One page per run: every capability with its number, bound and verdict, then the metrics."""

from __future__ import annotations

from typing import Any

POINTS = {"C2", "C3", "C7"}  # differences in success rate
SHARES = {"C1"}  # share of agent calls removed


def render(card: dict[str, Any]) -> str:
    lines = [
        f"# Scorecard · {card['run']}",
        "",
        f"Suite `{card['suite']}` · agent `{card['agent_model']}` · customer "
        f"`{card['customer_model'] or 'scripted'}` · {card['trials']} trial(s) · spent "
        f"${card['spent_usd'] or 0:.4f} of ${card['budget_usd']:.2f}"
        + (" · **budget reached, run incomplete**" if card.get("budget_hit") else ""),
        "",
        f"Conversations: {_counts(card)}",
        "",
        "## Capabilities",
        "",
        "| # | Capability | Arm | Value | Lower bound (95%) | Criterion | Verdict | Note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in card["capabilities"]:
        fmt = _pts if r["id"] in POINTS else _pct if r["id"] in SHARES else _num
        lines.append(
            f"| {r['id']} | {r['name']} | {r['arm'] or '–'} | {fmt(r['value'])} | "
            f"{fmt(r['lower'])} | {r['criterion'] or '–'} | {_verdict(r['verdict'])} | "
            f"{r['note'] or ''} |"
        )
    arms = list(card["arms"])
    lines += ["", "## Metrics per arm", "", "| Metric | " + " | ".join(arms) + " |",
              "| --- |" + " --- |" * len(arms)]

    def metric_row(label: str, cells: list[str]) -> str:
        return f"| {label} | " + " | ".join(cells) + " |"

    for label, key, fmt in METRICS:
        lines.append(metric_row(label, [fmt(card["arms"][a].get(key)) for a in arms]))
    lines.append(metric_row("Routes", [_routes(card["arms"][a]["routes"]) for a in arms]))
    lines.append(
        "| Runtime writes (accepted / sent) | "
        + " | ".join(_writes(card["arms"][a]["runtime_writes"]) for a in arms) + " |"
    )
    if card["comparisons"]:
        lines += ["", "## Paired with baseline", "",
                  "| Arm | Tasks | Success diff | Held-out success diff | Agent calls cut | "
                  "Tokens cut | Agent cost cut |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for arm, c in card["comparisons"].items():
            lines.append(
                f"| {arm} | {c['tasks']} | {_bounded(c['success_diff'], _pts)} | "
                f"{_bounded(c['success_diff_heldout'], _pts)} | "
                f"{_bounded(c['agent_calls_cut'], _pct)} | {_bounded(c['tokens_cut'], _pct)} | "
                f"{_bounded(c['agent_cost_cut'], _pct)} |"
            )
        lines += ["", "Values are point estimates with the one-sided 95% lower bound in brackets "
                  "(10,000 bootstrap resamples over tasks). Calls cut and cost cut are reported "
                  "separately: removed calls are often the cheap ones."]
    return "\n".join(lines) + "\n"


def _num(x: Any) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "–"
    return f"{x:.4g}" if isinstance(x, float) else str(x)


def _pts(x: Any) -> str:
    return "–" if x is None or x != x else f"{x * 100:+.1f} pts"


def _pct(x: Any) -> str:
    return "–" if x is None or x != x else f"{x * 100:.1f}%"


def _share(x: Any) -> str:
    return "–" if x is None or x != x else f"{x * 100:.1f}%"


def _usd(x: Any) -> str:
    return "–" if x is None or x != x else f"${x:.5f}"


def _ms(x: Any) -> str:
    return "–" if x is None or x != x else f"{x:.0f} ms"


def _bounded(m: dict[str, Any] | None, fmt: Any) -> str:
    if not m or m.get("value") is None:
        return "–"
    return fmt(m["value"]) + (f" [{fmt(m['lower'])}]" if m.get("lower") is not None else "")


def _verdict(v: str) -> str:
    return {"pass": "✅ pass", "fail": "❌ fail"}.get(v, v)


def _routes(routes: dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(routes.items())) or "–"


def _writes(w: dict[str, int]) -> str:
    return f"{w['accepted']} / {w['sent']}"


def _counts(card: dict[str, Any]) -> str:
    done = card.get("conversations") or {}
    parts = [f"{v} {k}" for k, v in sorted(done.items())]
    return ", ".join(parts) + f" of {card.get('conversations_planned')} planned"


METRICS = [
    ("Conversations", "conversations", _num),
    ("Errors", "errors", _num),
    ("Task success", "success", _share),
    ("pass^k", "pass_k", _share),
    ("Agent calls / conv", "agent_calls", _num),
    ("Compiled calls / conv", "compiled_calls", _num),
    ("Compiled share (no agent call)", "compiled_share", _share),
    ("Input tokens / conv", "input_tokens", _num),
    ("Output tokens / conv", "output_tokens", _num),
    ("Agent cost / conv", "agent_cost_usd", _usd),
    ("AgentCompile cost / conv", "ac_cost_usd", _usd),
    ("Customer cost / conv", "customer_cost_usd", _usd),
    ("Turn p50", "turn_ms_p50", _ms),
    ("Turn p95", "turn_ms_p95", _ms),
    ("Decision p50", "decide_ms_p50", _ms),
    ("Decision p95", "decide_ms_p95", _ms),
]
