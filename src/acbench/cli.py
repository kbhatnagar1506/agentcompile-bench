"""acbench run | score | gate | tasks"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .arms import ARMS, SdkSettings
from .runner import RunConfig, run
from .scorecard import render
from .scorer import failures, score
from .suites import SUITES, get_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="acbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run a suite through the arms, then score it")
    p.add_argument("--suite", required=True, choices=SUITES)
    p.add_argument("--agent", required=True, help="agent model, provider/model")
    p.add_argument("--customer", help="customer model, provider/model (simulated customers)")
    p.add_argument("--arms", default="baseline,sdk", help=f"comma list of {', '.join(ARMS)}")
    p.add_argument("--split", choices=["train", "dev", "heldout"])
    p.add_argument("--tasks", help="comma list of task ids (default: all in the split)")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--budget", type=float, default=5.0, help="hard cap in USD (default 5)")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--run-id")
    p.add_argument("--ac-key", help="AgentCompile key (default AGENTCOMPILE_KEY)")
    p.add_argument("--ac-url", help="decision service URL (default AGENTCOMPILE_URL or hosted)")
    p.add_argument("--ac-timeout", type=float, default=2.0)

    s = sub.add_parser("score", help="score a finished run and write its scorecard")
    s.add_argument("run_dir", type=Path)
    s.add_argument("--json", action="store_true", help="print the scorecard as JSON")

    g = sub.add_parser("gate", help="exit 1 if any capability failed its criterion")
    g.add_argument("run_dir", type=Path)

    t = sub.add_parser("tasks", help="list a suite's tasks")
    t.add_argument("--suite", required=True, choices=SUITES)
    t.add_argument("--split", choices=["train", "dev", "heldout"])

    args = parser.parse_args(argv)
    if args.command == "run":
        config = RunConfig(
            suite=args.suite,
            agent_model=args.agent,
            customer_model=args.customer,
            arms=[a.strip() for a in args.arms.split(",") if a.strip()],
            split=args.split,
            task_ids=[x.strip() for x in args.tasks.split(",")] if args.tasks else None,
            trials=args.trials,
            workers=args.workers,
            budget_usd=args.budget,
            max_steps=args.max_steps,
            out_dir=args.out,
            run_id=args.run_id,
        )
        sdk = SdkSettings(key=args.ac_key, base_url=args.ac_url, timeout=args.ac_timeout)
        run_dir = run(config, sdk)
        print(_write_scorecard(run_dir))
        print(f"\nrun: {run_dir}")
        return 0
    if args.command == "score":
        card = score(args.run_dir)
        if args.json:
            print(json.dumps(card, indent=2, default=str))
        else:
            print(_write_scorecard(args.run_dir, card))
        return 0
    if args.command == "gate":
        failed = failures(score(args.run_dir))
        for r in failed:
            print(f"FAIL {r['id']} {r['name']} ({r['arm']}): {r['criterion']}", file=sys.stderr)
        return 1 if failed else 0
    if args.command == "tasks":
        for task in get_suite(args.suite).select(args.split):
            print(f"{task.id}\t{task.split}\t{task.job or '-'}\t{','.join(task.tags)}")
        return 0
    return 2


def _write_scorecard(run_dir: Path, card: dict | None = None) -> str:
    card = card or score(run_dir)
    (run_dir / "scorecard.json").write_text(json.dumps(card, indent=2, default=str))
    text = render(card)
    (run_dir / "scorecard.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    sys.exit(main())
