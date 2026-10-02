"""Paired bootstrap over tasks (SPEC, Statistics).

Every comparison resamples tasks, never conversations: each unit carries one task's results in
both arms, so task difficulty cancels out. Bounds are one-sided at 95% from 10,000 resamples.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")

RESAMPLES = 10_000
ALPHA = 0.05
PARITY_MARGIN = -0.05  # "same" success: the lower bound of (with - without) is above -5 points


def lower_bound(
    units: Sequence[T],
    stat: Callable[[Sequence[T]], float],
    *,
    alpha: float = ALPHA,
    resamples: int = RESAMPLES,
    seed: int = 0,
) -> float:
    """One-sided (1 - alpha) lower bound of stat(units), by bootstrap over units."""
    if not units:
        raise ValueError("no units to resample")
    rng = random.Random(seed)
    n = len(units)
    samples = sorted(stat(rng.choices(units, k=n)) for _ in range(resamples))
    return samples[int(alpha * resamples)]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def mean_diff(units: Sequence[tuple[float, float]]) -> float:
    """Mean of (with - without) over tasks."""
    return mean([a - b for a, b in units])


def ratio_cut(units: Sequence[tuple[float, float]]) -> float:
    """1 - total(with) / total(without): the share removed, pooled over tasks."""
    without = sum(b for _, b in units)
    return 1 - sum(a for a, _ in units) / without if without else float("nan")
