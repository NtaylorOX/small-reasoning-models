"""Pure summary statistics for benchmark runs."""

from __future__ import annotations

import math
from collections.abc import Sequence


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (same as numpy's default), q in [0, 100]."""
    if not values:
        raise ValueError("percentile of empty sequence")
    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100]")
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize_latencies(latencies_s: Sequence[float]) -> dict[str, float]:
    return {
        "p50_s": percentile(latencies_s, 50),
        "p90_s": percentile(latencies_s, 90),
        "p99_s": percentile(latencies_s, 99),
        "mean_s": sum(latencies_s) / len(latencies_s),
    }


def throughput(tokens: int, seconds: float) -> float:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    return tokens / seconds
