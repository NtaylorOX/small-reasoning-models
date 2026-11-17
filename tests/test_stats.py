import pytest

from lab.stats import percentile, summarize_latencies, throughput


def test_percentile_matches_linear_interpolation() -> None:
    xs = [4.0, 1.0, 3.0, 2.0]
    assert percentile(xs, 0) == 1.0
    assert percentile(xs, 100) == 4.0
    assert percentile(xs, 50) == 2.5
    assert percentile([7.0], 99) == 7.0


def test_summary_and_throughput() -> None:
    s = summarize_latencies([1.0, 2.0, 3.0])
    assert s["p50_s"] == 2.0 and s["mean_s"] == 2.0
    assert throughput(100, 2.0) == 50.0
    with pytest.raises(ValueError):
        throughput(1, 0)
    with pytest.raises(ValueError):
        percentile([], 50)
