import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "projects" / "01_serving"))

from workload import synthetic_prompts  # noqa: E402


def test_exact_lengths_and_determinism() -> None:
    a = synthetic_prompts(4, 50, vocab_size=32000, seed=1)
    b = synthetic_prompts(4, 50, vocab_size=32000, seed=1)
    assert a == b
    assert all(len(p) == 50 for p in a)
    assert synthetic_prompts(4, 50, vocab_size=32000, seed=2) != a


def test_shared_prefix() -> None:
    ps = synthetic_prompts(3, 20, vocab_size=32000, seed=0, shared_prefix_len=12)
    assert ps[0][:12] == ps[1][:12] == ps[2][:12]
    assert ps[0][12:] != ps[1][12:]
    with pytest.raises(ValueError):
        synthetic_prompts(1, 5, vocab_size=32000, seed=0, shared_prefix_len=6)


def test_bench_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    import bench

    cfg = Path(__file__).resolve().parents[1] / "projects/01_serving/configs/qwen2.5-0.5b_t4.yaml"
    bench.main(["--config", str(cfg), "--dry-run", "--set", "batch_sizes=[1,2]"])
    out = capsys.readouterr().out
    assert "cells=4" in out
