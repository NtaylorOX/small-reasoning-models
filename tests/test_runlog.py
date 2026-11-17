import json
from pathlib import Path

import pytest

from lab.runlog import append_result, load_config


def test_load_config_with_overrides(tmp_path: Path) -> None:
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("a: 1\nb: [1, 2]\n")
    cfg = load_config(cfg_file, ["a=2", "b=[4,8]", "c=true"])
    assert cfg == {"a": 2, "b": [4, 8], "c": True}
    with pytest.raises(ValueError):
        load_config(cfg_file, ["no_equals_sign"])


def test_append_result_is_append_only(tmp_path: Path) -> None:
    out = tmp_path / "r" / "x.jsonl"
    append_result(out, {"v": 1})
    append_result(out, {"v": 2})
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["v"] for r in rows] == [1, 2]
    assert "timestamp" in rows[0]
