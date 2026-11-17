"""Config loading and results logging."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config, then apply `key=value` overrides (values parsed as YAML),
    e.g. `tensor_parallel_size=2` or `batch_sizes=[1,8,32]`."""
    with open(path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} must look like key=value")
        cfg[key.strip()] = yaml.safe_load(raw)
    return cfg


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def environment_info() -> dict[str, Any]:
    """GPU and library versions, so every result row says what it ran on."""
    info: dict[str, Any] = {"python": platform.python_version(), "git_commit": git_commit()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(n)]
    except ImportError:
        info["torch"] = None
    for lib in ("vllm", "transformers", "trl", "deepspeed"):
        try:
            mod = __import__(lib)
            info[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return info


def append_result(path: str | Path, row: dict[str, Any]) -> None:
    """Append one JSON line. Results files are append-only so runs are never overwritten."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **row}
    with p.open("a") as f:
        f.write(json.dumps(row) + "\n")
