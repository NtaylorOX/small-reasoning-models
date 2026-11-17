"""GRPO on GSM8K (stub). Implement with TRL's GRPOTrainer after projects 1-3.

python projects/04_grpo/train_grpo.py --config projects/04_grpo/configs/grpo.yaml
"""

from __future__ import annotations

import argparse

from lab.runlog import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    raise NotImplementedError(f"GRPO training not written yet; config: {cfg}")


if __name__ == "__main__":
    main()
