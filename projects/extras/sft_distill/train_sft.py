"""SFT training (stub). Implement with TRL once project 1 is done.

python projects/extras/sft_distill/train_sft.py \
    --config projects/extras/sft_distill/configs/sft.yaml
accelerate launch --num_processes N projects/extras/sft_distill/train_sft.py --config ...
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
    raise NotImplementedError(f"SFT training not written yet; config: {cfg}")


if __name__ == "__main__":
    main()
