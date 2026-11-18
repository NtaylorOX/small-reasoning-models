"""Fixed-step training loop for the scaling benchmark (stub).

Runs unchanged under `python`, `torchrun` and `accelerate launch`; the launcher picks the
strategy. To implement: model + synthetic batches, timed steps after warm-up, then log
tokens/s, peak memory and MFU per rank-0 to the results file.
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
    raise NotImplementedError(f"scaling benchmark not written yet; config: {cfg}")


if __name__ == "__main__":
    main()
