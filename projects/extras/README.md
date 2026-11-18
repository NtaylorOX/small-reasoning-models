# Extras: supporting skills

Not on the critical path, but each feeds the main projects and covers common interview topics.

- [`sft_distill/`](sft_distill/): SFT warm-start / distillation from reasoning traces (the R1 recipe's
  cold-start step), plus the original DPO and reward-model-vs-judge plan as an option.
- [`distributed/`](distributed/): DDP vs FSDP2 vs ZeRO-2/3 scaling benchmark. Its launcher configs are
  what projects 2-5 use to go from 1 to N GPUs.
