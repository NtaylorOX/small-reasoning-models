# Project 4: GRPO for math reasoning (GSM8K, then MATH)

**Status: stub. Nothing implemented or run.**

**Question:** how much does GRPO improve pass@1 for Qwen2.5 0.5-1.5B on GSM8K and MATH, and how
do group size, KL coefficient and reward shaping (format vs correctness) change that?

## Plan

1. Baseline pass@1 on the GSM8K and MATH test sets (greedy, vLLM).
2. GRPO with TRL `GRPOTrainer`, vLLM for generation (`use_vllm: true`). Rewards to write in a pure, unit-tested `rewards.py`:
   exact-match correctness on the final number, plus an optional format reward.
3. Ablations: group size {4, 8, 16}, KL beta {0, 0.04}, correctness-only vs correctness+format.
4. Report pass@1 on the held-out test sets only from full logged runs; never tune on test.
5. Short preprint-style write-up.

## 1 GPU → N GPUs

1 GPU: generation and training share the GPU (vLLM colocate mode). On N GPUs, either colocate
on all of them or dedicate one GPU to a `trl vllm-serve` process and train on the rest:
`accelerate launch --num_processes N-1 ...`. Check the current TRL GRPO docs for the flag names
(unverified here; they changed across TRL releases).

Datasets to confirm on the Hub (unverified): `openai/gsm8k` (config `main`), a MATH split such as
`HuggingFaceH4/MATH-500` for eval.
