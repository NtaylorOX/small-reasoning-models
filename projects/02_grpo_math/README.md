# Project 2: GRPO on math with verifiable rewards (the core baseline)

**Status: stub. Nothing implemented or run.**

**Goal:** a clean, reproducible RLVR baseline on a small model that matches a published number,
so every later experiment has a trusted starting point.

**Question:** how much does GRPO improve pass@1 for a 0.5-1.7B model on GSM8K, MATH-500 and
AIME/AMC, and what does it cost in GPU-hours on the hardware in this repo?

## Plan

1. **Evaluate first.** Base-model pass@1 and pass@k (k up to 64-256) on GSM8K test, MATH-500,
   AMC23 and AIME24/25 with vLLM sampling. Use one eval script for every later run.
2. **Reproduce a known small-model result** before changing anything. Candidates (code confirmed
   to exist on GitHub, numbers to check in each paper):
   - [TinyZero](https://github.com/Jiayi-Pan/TinyZero): R1-Zero on Countdown, cheapest "aha" reproduction
   - [SimpleRL-Zoo](https://github.com/hkust-nlp/simpleRL-reason): zero-RL across many small base models
   - [Open-RS](https://github.com/knoveleng/open-rs): RL for 1.5B reasoning on a tight budget
   - [understand-r1-zero (Dr. GRPO)](https://github.com/sail-sg/understand-r1-zero): fixes GRPO's length and std biases
3. **Train** with TRL `GRPOTrainer` and vLLM rollouts. Rewards in a pure, unit-tested `rewards.py`:
   exact match on the boxed/final answer (use a math verifier such as `math-verify`), optional
   format reward.
4. **Data ladder:** GSM8K → MATH (levels 3-5) → DAPO-Math-17k or DeepScaleR (see
   [docs/resources.md](../../docs/resources.md)).
5. Log reward, response length, entropy and KL per step; report test numbers only from full runs.

## Models

- Primary: `Qwen2.5-Math-1.5B` (most-used zero-RL baseline) or the newest small Qwen3.x model.
- **Control:** a non-Qwen model (Llama-3.2-1B, OLMo-2-1B or similar). Qwen2.5-Math improves even
  under random or spurious rewards ([Spurious Rewards](https://github.com/ruixin31/Spurious_Rewards)),
  so a Qwen-only gain is not evidence that the reward did the work.

## 1 GPU → N GPUs

1 GPU: vLLM colocated with training (`vllm_mode="colocate"` in recent TRL; check flag names for the
installed version). N GPUs: colocate across all, or one GPU for `trl vllm-serve` and N-1 training via
`accelerate launch` with a config from [extras/distributed](../extras/distributed/). Use LoRA on T4s.
