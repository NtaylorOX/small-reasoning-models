# Project 5: Small models in a coding harness, then multi-turn RL (capstone)

**Status: plan only.** Inspired by Tufa Labs'
[duck harness](https://tufalabs.ai/research/duck-harness/) ([code](https://github.com/Tufalabs/duck-harness)).

## What the duck harness shows

Tufa Labs won ARC-AGI-3 Milestone 1 (Kaggle) with a deliberately minimal harness: the model works
in a Python REPL where game observations are variables, inspects them with tool calls and helper
functions, acts, and gets fresh variables back. It sees both image and text views of the 64x64
grid and evicts the oldest messages to keep context short. Kaggle evaluation runs on one GPU, so
they used an open model (Qwen 3.6 27B FP8) and **no training**: the post describes no RL or
fine-tuning. Their own conclusion: solvability depends mostly on the model, cost mostly on the
harness. Open problems they name: context management and perception.

That gap is this project: **can RLVR make a small model better inside a harness like this?**

## Plan

1. **Harness-only baseline.** Run a small open model (1.5-8B) in a REPL harness on a cheap,
   verifiable multi-turn environment: Reasoning Gym ARC-style tasks or games, then the
   [ARC-AGI-3 agents API](https://github.com/arcprize/ARC-AGI-3-Agents). Measure success, actions
   per solve and tokens per solve.
2. **Harness ablations without training:** context eviction policy, text vs image observations,
   with and without helper functions. Cheap and directly answers Tufa's open questions.
3. **Multi-turn RL.** GRPO over whole episodes (trajectory-level reward, tool-call turns masked
   correctly) with `verifiers` multi-turn environments, TRL, or [rLLM](https://github.com/agentica-project/rllm).
4. **Transfer to triage-gym:** the same loop with simulated patients as the environment.

## Compute

Harness evals on 1-2 T4s for ≤4B models; multi-turn RL realistically needs a rented 1-2x A100/H100.
