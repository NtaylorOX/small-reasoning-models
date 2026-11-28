# small-reasoning-models

Training small (0.5-8B) reasoning models with **reinforcement learning from verifiable rewards
(RLVR)**: GRPO-style training on math and procedural tasks, careful ablations of what RL actually
changes, and multi-turn RL for models working inside a tool harness. The serving and multi-GPU
infrastructure that makes RL rollouts fast is part of the repo too.

Everything runs on **1 GPU** and scales to **N** by changing a config value or the launcher
(`tensor_parallel_size` for vLLM, `accelerate launch` / `torchrun --nproc_per_node N` for training).
The default target is Kaggle's free 2x T4, with short rented A100/H100 runs for headline numbers.

> Research and learning code. No result is claimed until it is in `results/` with the config and
> environment that produced it. Models trained here are research artefacts.

## Roadmap

| # | Project | Status | Output | Compute |
|---|---|---|---|---|
| 1 | [Rollout engine: vLLM vs HF, KV cache, prefix caching, TP](projects/01_serving/) | starter written, not run | throughput/latency table | Kaggle 2x T4 |
| 2 | [GRPO on math: reproduce a small-model baseline](projects/02_grpo_math/) | stub | pass@1/pass@k on MATH-500, AIME, AMC | T4 (LoRA), then ~$30 H100 |
| 3 | [What does RLVR do? pass@k, spurious rewards, GRPO variants](projects/03_rlvr_ablations/) | plan | preprint-style report | ~$40 H100 |
| 4 | [Procedural environments and curricula (Reasoning Gym)](projects/04_procedural_envs/) | plan | transfer results | T4 / A100 |
| 5 | [Small models in a coding harness, then multi-turn RL](projects/05_harness_agents/) | plan | ARC-style harness results; bridge to triage-gym | A100/H100 |
| extras | [SFT distillation, DPO](projects/extras/sft_distill/) · [DDP/FSDP/ZeRO scaling](projects/extras/distributed/) | stubs | supporting skills | as needed |

One project at a time, in order. Project 1 is free and makes every RL run after it cheaper to
reason about; project 2 is the trusted baseline everything else builds on.

**Direction.** Recent frontier work points at three things this roadmap mirrors at small scale:
RLVR recipes and their failure modes (DAPO, Dr. GRPO, pass@k and spurious-reward critiques),
procedurally generated environments instead of fixed datasets, and harness-based agentic reasoning
(e.g. Tufa Labs' [duck harness](https://tufalabs.ai/research/duck-harness/) for ARC-AGI-3, which
uses an untrained open model; project 5 asks whether RL helps inside such a harness).

Datasets, benchmarks, frameworks and papers: **[docs/resources.md](docs/resources.md)**.

**Parallel track: agent security and safety evals.** Red teaming (PAIR/TAP/GCG), refusal-direction
robustness, prompt injection against tool agents, and cyber evals, all on local models in an isolated
sandbox. Plan, courses, tools and projects S1-S6: **[docs/security-track.md](docs/security-track.md)**.
The hands-on code lives in a separate sandbox repo.

## Layout

```
src/lab/            shared helpers: KV cache maths, stats, config + results logging, seeding
projects/NN_*/      one directory per project: README, configs/, scripts
projects/extras/    supporting projects (SFT/DPO, distributed scaling)
docs/               resources and write-ups
results/            append-only JSONL results (committed); raw logs in results/raw/ (ignored)
tests/              offline unit tests (no GPU, no downloads)
```

## Setup

```bash
uv sync --extra dev                  # anywhere: lint, types, tests
uv sync --extra dev --extra serve    # GPU box, project 1
uv sync --extra dev --extra train    # GPU box, projects 2-5 (add --extra deepspeed for ZeRO)
make check                           # ruff, mypy --strict on src/, pytest
```

On Kaggle or Colab, `pip install -e ".[serve]"` works too. Keys go in `.env` (see `.env.example`).

## Compute notes

- **Kaggle**: 2x T4 (16 GB each), 12 h sessions, about 30 GPU h/week (user-reported, unverified).
  T4s are compute capability 7.5: fp16 only (no bf16), no FlashAttention-2, PCIe without NVLink.
  fp16 RL can be numerically fragile; use LoRA and small models here, and do headline runs in bf16.
- **Colab free / Lightning AI free**: one GPU, quick tests. **Modal Starter**: $30/month credit (unverified).
- **Rented**: Vast.ai / RunPod for 1-8 GPUs by the hour (H100 roughly $1.8-2.7/h, Oct-Nov 2026 third-party
  pages, unverified). Plan budget: about $100. Stop instances and download `results/` before shutting down.

## Conventions

- Seeds are explicit config values; every result row records config, git commit, GPUs and library versions.
- Evaluate base models with the same script before training; report test-set numbers only from full runs.
- `make check` passes before every push. Tests run offline: no GPU, no model downloads.
