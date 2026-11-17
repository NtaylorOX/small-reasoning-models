# llm-post-training-lab

Four hands-on projects on LLM serving, post-training and multi-GPU training, built on small open
models (Qwen2.5 0.5-1.5B) and known-good public datasets. Each project ends with code, a results
table backed by logged runs, and a short write-up.

Everything runs on **1 GPU** and scales to **N GPUs** by changing a config value or the launcher
(`tensor_parallel_size` for serving, `accelerate launch` / `torchrun --nproc_per_node N` for training).
The default target is Kaggle's free 2x T4.

> Research and learning code. No result in this repo is claimed until it is in `results/` with the
> config and environment that produced it.

## Roadmap

| # | Project | Status | Weeks | Output | Compute |
|---|---|---|---|---|---|
| 1 | [Serving benchmark: vLLM vs HF, KV cache, prefix caching, TP](projects/01_serving/) | starter written, not run | 1-2 | Throughput/latency table + blog post | Kaggle 2x T4 (free) |
| 2 | [SFT then DPO, reward model vs LLM judge](projects/02_sft_dpo/) | stub | 3-6 | Model + eval report on the Hub | Kaggle, then ~$30 A100/H100 |
| 3 | [DDP vs FSDP2 vs ZeRO-2/3 scaling](projects/03_distributed/) | stub | 5-8 | Tokens/s, memory, MFU at 1/2/4/8 GPUs | Kaggle 2x T4, then ~$20 on 4-8 GPUs |
| 4 | [GRPO on GSM8K / MATH with ablations](projects/04_grpo/) | stub | 8-12 | Preprint-style write-up | ~$50 H100 |

Work one project at a time, in order. Project 1 closes the serving gap first and is free.

## Layout

```
src/lab/            shared helpers: KV cache maths, stats, config + results logging, seeding
projects/NN_*/      one directory per project: README, configs/, scripts
results/            append-only JSONL results (committed); raw logs go in results/raw/ (ignored)
tests/              offline unit tests (no GPU, no downloads)
```

## Setup

```bash
uv sync --extra dev                  # anywhere: lint, types, tests
uv sync --extra dev --extra serve    # GPU box, project 1
uv sync --extra dev --extra train    # GPU box, projects 2-4 (add --extra deepspeed for ZeRO)
make check                           # ruff, mypy --strict on src/, pytest
```

On Kaggle or Colab, `pip install -e ".[serve]"` works as well. Keys go in `.env` (see `.env.example`).

## Reading list

Starred items first.

**Serving**
- ★ [PagedAttention / vLLM paper](https://arxiv.org/abs/2309.06180), with the vLLM docs on prefix caching and continuous batching
- CS336 inference lecture ([playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_))
- Know by heart: KV bytes = 2 × layers × KV heads × head dim × bytes per value × tokens (`src/lab/kv_cache.py`)

**Distributed training**
- ★ [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (data parallel and ZeRO chapters first)
- ★ [Stanford CS336](https://stanford-cs336.github.io/spring2025/), assignment 2 (profiling, FlashAttention2 in Triton, DDP)
- [GPU MODE lectures](https://github.com/gpu-mode/lectures)
- Papers: [ZeRO](https://arxiv.org/abs/1910.02054), [Megatron-LM](https://arxiv.org/abs/1909.08053), [FlashAttention](https://arxiv.org/abs/2205.14135)

**Post-training**
- ★ [RLHF Book and course](https://rlhfbook.com/) by Nathan Lambert ([lectures](https://rlhfbook.com/course))
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/en/chapter12/1), chapters 11 (fine-tuning) and 12 (GRPO with TRL)
- Papers, in order: [InstructGPT](https://arxiv.org/abs/2203.02155), [DPO](https://arxiv.org/abs/2305.18290),
  [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300), [DeepSeek-R1](https://arxiv.org/abs/2501.12948),
  [Tulu 3](https://arxiv.org/abs/2411.15124)

## Compute notes

- **Kaggle Notebooks**: 2x T4 (16 GB each), 12 h sessions, about 30 GPU h/week (user-reported, unverified).
  The only free option with two GPUs in one machine, so enough for DDP/FSDP/TP at N=2.
- **T4 caveats**: compute capability 7.5, so fp16 only (no bf16), no FlashAttention-2, PCIe with no
  NVLink (expect weak tensor-parallel scaling). Keep `dtype: half` / `fp16: true` in configs.
- **Colab free / Lightning AI free**: single GPU, fine for quick tests.
- **Modal Starter**: $30/month free serverless credit (unverified), good for short A100/H100 runs.
- **Beyond 2 GPUs**: rent 4-8 GPUs for a few hours on Vast.ai or RunPod; Lambda for reliable 8x H100.
  Third-party prices from Oct-Nov 2026 (unverified): RTX 4090 ~$0.36-0.40/h on Vast.ai, H100 ~$1.8-2.7/h
  on Vast.ai/RunPod. Total budget for the plan: about $100.
- Always stop rented instances and download `results/` before a session ends.

## Conventions

- Seeds are explicit config values. Every result row records config, git commit, GPUs and library versions.
- `make check` passes before every push. Tests run offline: no GPU, no model downloads.
- Not medical, legal or production advice; models trained here are research artefacts.
