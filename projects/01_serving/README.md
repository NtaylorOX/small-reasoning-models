# Project 1: Serving benchmark (vLLM vs HF `generate`)

**Status: starter code written, not yet run on a GPU. No results yet.**

**Question:** how much faster is vLLM than naive Hugging Face generation, and *why*? Which of
continuous batching, PagedAttention, prefix caching, quantization and tensor parallelism
matters at which batch size and prompt length?

**Why it comes first in an RLVR repo:** in GRPO-style training, rollout generation is often most of the wall-clock time
(measure the split yourself in project 2). The vLLM
engine benchmarked here is the same one TRL and verl use for rollouts, and the shared-prefix
experiment is exactly the GRPO case: G samples of one prompt share the whole prompt prefix.

**CV line to earn:** "Benchmarked vLLM vs HF on 2x T4: N.Nx output throughput at batch 64,
prefix caching cut p50 latency by N% on shared-prompt workloads; KV cache model within N% of
measured."

## What's here

| File | What it does |
|---|---|
| `bench.py` | Offline throughput and batch-latency sweep, `vllm` or `hf` backend, one JSON line per cell |
| `workload.py` | Seeded synthetic prompts of exact token length, optional shared prefix |
| `kv_calc.py` | KV cache memory calculator from any HF config (uses `lab.kv_cache`) |
| `configs/qwen2.5-0.5b_t4.yaml` | 1 GPU baseline sweep |
| `configs/qwen2.5-1.5b_tp2.yaml` | Tensor parallel across Kaggle's 2 T4s |
| `configs/prefix_cache.yaml` | Shared 1,536-token system prompt, prefix caching on vs off |

## Run it

```bash
# On the GPU machine, from the repo root
pip install -e ".[serve]"          # or: make install-serve

# Check the plan without a GPU
python projects/01_serving/bench.py --config projects/01_serving/configs/qwen2.5-0.5b_t4.yaml --dry-run

# 1 GPU: vLLM, then the HF baseline on the same cells
python projects/01_serving/bench.py --config projects/01_serving/configs/qwen2.5-0.5b_t4.yaml
python projects/01_serving/bench.py --config projects/01_serving/configs/qwen2.5-0.5b_t4.yaml \
    --set backend=hf --set 'batch_sizes=[1,4,16]'

# N GPUs: tensor parallelism
python projects/01_serving/bench.py --config projects/01_serving/configs/qwen2.5-1.5b_tp2.yaml
python projects/01_serving/bench.py --config projects/01_serving/configs/qwen2.5-1.5b_tp2.yaml \
    --set tensor_parallel_size=1      # the comparison point

# Prefix caching on vs off
python projects/01_serving/bench.py --config projects/01_serving/configs/prefix_cache.yaml
python projects/01_serving/bench.py --config projects/01_serving/configs/prefix_cache.yaml \
    --set enable_prefix_caching=false

# KV cache arithmetic, then compare with the "KV cache" / "GPU blocks" line vLLM logs at start-up
python projects/01_serving/kv_calc.py --model Qwen/Qwen2.5-1.5B-Instruct --tokens 4096 --batch 32 --tp 2
```

Results append to `results/01_serving.jsonl` with the full config, GPU names and library versions.

### On Kaggle

1. New notebook → Settings → Accelerator **GPU T4 x2**, Internet **on**.
2. In a cell: `!git clone https://github.com/NtaylorOX/small-reasoning-models && cd small-reasoning-models && pip install -e ".[serve]"`
   (a private repo needs a GitHub token stored as a Kaggle secret, or make the repo public first).
3. Run the commands above with `!`. Download `results/01_serving.jsonl` before the session ends.

Kaggle's preinstalled torch may conflict with the vLLM wheel; if so, install vLLM into a fresh
venv (`uv venv && uv pip install vllm`). T4s are compute capability 7.5: fp16 only, no bf16,
no FlashAttention-2, so vLLM falls back to another attention backend. Note which one it picks.

## Online serving latency (step 2 of this project)

`bench.py` measures offline batch throughput. Interviewers mostly ask about online serving:
time-to-first-token (TTFT), inter-token latency (ITL) and p99 under a request rate. Use vLLM's
own tooling for that rather than writing a load generator:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype half --max-model-len 4096 &
vllm bench serve --model Qwen/Qwen2.5-0.5B-Instruct --dataset-name random \
    --random-input-len 1024 --random-output-len 128 --num-prompts 500 --request-rate 8
```

Sweep `--request-rate` until p99 TTFT blows up; that knee is the headline number. Check the exact
flag names against `vllm bench serve --help` for the installed version (unverified here).

## Experiments to run, in order

1. Baseline: vLLM vs HF, batch {1, 4, 16, 64} × prompt {128, 1024}. Plot output tok/s vs batch.
2. KV cache: predict max concurrent sequences with `kv_calc.py`, compare with vLLM's log.
3. Prefix caching on/off with a shared system prompt.
4. Quantization: an AWQ or GPTQ checkpoint of the same model (`--set quantization=awq --set model=...`).
5. Tensor parallel: 1.5B on 1 vs 2 T4s. Expect little gain (PCIe, no NVLink); explain why.
6. Online: `vllm bench serve` request-rate sweep, TTFT/ITL p50/p99.
7. Optional: SGLang on the same workloads.

## Reading

- [PagedAttention paper](https://arxiv.org/abs/2309.06180), vLLM docs on prefix caching and continuous batching
- CS336 inference lecture
