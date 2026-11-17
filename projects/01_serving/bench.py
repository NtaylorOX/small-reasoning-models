"""Offline generation benchmark: vLLM vs Hugging Face `generate`.

For every (batch_size, prompt_len) in the config it generates exactly `output_len` tokens per
prompt (EOS ignored), repeats the batch `repeats` times after one warm-up, and appends one JSON
line per cell to the results file with throughput, batch latency percentiles and the environment.

Usage (1 GPU):
    python projects/01_serving/bench.py --config projects/01_serving/configs/qwen2.5-0.5b_t4.yaml
Two GPUs with tensor parallelism (vLLM only):
    python projects/01_serving/bench.py --config ... --set tensor_parallel_size=2
Override anything else the same way, e.g. --set backend=hf --set 'batch_sizes=[1,4]'.

This measures offline batch throughput. For online serving latency (TTFT, ITL, p99 under a
request rate) use `vllm serve` + `vllm bench serve`; see this project's README.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workload import synthetic_prompts  # noqa: E402

from lab.runlog import append_result, environment_info, load_config  # noqa: E402
from lab.seed import seed_everything  # noqa: E402
from lab.stats import summarize_latencies, throughput  # noqa: E402


class Backend(Protocol):
    vocab_size: int

    def generate(self, prompts: list[list[int]], output_len: int) -> None: ...


class VLLMBackend:
    def __init__(self, cfg: dict[str, Any]) -> None:
        from vllm import LLM, SamplingParams

        self._sampling_params = SamplingParams
        self.llm = LLM(
            model=cfg["model"],
            dtype=cfg.get("dtype", "half"),
            tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.9),
            max_model_len=cfg.get("max_model_len"),
            enable_prefix_caching=cfg.get("enable_prefix_caching", False),
            quantization=cfg.get("quantization"),
            seed=cfg.get("seed", 0),
        )
        self.vocab_size = self.llm.get_tokenizer().vocab_size

    def generate(self, prompts: list[list[int]], output_len: int) -> None:
        params = self._sampling_params(temperature=0.0, max_tokens=output_len, ignore_eos=True)
        inputs: Any = [{"prompt_token_ids": p} for p in prompts]
        self.llm.generate(inputs, params, use_tqdm=False)


class HFBackend:
    def __init__(self, cfg: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if cfg.get("tensor_parallel_size", 1) != 1:
            raise ValueError("the HF baseline is single-GPU; set tensor_parallel_size=1")
        self.torch = torch
        dtype = {"half": torch.float16, "float16": torch.float16, "bfloat16": torch.bfloat16}[
            cfg.get("dtype", "half")
        ]
        self.model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=dtype).to(
            "cuda"
        )
        self.model.eval()
        self.vocab_size = AutoTokenizer.from_pretrained(cfg["model"]).vocab_size

    def generate(self, prompts: list[list[int]], output_len: int) -> None:
        ids = self.torch.tensor(prompts, device="cuda")  # equal lengths, so no padding
        with self.torch.inference_mode():
            self.model.generate(
                input_ids=ids,
                attention_mask=self.torch.ones_like(ids),
                max_new_tokens=output_len,
                min_new_tokens=output_len,
                do_sample=False,
                pad_token_id=self.model.config.eos_token_id,
            )
        self.torch.cuda.synchronize()


def make_backend(cfg: dict[str, Any]) -> Backend:
    name = cfg.get("backend", "vllm")
    if name == "vllm":
        return VLLMBackend(cfg)
    if name == "hf":
        return HFBackend(cfg)
    raise ValueError(f"unknown backend {name!r}")


def run_cell(
    backend: Backend, cfg: dict[str, Any], batch_size: int, prompt_len: int
) -> dict[str, Any]:
    output_len = int(cfg["output_len"])
    repeats = int(cfg.get("repeats", 3))
    prompts = synthetic_prompts(
        num_prompts=batch_size,
        prompt_len=prompt_len,
        vocab_size=backend.vocab_size,
        seed=int(cfg.get("seed", 0)),
        shared_prefix_len=min(int(cfg.get("shared_prefix_len", 0)), prompt_len),
    )
    backend.generate(prompts, output_len)  # warm-up (CUDA graphs, allocator, prefix cache)
    latencies = []
    for _ in range(repeats):
        start = time.perf_counter()
        backend.generate(prompts, output_len)
        latencies.append(time.perf_counter() - start)
    total_s = sum(latencies)
    out_tokens = batch_size * output_len * repeats
    all_tokens = batch_size * (prompt_len + output_len) * repeats
    return {
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "output_len": output_len,
        "repeats": repeats,
        "output_tok_per_s": throughput(out_tokens, total_s),
        "total_tok_per_s": throughput(all_tokens, total_s),
        "batch_latency": summarize_latencies(latencies),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.set)
    cells = [(b, p) for p in cfg["prompt_lens"] for b in cfg["batch_sizes"]]
    print(f"backend={cfg.get('backend', 'vllm')} model={cfg['model']} cells={len(cells)}")
    if args.dry_run:
        for b, p in cells:
            print(f"  batch_size={b} prompt_len={p} output_len={cfg['output_len']}")
        return

    seed_everything(int(cfg.get("seed", 0)))
    backend = make_backend(cfg)
    env = environment_info()
    for b, p in cells:
        row = run_cell(backend, cfg, b, p)
        print(
            f"  bs={b:>4} prompt={p:>5}  out tok/s={row['output_tok_per_s']:>9.1f}"
            f"  p50 batch={row['batch_latency']['p50_s']:.3f}s"
        )
        append_result(
            cfg.get("results_file", "results/01_serving.jsonl"),
            {
                "config": cfg,
                "env": env,
                **row,
            },
        )


if __name__ == "__main__":
    main()
