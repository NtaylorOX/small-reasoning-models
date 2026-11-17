"""KV cache memory calculator for any Hugging Face model.

    python projects/01_serving/kv_calc.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --tokens 4096 --batch 32 --dtype fp16 --tp 2

Compare its prediction with what vLLM logs at start-up ("# GPU blocks" / "KV cache size"),
then write down where the two differ and why.
"""

from __future__ import annotations

import argparse

from lab.kv_cache import AttentionShape, kv_bytes_per_token, kv_cache_bytes, max_cached_tokens

GIB = 1024**3


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="KV cache memory calculator")
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--tokens", type=int, default=4096, help="tokens per sequence")
    p.add_argument("--batch", type=int, default=1, help="concurrent sequences")
    p.add_argument("--dtype", default="fp16")
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs")
    p.add_argument("--free-gib", type=float, default=None, help="KV budget per GPU, in GiB")
    args = p.parse_args(argv)

    from transformers import AutoConfig

    shape = AttentionShape.from_hf_config(AutoConfig.from_pretrained(args.model))
    total = kv_cache_bytes(shape, args.tokens * args.batch, args.dtype, args.tp)
    print(f"{args.model}: {shape}")
    print(f"per token (all layers, 1 GPU): {kv_bytes_per_token(shape, args.dtype) / 1024:.1f} KiB")
    print(f"{args.batch} x {args.tokens} tokens, tp={args.tp}: {total / GIB:.3f} GiB per GPU")
    if args.free_gib is not None:
        n = max_cached_tokens(shape, args.free_gib * GIB, args.dtype, args.tp)
        print(f"{args.free_gib} GiB holds {n:,} tokens ({n // args.tokens} sequences)")


if __name__ == "__main__":
    main()
