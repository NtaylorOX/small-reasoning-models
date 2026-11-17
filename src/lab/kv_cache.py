"""KV cache memory calculator.

bytes = 2 (K and V) x layers x kv_heads x head_dim x bytes_per_value x tokens

Grouped-query attention is why `kv_heads` is usually much smaller than the number of
query heads; that ratio is the main lever on KV memory for modern models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BYTES_PER_DTYPE: dict[str, float] = {
    "fp32": 4,
    "float32": 4,
    "fp16": 2,
    "float16": 2,
    "half": 2,
    "bf16": 2,
    "bfloat16": 2,
    "fp8": 1,
    "int8": 1,
}


@dataclass(frozen=True)
class AttentionShape:
    num_layers: int
    num_kv_heads: int
    head_dim: int

    @classmethod
    def from_hf_config(cls, config: Any) -> AttentionShape:
        """Read the shape from a Hugging Face `PretrainedConfig` (or any object with the same
        attributes). Falls back to hidden_size / num_attention_heads when head_dim is absent,
        and to num_attention_heads when the model has no GQA."""
        num_heads = int(config.num_attention_heads)
        kv_heads = getattr(config, "num_key_value_heads", None) or num_heads
        head_dim = getattr(config, "head_dim", None) or int(config.hidden_size) // num_heads
        return cls(int(config.num_hidden_layers), int(kv_heads), int(head_dim))


def kv_bytes_per_token(shape: AttentionShape, dtype: str = "fp16") -> float:
    if dtype not in BYTES_PER_DTYPE:
        raise ValueError(f"unknown dtype {dtype!r}; choose from {sorted(BYTES_PER_DTYPE)}")
    return 2 * shape.num_layers * shape.num_kv_heads * shape.head_dim * BYTES_PER_DTYPE[dtype]


def kv_cache_bytes(
    shape: AttentionShape, tokens: int, dtype: str = "fp16", tensor_parallel: int = 1
) -> float:
    """Total KV bytes for `tokens` cached tokens (summed over all sequences), per GPU.

    With tensor parallelism the KV heads are split across GPUs. If there are fewer KV heads
    than GPUs, vLLM replicates them, so each GPU holds at least one head.
    """
    if tokens < 0 or tensor_parallel < 1:
        raise ValueError("tokens must be >= 0 and tensor_parallel >= 1")
    per_gpu_heads = max(1, -(-shape.num_kv_heads // tensor_parallel))
    per_gpu = AttentionShape(shape.num_layers, per_gpu_heads, shape.head_dim)
    return kv_bytes_per_token(per_gpu, dtype) * tokens


def max_cached_tokens(
    shape: AttentionShape, free_bytes: float, dtype: str = "fp16", tensor_parallel: int = 1
) -> int:
    """How many tokens fit in `free_bytes` of KV cache memory per GPU."""
    per_token = kv_cache_bytes(shape, 1, dtype, tensor_parallel)
    return int(free_bytes // per_token)
