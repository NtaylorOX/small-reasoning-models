from types import SimpleNamespace

import pytest

from lab.kv_cache import AttentionShape, kv_bytes_per_token, kv_cache_bytes, max_cached_tokens


def test_bytes_per_token_formula() -> None:
    shape = AttentionShape(num_layers=24, num_kv_heads=2, head_dim=64)
    assert kv_bytes_per_token(shape, "fp16") == 2 * 24 * 2 * 64 * 2


def test_from_hf_config_gqa_and_fallbacks() -> None:
    gqa = SimpleNamespace(
        num_hidden_layers=28, num_attention_heads=12, num_key_value_heads=2, hidden_size=1536
    )
    assert AttentionShape.from_hf_config(gqa) == AttentionShape(28, 2, 128)
    mha = SimpleNamespace(num_hidden_layers=12, num_attention_heads=12, hidden_size=768)
    assert AttentionShape.from_hf_config(mha) == AttentionShape(12, 12, 64)
    explicit = SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        hidden_size=512,
        head_dim=128,
    )
    assert AttentionShape.from_hf_config(explicit).head_dim == 128


def test_tensor_parallel_splits_heads_but_never_below_one() -> None:
    shape = AttentionShape(10, 4, 64)
    assert kv_cache_bytes(shape, 100, tensor_parallel=2) == kv_cache_bytes(shape, 100) / 2
    two_heads = AttentionShape(10, 2, 64)
    assert kv_cache_bytes(two_heads, 100, tensor_parallel=4) == kv_cache_bytes(
        AttentionShape(10, 1, 64), 100
    )


def test_max_cached_tokens_inverts_cache_bytes() -> None:
    shape = AttentionShape(24, 2, 64)
    budget = kv_cache_bytes(shape, 1000)
    assert max_cached_tokens(shape, budget) == 1000


def test_rejects_bad_inputs() -> None:
    shape = AttentionShape(1, 1, 1)
    with pytest.raises(ValueError):
        kv_bytes_per_token(shape, "fp7")
    with pytest.raises(ValueError):
        kv_cache_bytes(shape, -1)
