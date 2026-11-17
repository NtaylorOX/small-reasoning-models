"""Synthetic, seeded prompt workloads.

Prompts are random token ids of an exact length, so every backend sees identical inputs, there
is no padding, and prompt length is a controlled variable. `shared_prefix_len` makes the first
N tokens identical across prompts (a shared system prompt) to exercise prefix caching.
"""

from __future__ import annotations

import random


def synthetic_prompts(
    num_prompts: int,
    prompt_len: int,
    vocab_size: int,
    seed: int,
    shared_prefix_len: int = 0,
    low_id: int = 100,
) -> list[list[int]]:
    if not 0 <= shared_prefix_len <= prompt_len:
        raise ValueError("shared_prefix_len must be between 0 and prompt_len")
    high_id = min(vocab_size, 30_000)  # stay inside ordinary-token ids for most tokenizers
    if high_id <= low_id:
        raise ValueError("vocab_size too small")
    rng = random.Random(seed)
    prefix = [rng.randrange(low_id, high_id) for _ in range(shared_prefix_len)]
    return [
        prefix + [rng.randrange(low_id, high_id) for _ in range(prompt_len - shared_prefix_len)]
        for _ in range(num_prompts)
    ]
