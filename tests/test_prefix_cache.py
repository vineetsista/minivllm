"""Phase A: RadixAttention prefix caching must save compute, not change output.

All fast (tiny random-weight model). The load-bearing gate: prefilling with a warm
prefix cache (reusing cached blocks + forwarding only the suffix) produces the
*same* next-token logits as a full-prompt forward — because the cached prefix K/V
was computed by an identical-prefix pass. Plus radix match/insert/evict mechanics.
"""

from __future__ import annotations

import torch

from minivllm.cache import KVCache
from minivllm.config import ModelConfig
from minivllm.model import Qwen3ForCausalLM
from minivllm.prefix_cache import RadixPrefixCache, cached_prefill

BS = 4  # small block size so short prompts span several blocks


def _cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=16,
        tie_word_embeddings=True,
        num_hidden_layers=2,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        max_position_embeddings=256,
    )


def _model(cfg) -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).eval()


# --- radix mechanics (no model) -------------------------------------------------


def _fake_tmp(cfg, length) -> KVCache:
    tmp = KVCache(cfg, max_seq_len=length)
    for layer in range(cfg.num_hidden_layers):
        tmp.k[layer][:, :, :length] = torch.randn_like(tmp.k[layer][:, :, :length])
        tmp.v[layer][:, :, :length] = torch.randn_like(tmp.v[layer][:, :, :length])
    tmp._length = length
    return tmp


def test_match_insert_and_prefix_sharing():
    cfg = _cfg()
    pc = RadixPrefixCache(cfg, block_size=BS, capacity_blocks=64)
    a = list(range(12))  # 3 full blocks
    pc.insert(a, _fake_tmp(cfg, 12))
    assert pc.n_blocks == 3

    # A prompt sharing the first 2 blocks -> matches them (and reuses; last-block
    # rule never trims here because the new prompt diverges in block 3).
    b = list(range(8)) + [99, 99, 99, 99]
    matched, m = pc.match(b)
    assert m == 8 and len(matched) == 2

    pc.insert(b, _fake_tmp(cfg, 12))
    assert pc.n_blocks == 4  # 2 shared + 1 new (block 3 of b); block "[99x4]" added


def test_match_never_reuses_whole_prompt():
    cfg = _cfg()
    pc = RadixPrefixCache(cfg, block_size=BS, capacity_blocks=64)
    a = list(range(12))
    pc.insert(a, _fake_tmp(cfg, 12))
    # Exact same prompt: all 3 blocks cached, but the last is dropped so the
    # suffix (>=1 block) is recomputed to produce next-token logits.
    matched, m = pc.match(a)
    assert m == 8


def test_lru_eviction():
    cfg = _cfg()
    pc = RadixPrefixCache(cfg, block_size=BS, capacity_blocks=2)
    pc.insert(list(range(12)), _fake_tmp(cfg, 12))  # 3 blocks -> over capacity 2
    assert pc.n_blocks == 2
    assert pc.evictions >= 1


# --- the correctness gate (tiny model) ------------------------------------------


def test_cached_prefill_matches_full_forward():
    cfg = _cfg()
    model = _model(cfg)
    prompt = [3, 8, 1, 5, 9, 2, 7, 4, 6, 0, 11, 13]  # 12 tokens = 3 blocks of 4

    full = model(torch.tensor([prompt]))[0, -1]

    pc = RadixPrefixCache(cfg, block_size=BS, capacity_blocks=64)
    cold, _ = cached_prefill(model, prompt, pc)  # miss -> full forward + insert
    assert pc.hits == 0 and pc.misses == 1
    assert torch.allclose(cold, full, atol=1e-4)

    warm, _ = cached_prefill(model, prompt, pc)  # hit -> reuse 2 blocks, forward suffix
    assert pc.hits == 1 and pc.prefix_tokens_reused == 8
    assert torch.allclose(warm, full, atol=1e-4), "prefix reuse changed the logits"


def test_no_cache_is_plain_forward():
    cfg = _cfg()
    model = _model(cfg)
    prompt = [1, 2, 3, 4, 5, 6, 7]
    full = model(torch.tensor([prompt]))[0, -1]
    out, tmp = cached_prefill(model, prompt, None)
    assert torch.allclose(out, full, atol=1e-5)
    assert tmp.length == len(prompt)
