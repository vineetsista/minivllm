"""Wave 2 correctness: paging the batched serving cache changes memory, not output.

All fast (tiny random-weight model, no download):
  * the shared block pool reserves one block per slot, grows on demand, and
    reclaims growth blocks on release;
  * the continuous-batching engine backed by the paged pool produces, for every
    request, exactly what the contiguous batched engine and single-sequence
    greedy decode produce.
"""

from __future__ import annotations

import torch

from minivllm.config import ModelConfig
from minivllm.engine import ContinuousBatchingEngine, Request
from minivllm.generate import SamplingParams, generate
from minivllm.model import Qwen3ForCausalLM
from minivllm.paged_batched_cache import PagedBatchedKVCache


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        hidden_size=16,
        tie_word_embeddings=True,
        num_hidden_layers=2,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        max_position_embeddings=128,
    )


def _tiny_model(cfg: ModelConfig) -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).eval()


# --- fast: pool bookkeeping -----------------------------------------------------


def test_pool_reserves_and_reclaims():
    cfg = _tiny_cfg()
    cache = PagedBatchedKVCache(cfg, num_slots=2, max_seq_len=64, block_size=4, num_blocks=10)
    # Two reserved blocks (one per slot) -> 8 free in the shared pool.
    assert cache.num_free == 8

    # Grow slot 0 to 10 tokens -> ceil(10/4)=3 blocks total, i.e. +2 from the pool.
    cache._ensure_capacity(0, 10)
    assert cache.num_free == 6
    cache.release(0)  # growth blocks return; reserved block stays
    assert cache.num_free == 8


def test_pool_exhaustion_raises():
    cfg = _tiny_cfg()
    # 2 slots reserve 2 blocks; only 1 left in the pool.
    cache = PagedBatchedKVCache(cfg, num_slots=2, max_seq_len=64, block_size=4, num_blocks=3)
    cache._ensure_capacity(0, 5)  # needs a 2nd block — ok (1 free)
    try:
        cache._ensure_capacity(0, 9)  # needs a 3rd block — pool now empty
    except RuntimeError:
        return
    raise AssertionError("expected pool exhaustion to raise")


# --- fast: paged engine == contiguous engine == single-sequence greedy ----------


def test_paged_engine_matches_contiguous_and_single_sequence():
    cfg = _tiny_cfg()
    model = _tiny_model(cfg)
    prompt = [1, 2, 3, 4, 5]
    reqs = [
        Request(id=i, prompt_ids=list(prompt), max_new_tokens=n) for i, n in enumerate([4, 9, 6, 7])
    ]

    ref = {}
    for r in reqs:
        ref[r.id] = generate(
            model,
            torch.tensor([r.prompt_ids]),
            SamplingParams(max_new_tokens=r.max_new_tokens, temperature=0.0),
            eos_token_id=None,
            use_cache=True,
        ).generated_token_ids

    contiguous = ContinuousBatchingEngine(model, max_slots=2, eos_token_id=None, paged=False)
    paged = ContinuousBatchingEngine(
        model, max_slots=2, eos_token_id=None, paged=True, block_size=4
    )
    out_c, _ = contiguous.run(reqs, SamplingParams(temperature=0.0), policy="continuous")
    out_p, _ = paged.run(reqs, SamplingParams(temperature=0.0), policy="continuous")

    for rid in ref:
        assert out_c[rid] == ref[rid], f"contiguous request {rid} diverged"
        assert out_p[rid] == ref[rid], f"paged request {rid} diverged"
