"""Phase 3 correctness: the KV cache must not change the output, only the cost.

* Fast unit test for the cache buffer mechanics (no model download).
* Slow gate: cached greedy decode == naive greedy decode == HuggingFace
  greedy loop, token-for-token. If caching the post-RoPE keys or the
  position/mask bookkeeping were wrong, this is where it would show up.
"""

from __future__ import annotations

import torch

from minivllm.cache import KVCache
from minivllm.config import ModelConfig
from minivllm.generate import SamplingParams, generate

MODEL = "Qwen/Qwen3-0.6B"


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        hidden_size=8,
        tie_word_embeddings=True,
        num_hidden_layers=2,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        max_position_embeddings=128,
    )


# --- fast unit test (no model) --------------------------------------------------


def test_cache_extend_and_advance():
    cfg = _tiny_cfg()
    cache = KVCache(cfg, max_seq_len=10)
    assert cache.length == 0

    # Prefill 3 tokens into layer 0.
    k = torch.randn(1, cfg.num_key_value_heads, 3, cfg.head_dim)
    v = torch.randn(1, cfg.num_key_value_heads, 3, cfg.head_dim)
    fk, fv = cache.extend(0, k, v)
    assert fk.shape[2] == 3 and fv.shape[2] == 3
    cache.advance(3)
    assert cache.length == 3

    # One decode step: appends at position 3, returns the full length-4 history.
    k1 = torch.randn(1, cfg.num_key_value_heads, 1, cfg.head_dim)
    v1 = torch.randn(1, cfg.num_key_value_heads, 1, cfg.head_dim)
    fk, fv = cache.extend(0, k1, v1)
    assert fk.shape[2] == 4
    # The previously written keys are preserved and the new one is appended.
    assert torch.equal(fk[:, :, :3], k)
    assert torch.equal(fk[:, :, 3:4], k1)


def test_cache_overflow_raises():
    cfg = _tiny_cfg()
    cache = KVCache(cfg, max_seq_len=2)
    k = torch.randn(1, cfg.num_key_value_heads, 3, cfg.head_dim)
    try:
        cache.extend(0, k, k)
    except ValueError:
        return
    raise AssertionError("expected KV cache overflow to raise")


# --- slow gate: cached == naive == reference ------------------------------------


@torch.no_grad()
def _hf_greedy_tokens(model_id: str, prompt: str, n: int) -> list[int]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    ref = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    ids = tok(prompt, return_tensors="pt").input_ids
    out: list[int] = []
    for _ in range(n):
        nxt = int(ref(ids).logits[0, -1].argmax().item())
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return out


def test_cached_decode_matches_naive_and_reference():
    from transformers import AutoTokenizer

    from minivllm.loader import load_model

    prompt = "The capital of France is"
    n = 12
    model, _ = load_model(MODEL, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(MODEL)
    input_ids = tok(prompt, return_tensors="pt").input_ids
    params = SamplingParams(max_new_tokens=n, temperature=0.0)

    naive = generate(
        model, input_ids, params, eos_token_id=None, use_cache=False
    ).generated_token_ids
    cached = generate(
        model, input_ids, params, eos_token_id=None, use_cache=True
    ).generated_token_ids
    ref = _hf_greedy_tokens(MODEL, prompt, n)

    assert cached == naive, f"cache changed the output:\n naive={naive}\n cache={cached}"
    assert cached == ref, f"cached decode diverged from reference:\n ours={cached}\n ref ={ref}"
