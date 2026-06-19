"""Phase 4 correctness: paging changes storage, not output, and recycles blocks.

* Fast: allocator alloc/free/reuse, block-table growth, pool sharing across
  sequences, overflow, and that the paged gather reproduces a plain
  contiguous buffer bit-for-bit.
* Slow gate: paged greedy decode == contiguous greedy decode == HuggingFace
  greedy loop, token-for-token.
"""

from __future__ import annotations

import torch

from minivllm.config import ModelConfig
from minivllm.generate import SamplingParams, generate
from minivllm.paged_cache import BlockAllocator, PagedKVCache, kv_bytes_per_token

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


# --- fast: allocator + block table ----------------------------------------------


def test_allocator_alloc_free_reuse():
    cfg = _tiny_cfg()
    alloc = BlockAllocator(cfg, num_blocks=4, block_size=8)
    assert alloc.num_free == 4
    a, b = alloc.allocate(), alloc.allocate()
    assert alloc.num_free == 2
    alloc.free([a, b])
    assert alloc.num_free == 4  # freed blocks return to the pool


def test_allocator_exhaustion_raises():
    cfg = _tiny_cfg()
    alloc = BlockAllocator(cfg, num_blocks=1, block_size=8)
    alloc.allocate()
    try:
        alloc.allocate()
    except RuntimeError:
        return
    raise AssertionError("expected pool exhaustion to raise")


def test_block_table_grows_with_length():
    cfg = _tiny_cfg()
    cache = PagedKVCache(cfg, max_seq_len=64, block_size=4)
    k = torch.randn(1, cfg.num_key_value_heads, 6, cfg.head_dim)  # 6 tokens, block 4
    cache.extend(0, k, k)
    cache.advance(6)
    assert cache.num_blocks == 2  # ceil(6 / 4)
    k1 = torch.randn(1, cfg.num_key_value_heads, 1, cfg.head_dim)
    cache.extend(0, k1, k1)
    cache.advance(1)  # 7 tokens
    assert cache.num_blocks == 2  # still fits 2 blocks of 4


def test_pool_shared_across_sequences_and_reclaimed():
    cfg = _tiny_cfg()
    alloc = BlockAllocator(cfg, num_blocks=8, block_size=4)
    seqs = [PagedKVCache(cfg, max_seq_len=64, block_size=4, allocator=alloc) for _ in range(3)]
    for s in seqs:
        k = torch.randn(1, cfg.num_key_value_heads, 4, cfg.head_dim)
        s.extend(0, k, k)
        s.advance(4)
    assert alloc.num_free == 8 - 3  # three sequences, one block each, one pool
    seqs[0].free()
    assert alloc.num_free == 8 - 2  # block returned for reuse


def test_paged_gather_matches_contiguous():
    """The gathered K/V must equal a plain contiguous buffer of the same tokens."""
    cfg = _tiny_cfg()
    cache = PagedKVCache(cfg, max_seq_len=64, block_size=4)
    ref_k, ref_v = [], []
    length = 0
    for step_len in (5, 1, 1, 3):  # prefill then decode steps, crossing blocks
        k = torch.randn(1, cfg.num_key_value_heads, step_len, cfg.head_dim)
        v = torch.randn(1, cfg.num_key_value_heads, step_len, cfg.head_dim)
        gk, gv = cache.extend(0, k, v)
        cache.advance(step_len)
        ref_k.append(k)
        ref_v.append(v)
        length += step_len
        expect_k = torch.cat(ref_k, dim=2)
        expect_v = torch.cat(ref_v, dim=2)
        assert torch.equal(gk, expect_k), f"key gather mismatch at length {length}"
        assert torch.equal(gv, expect_v), f"value gather mismatch at length {length}"


def test_kv_bytes_per_token():
    cfg = _tiny_cfg()  # 2 layers, 1 kv head, head_dim 4
    # 2 (K+V) * 2 layers * 1 head * 4 dim * 4 bytes (fp32) = 64
    assert kv_bytes_per_token(cfg, torch.float32) == 64


# --- slow gate: paged == contiguous == reference --------------------------------


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


def test_paged_decode_matches_contiguous_and_reference():
    from transformers import AutoTokenizer

    from minivllm.loader import load_model

    prompt = "The capital of France is"
    n = 12
    model, _ = load_model(MODEL, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(MODEL)
    input_ids = tok(prompt, return_tensors="pt").input_ids
    params = SamplingParams(max_new_tokens=n, temperature=0.0)

    contig = generate(
        model, input_ids, params, eos_token_id=None, use_cache=True
    ).generated_token_ids
    # Small block size on purpose, so the sequence spans several blocks.
    pgd = generate(
        model, input_ids, params, eos_token_id=None, use_cache=True, paged=True, block_size=4
    ).generated_token_ids
    ref = _hf_greedy_tokens(MODEL, prompt, n)

    assert pgd == contig, f"paging changed output:\n contig={contig}\n paged ={pgd}"
    assert pgd == ref, f"paged decode diverged from reference:\n ours={pgd}\n ref ={ref}"
