"""Phase 5 correctness: batching changes the schedule, not the tokens.

  * Fast: BatchedKVCache scatters each slot's token at its own length and masks
    each row to its own valid range (no model needed).
  * Slow gate: both the continuous and static engines reproduce, for every
    request, exactly what single-sequence greedy decode produces. A batched-mask
    or position bug would corrupt one row's attention and show up here.
"""

from __future__ import annotations

import torch

from minivllm.batched_cache import BatchedKVCache
from minivllm.config import ModelConfig
from minivllm.engine import ContinuousBatchingEngine, Request
from minivllm.generate import SamplingParams, generate

MODEL = "Qwen/Qwen3-0.6B"


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        hidden_size=8,
        tie_word_embeddings=True,
        num_hidden_layers=1,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        max_position_embeddings=128,
    )


# --- fast: batched cache mechanics ----------------------------------------------

def test_batched_scatter_writes_per_slot_position():
    cfg = _tiny_cfg()
    cache = BatchedKVCache(cfg, num_slots=2, max_seq_len=5)
    cache.lengths = torch.tensor([0, 2])  # slot 0 empty, slot 1 has 2 tokens

    k = torch.randn(2, cfg.num_key_value_heads, 1, cfg.head_dim)
    v = torch.randn(2, cfg.num_key_value_heads, 1, cfg.head_dim)
    cache.extend(0, k, v)

    # Each slot's new token landed at its own length offset.
    assert torch.equal(cache.k[0][0, :, 0, :], k[0, :, 0, :])
    assert torch.equal(cache.k[0][1, :, 2, :], k[1, :, 0, :])


def test_batched_mask_is_per_row():
    cfg = _tiny_cfg()
    cache = BatchedKVCache(cfg, num_slots=2, max_seq_len=5)
    cache.lengths = torch.tensor([0, 2])
    mask = cache.make_mask(torch.float32)  # [2, 1, 1, 5]

    finite = torch.isfinite(mask)[:, 0, 0, :]
    # Row 0 may attend only to position 0; row 1 to positions 0..2.
    assert finite[0].tolist() == [True, False, False, False, False]
    assert finite[1].tolist() == [True, True, True, False, False]


# --- slow gate: engine == single-sequence decode --------------------------------

def test_engine_matches_single_sequence_both_policies():
    from transformers import AutoTokenizer

    from minivllm.loader import load_model

    model, _ = load_model(MODEL, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = tok("The capital of France is", return_tensors="pt").input_ids[0].tolist()

    # Varied generation lengths — the case where scheduling actually matters.
    reqs = [Request(id=i, prompt_ids=list(prompt_ids), max_new_tokens=n)
            for i, n in enumerate([4, 9, 6, 5, 8])]

    # Reference: single-sequence greedy decode (EOS ignored for fixed lengths).
    ref = {}
    for r in reqs:
        out = generate(
            model, torch.tensor([r.prompt_ids]),
            SamplingParams(max_new_tokens=r.max_new_tokens, temperature=0.0),
            eos_token_id=None, use_cache=True,
        )
        ref[r.id] = out.generated_token_ids

    engine = ContinuousBatchingEngine(model, max_slots=2, eos_token_id=None)
    for policy in ("continuous", "static"):
        outputs, stats = engine.run(reqs, SamplingParams(temperature=0.0), policy=policy)
        assert set(outputs) == set(ref), f"{policy}: missing requests"
        for rid, toks in ref.items():
            assert outputs[rid] == toks, (
                f"{policy} request {rid} diverged:\n single={toks}\n batch ={outputs[rid]}"
            )
        assert stats.total_generated == sum(len(v) for v in ref.values())
