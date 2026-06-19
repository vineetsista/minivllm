"""Phase 7 correctness: speculative decoding must equal plain greedy, exactly.

  * Fast: n-gram drafter logic, and — the important one — a tiny random-weight
    target where speculative greedy output must match single-sequence greedy
    output token-for-token. That exercises the parallel verify + KV rollback
    (the hard part) with no model download, at any acceptance level.
  * Slow gate: same invariant on the real Qwen3-0.6B, on a repetitive prompt
    where the n-gram drafter actually lands tokens (acceptance > 0, fewer target
    forwards than tokens generated).
"""

from __future__ import annotations

import torch

from minivllm.config import ModelConfig
from minivllm.generate import SamplingParams, generate
from minivllm.model import Qwen3ForCausalLM
from minivllm.speculative import NgramDrafter, SpeculativeDecoder

MODEL = "Qwen/Qwen3-0.6B"


def _tiny_model() -> Qwen3ForCausalLM:
    cfg = ModelConfig(
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
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg)
    model.eval()
    return model


# --- fast: drafter logic --------------------------------------------------------

def test_ngram_drafter_proposes_recurring_continuation():
    drafter = NgramDrafter(max_ngram=3)
    # "a b c" occurred earlier followed by "d e"; the suffix "... a b c" should
    # propose "d e".
    history = [10, 11, 12, 13, 14, 99, 10, 11, 12]
    assert drafter.propose(history, k=2) == [13, 14]


def test_ngram_drafter_empty_when_no_match():
    drafter = NgramDrafter(max_ngram=3)
    assert drafter.propose([1, 2, 3, 4, 5], k=3) == []


# --- fast: exact-match against greedy on a tiny random model --------------------

def test_speculative_matches_greedy_tiny():
    model = _tiny_model()
    prompt = [1, 2, 3, 4, 5]
    n = 20

    greedy = generate(
        model, torch.tensor([prompt]),
        SamplingParams(max_new_tokens=n, temperature=0.0), eos_token_id=None, use_cache=True,
    ).generated_token_ids

    for k in (1, 4, 8):  # acceptance window size must not change the output
        spec, _ = SpeculativeDecoder(model, drafter=NgramDrafter(), k=k).generate(prompt, n)
        assert spec == greedy, f"k={k} diverged:\n greedy={greedy}\n spec  ={spec}"


# --- slow gate: exact-match on the real model, with real acceptance -------------

def test_speculative_matches_greedy_real_model():
    from transformers import AutoTokenizer

    from minivllm.loader import load_model

    model, _ = load_model(MODEL, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(MODEL)
    # A deliberately repetitive prompt so the n-gram drafter lands tokens.
    prompt = "Repeat after me: the cat sat on the mat. the cat sat on the mat. the cat"
    prompt_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    n = 40

    greedy = generate(
        model, torch.tensor([prompt_ids]),
        SamplingParams(max_new_tokens=n, temperature=0.0), eos_token_id=None, use_cache=True,
    ).generated_token_ids

    spec, stats = SpeculativeDecoder(model, k=4).generate(prompt_ids, n)
    assert spec == greedy, f"speculative diverged from greedy:\n greedy={greedy}\n spec={spec}"
    # The drafter should actually help here: some acceptance, fewer target passes.
    assert stats.accepted_tokens > 0, "expected the n-gram drafter to land some tokens"
    assert stats.target_forwards < n, "expected fewer target forwards than tokens generated"
