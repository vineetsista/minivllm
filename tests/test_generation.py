"""Phase 2 correctness: the decode loop, not just a single forward.

Two layers:
  * Fast unit tests for the token-selection logic (no model download).
  * A slow gate: our greedy decode matches a manual HuggingFace greedy loop
    token-for-token. This extends the Phase 1 logit-parity guarantee across
    many autoregressive steps — if our cache-free loop drifts from the
    reference, the optimized phases would inherit the bug.
"""

from __future__ import annotations

import torch

from minivllm.generate import SamplingParams, _select_next_token, generate

MODEL = "Qwen/Qwen3-0.6B"


# --- fast unit tests (no model) -------------------------------------------------

def test_greedy_selects_argmax():
    logits = torch.tensor([0.1, 5.0, -2.0, 3.0])
    params = SamplingParams(temperature=0.0)
    assert _select_next_token(logits, params, generator=None) == 1


def test_sampling_is_seed_reproducible():
    logits = torch.randn(128)
    params = SamplingParams(temperature=1.0, seed=0)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = _select_next_token(logits, params, generator=g1)
    b = _select_next_token(logits, params, generator=g2)
    assert a == b


def test_top_k_one_is_greedy():
    logits = torch.tensor([0.1, 5.0, -2.0, 3.0])
    params = SamplingParams(temperature=1.0, top_k=1, seed=0)
    g = torch.Generator().manual_seed(0)
    # With only the argmax surviving, sampling must return it.
    assert _select_next_token(logits, params, generator=g) == 1


# --- slow gate: greedy decode matches the HF reference loop ---------------------

@torch.no_grad()
def _hf_greedy_tokens(model_id: str, prompt: str, n: int) -> list[int]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    ref = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    ids = tok(prompt, return_tensors="pt").input_ids
    out: list[int] = []
    for _ in range(n):
        logits = ref(ids).logits[0, -1]
        nxt = int(logits.argmax().item())
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return out


def test_greedy_decode_matches_reference():
    from transformers import AutoTokenizer

    from minivllm.loader import load_model

    prompt = "The capital of France is"
    n = 8

    model, _ = load_model(MODEL, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(MODEL)
    input_ids = tok(prompt, return_tensors="pt").input_ids

    ours = generate(
        model,
        input_ids,
        SamplingParams(max_new_tokens=n, temperature=0.0),
        eos_token_id=None,  # force a fixed length so the comparison is exact
    ).generated_token_ids
    ref = _hf_greedy_tokens(MODEL, prompt, n)

    assert ours == ref, f"greedy decode diverged:\n ours={ours}\n ref ={ref}"
