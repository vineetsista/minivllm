"""Phase 1 correctness gate, as a pytest.

Marked slow because it downloads the model and runs two full forward passes.
Run with:  python -m pytest tests/test_logits.py -v
"""

import pytest
import torch

from minivllm.validation import compare_to_reference

MODEL = "Qwen/Qwen3-0.6B"


@pytest.mark.parametrize(
    "prompt",
    [
        "The capital of France is",
        "Once upon a time, in a land far away,",
        "def fibonacci(n):",
    ],
)
def test_logits_match_reference(prompt):
    r = compare_to_reference(MODEL, prompt=prompt, dtype=torch.float32)
    # Token-for-token: argmax agrees at every position.
    assert r.argmax_match_frac == 1.0, f"argmax mismatch: {r.argmax_match_frac:.4f}"
    # Numerically close in float32.
    assert r.max_abs_diff < 1e-2, f"max abs logit diff too large: {r.max_abs_diff:.3e}"
    assert r.topk_match, "top-k final-token candidates disagree"
