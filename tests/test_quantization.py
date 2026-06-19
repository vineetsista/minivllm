"""Wave 4: int8 weight-only quantization (lossy) and lower-precision dtypes.

All fast (tiny random-weight model). Quantization is the one lossy optimization,
so the gates check that the error is *small* (round-trip within the scale, logits
highly correlated with float32) and that memory actually drops — not exact
equality.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minivllm.config import ModelConfig
from minivllm.model import Qwen3ForCausalLM
from minivllm.quantization import QuantizedLinear, model_nbytes, quantize_linears


def _cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        tie_word_embeddings=True,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        max_position_embeddings=128,
    )


def _model() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    return Qwen3ForCausalLM(_cfg()).eval()


def test_quantized_linear_roundtrip_within_scale():
    lin = nn.Linear(16, 8, bias=False)
    q = QuantizedLinear.from_linear(lin)
    dequant = q.qweight.float() * q.scale[:, None]
    # Each weight is within half a quantization step (= scale/?) of the original;
    # bound generously by the per-channel scale.
    err = (dequant - lin.weight.data).abs()
    assert (err <= q.scale[:, None] + 1e-6).all()


def test_quantization_shrinks_memory_and_preserves_logits():
    model = _model()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        ref_logits = model(ids)

    before = model_nbytes(model)
    quantize_linears(model)
    after = model_nbytes(model)
    assert after < before, "int8 weights must reduce model memory"

    # No nn.Linear left except the tied lm_head.
    linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    assert linears == ["lm_head"], f"unexpected un-quantized linears: {linears}"

    with torch.no_grad():
        q_logits = model(ids)
    assert torch.isfinite(q_logits).all()
    cos = F.cosine_similarity(ref_logits.flatten(), q_logits.flatten(), dim=0)
    assert cos > 0.99, f"quantized logits drifted too far (cos={cos:.4f})"


def test_bf16_forward_close_to_fp32():
    model = _model()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        fp32 = model(ids)
        bf16 = model.to(torch.bfloat16)(ids).float()
    cos = F.cosine_similarity(fp32.flatten(), bf16.flatten(), dim=0)
    assert cos > 0.99, f"bf16 logits drifted too far (cos={cos:.4f})"
