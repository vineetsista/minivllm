"""INT8 weight-only quantization — Wave 4.

The single biggest lever on inference memory (and, on GPU, bandwidth-bound decode
speed) is the weights. This stores each Linear's weight as int8 with a per-output-
channel scale — 4x smaller than float32 — and dequantizes on the fly in the
forward pass. Unlike everything else in this engine it is *lossy*: a small,
measured quality cost for a large memory saving (scripts/quant_demo.py reports the
logit fidelity).

Symmetric per-output-channel quantization: for weight row w (one output channel),
scale = max|w| / 127, q = round(w / scale) in [-127, 127]. Dequant is q * scale.
Per-channel (rather than per-tensor) keeps error low because output channels have
very different magnitudes.

Tied embeddings (lm_head) are skipped: the weight is shared with the token
embedding and quantizing it would break the tie and hurt output quality most.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinear(nn.Module):
    """A drop-in for nn.Linear holding int8 weights + per-output-channel scales."""

    def __init__(self, qweight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.in_features = qweight.shape[1]
        self.out_features = qweight.shape[0]
        self.register_buffer("qweight", qweight)  # int8 [out, in]
        self.register_buffer("scale", scale)  # float [out]
        self.register_buffer("bias", bias if bias is not None else None)

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> QuantizedLinear:
        w = lin.weight.data
        scale = (w.abs().amax(dim=1) / 127.0).clamp(min=1e-8)  # [out]
        q = (w / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        bias = lin.bias.data.clone() if lin.bias is not None else None
        return cls(q, scale.to(w.dtype), bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize to the activation dtype, then a normal matmul. (On GPU a real
        # int8 kernel would fuse this; here the win is the 4x smaller stored weight.)
        qweight = cast(torch.Tensor, self.qweight)
        scale = cast(torch.Tensor, self.scale)
        bias = cast("torch.Tensor | None", self.bias)
        w = qweight.to(x.dtype) * scale[:, None]
        return F.linear(x, w, bias)


def quantize_linears(model: nn.Module, skip: tuple[str, ...] = ("lm_head",)) -> nn.Module:
    """Replace every nn.Linear in `model` (except `skip` names) with an int8
    QuantizedLinear, in place. Returns the model for chaining."""
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and not any(s in full for s in skip):
                setattr(module, child_name, QuantizedLinear.from_linear(child))
    return model


def model_nbytes(model: nn.Module) -> int:
    """Total bytes of parameters + buffers (counts int8 weights as 1 byte each)."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        if b is not None:
            total += b.numel() * b.element_size()
    return total
