"""Primitive layers: RMSNorm and rotary position embeddings.

These are kept separate from the model so Phase 6 can swap in fused Triton
kernels behind the same call signatures without touching model.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm.

    Matches the HF reference exactly: the normalization is computed in float32
    and the learned weight is applied after casting back to the input dtype.
    Getting the dtype dance right is what makes logits match bit-for-bit on
    lower-precision runs; on our float32 CPU build it is a no-op but we keep it
    faithful so the code is correct when we move to a GPU + bf16.
    """

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Precomputes the RoPE inv-freqs; computes cos/sin for given positions.

    GPT-NeoX / Llama style: the head dim is split into two halves and the same
    set of frequencies is duplicated across both halves. `position_ids` is
    explicit so that with a KV cache (Phase 3) we can pass the absolute
    position of each new token rather than assuming positions start at 0.
    """

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        # buffer (not a parameter) so it moves with .to(device) but isn't trained/saved
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [batch, seq] -> freqs: [batch, seq, head_dim/2]
        inv_freq = self.inv_freq.to(position_ids.device)
        freqs = position_ids.float()[..., None] * inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1)  # [batch, seq, head_dim]
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dim: [x1, x2] -> [-x2, x1]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key.

    q, k:    [batch, n_heads, seq, head_dim]
    cos,sin: [batch, seq, head_dim]  ->  unsqueeze the head axis to broadcast.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
