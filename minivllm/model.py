"""From-scratch Qwen3 decoder.

Phase 1 implements a correct full-sequence (prefill) forward pass. The
attention module is written so that a KV cache can be threaded through in
Phase 3 with minimal change: it already takes explicit `position_ids` and
returns the per-layer key/value tensors it computed.

Qwen3-specific details that matter for matching the reference:
  * head_dim is independent of hidden_size // num_heads
  * QK-Norm: an RMSNorm is applied per-head to queries and keys (over head_dim)
    *before* RoPE
  * no bias on q/k/v/o projections
  * grouped-query attention (num_key_value_heads < num_attention_heads)
  * SwiGLU MLP, RMSNorm, RoPE with a large theta
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minivllm.config import ModelConfig
from minivllm.layers import RMSNorm, RotaryEmbedding, apply_rotary_pos_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match query heads for grouped-query attention.

    x: [batch, n_kv_heads, seq, head_dim] -> [batch, n_kv_heads*n_rep, seq, head_dim]
    """
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, s, d)
        .reshape(b, n_kv * n_rep, s, d)
    )


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)

        # QK-Norm: RMSNorm over head_dim, applied per head before RoPE.
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        b, s, _ = x.shape

        # Project then split into heads. q_norm/k_norm operate on the last
        # (head_dim) axis, so we apply them on the [b, s, n, head_dim] view
        # before transposing to [b, n, s, head_dim].
        q = self.q_norm(self.q_proj(x).view(b, s, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: replicate KV heads up to the query-head count.
        k = repeat_kv(k, self.cfg.num_kv_groups)
        v = repeat_kv(v, self.cfg.num_kv_groups)

        # Scaled dot-product attention, softmax in float32 for stability/parity.
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        if attn_mask is not None:
            scores = scores + attn_mask
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(weights, v)  # [b, n_heads, s, head_dim]

        out = out.transpose(1, 2).reshape(b, s, self.n_heads * self.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, attn_mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attn_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3Model(nn.Module):
    """Embedding + stack of decoder layers + final norm."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, s = input_ids.shape
        device = input_ids.device

        if position_ids is None:
            position_ids = torch.arange(s, device=device).unsqueeze(0).expand(b, s)

        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary(position_ids)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        # Causal mask: additive, [1, 1, s, s], -inf above the diagonal.
        mask = torch.full((s, s), float("-inf"), device=device, dtype=x.dtype)
        mask = torch.triu(mask, diagonal=1)[None, None, :, :]

        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.norm(x)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.model = Qwen3Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.model(input_ids, position_ids)
        return self.lm_head(hidden)
