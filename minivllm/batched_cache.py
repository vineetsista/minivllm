"""Batched KV cache — the storage Phase 5's continuous batching decodes against.

Continuous batching runs one decode step over B "slots" at once, where each slot
holds a different in-flight sequence at a different length. This cache keeps one
[B, n_kv, max_len, head_dim] buffer per layer plus a per-slot `lengths` vector,
and masks each row to its own valid length so a single batched matmul serves all
slots. New tokens are scattered into each slot's next free position.

It speaks the same `extend` contract the attention module already uses (write the
new K/V, return the full K/V), so `Attention.forward` runs batched unchanged.
Length commit is split out into `advance(active)` because, unlike single-sequence
decode, only the *active* slots step forward each iteration — the engine, not the
cache, knows which slots are live.

Prefill is done per sequence into a contiguous cache and copied into a slot via
`load_prefill` (decode is the batched part; mixing ragged prefill into the batch
is a further optimization beyond Phase 5's scheduler focus).
"""

from __future__ import annotations

import torch

from minivllm.cache import KVCache
from minivllm.config import ModelConfig


class BatchedKVCache:
    def __init__(
        self,
        cfg: ModelConfig,
        num_slots: int,
        max_seq_len: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ):
        dtype = dtype or cfg.dtype
        shape = (num_slots, cfg.num_key_value_heads, max_seq_len, cfg.head_dim)
        self.k = [
            torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.num_hidden_layers)
        ]
        self.v = [
            torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.num_hidden_layers)
        ]
        self.num_slots = num_slots
        self.max_seq_len = max_seq_len
        self.device = device
        # Per-slot count of tokens currently cached (also the next write index).
        self.lengths = torch.zeros(num_slots, dtype=torch.long, device=device)
        self._rows = torch.arange(num_slots, device=device)

    def _width(self) -> int:
        """Active key width this step = longest live history + 1 (the token just
        written). Decode attends only over this, not the whole buffer — otherwise
        every step pays attention over max_seq_len regardless of real lengths."""
        return min(int(self.lengths.max().item()) + 1, self.max_seq_len)

    def extend(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scatter each slot's new token K/V at its own `lengths` position and
        return the per-layer buffers sliced to the active width. k, v:
        [num_slots, n_kv, 1, head_dim].

        Every layer in a step writes at the same `lengths` (advance is deferred
        to the engine), so the writes — and the width — stay consistent across
        the layer loop.
        """
        self.k[layer_idx][self._rows, :, self.lengths, :] = k[:, :, 0, :]
        self.v[layer_idx][self._rows, :, self.lengths, :] = v[:, :, 0, :]
        w = self._width()
        return self.k[layer_idx][:, :, :w, :], self.v[layer_idx][:, :, :w, :]

    def make_mask(self, dtype: torch.dtype) -> torch.Tensor:
        """Additive attention mask [num_slots, 1, 1, width]: row b may attend to
        key positions 0..lengths[b] (inclusive of the token just written), masked
        beyond that. Width matches the sliced buffers returned by `extend`."""
        w = self._width()
        positions = torch.arange(w, device=self.device)
        valid = positions[None, :] <= self.lengths[:, None]  # [num_slots, width]
        mask = torch.zeros_like(valid, dtype=dtype)
        mask.masked_fill_(~valid, float("-inf"))
        return mask[:, None, None, :]

    def position_ids(self) -> torch.Tensor:
        """Absolute position of each slot's new token = its current length."""
        return self.lengths[:, None]

    def advance(self, active: torch.Tensor) -> None:
        """Commit one token for the active slots only. `active`: bool [num_slots]."""
        self.lengths[active] += 1

    def load_prefill(self, slot: int, prefill: KVCache, prompt_len: int) -> None:
        """Copy a single-sequence prefill's K/V (positions 0..prompt_len-1) into
        `slot` and set its length. Overwrites whatever the slot held before."""
        for layer in range(len(self.k)):
            self.k[layer][slot, :, :prompt_len, :] = prefill.k[layer][0, :, :prompt_len, :]
            self.v[layer][slot, :, :prompt_len, :] = prefill.v[layer][0, :, :prompt_len, :]
        self.lengths[slot] = prompt_len

    def release(self, slot: int) -> None:
        """Free a slot for reuse. Length resets; stale K/V is overwritten on the
        next prefill, so no need to zero the buffers."""
        self.lengths[slot] = 0
