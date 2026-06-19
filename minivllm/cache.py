"""KV cache — Phase 3.

The naive baseline recomputes attention over the whole sequence every decode
step (O(n^2)). The fix is to keep each layer's past keys and values around and,
on each step, only compute K/V for the *new* token and append it. Attention then
runs the new query against the full cached K/V — decode becomes O(n) per step.

This Phase 3 cache is a simple **contiguous** per-layer buffer for a single
sequence. We pre-allocate to the max sequence length up front so the decode hot
path does no tensor allocation and no `torch.cat` — it writes into a slice and
returns a view. Phase 4 swaps this contiguous buffer for a paged (block-based)
allocator behind the same `extend` / `advance` interface, which is why the model
talks to the cache only through those two methods.
"""

from __future__ import annotations

import torch

from minivllm.config import ModelConfig


class KVCache:
    """Contiguous per-layer key/value cache for one sequence (batch size 1)."""

    def __init__(
        self,
        cfg: ModelConfig,
        max_seq_len: int,
        batch_size: int = 1,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ):
        dtype = dtype or cfg.dtype
        shape = (batch_size, cfg.num_key_value_heads, max_seq_len, cfg.head_dim)
        # One K and one V buffer per layer; pre-allocated, never re-allocated.
        self.k = [
            torch.empty(shape, device=device, dtype=dtype) for _ in range(cfg.num_hidden_layers)
        ]
        self.v = [
            torch.empty(shape, device=device, dtype=dtype) for _ in range(cfg.num_hidden_layers)
        ]
        self.max_seq_len = max_seq_len
        self._length = 0  # tokens currently stored (shared across layers)

    @property
    def length(self) -> int:
        return self._length

    def extend(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append this step's K/V for `layer_idx`; return the full cached K/V.

        k, v: [batch, n_kv_heads, new_tokens, head_dim] (post-RoPE keys).
        Returns views spanning positions [0 : length + new_tokens].
        """
        new = k.shape[2]
        end = self._length + new
        if end > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: need {end} positions, capacity {self.max_seq_len}"
            )
        self.k[layer_idx][:, :, self._length : end] = k
        self.v[layer_idx][:, :, self._length : end] = v
        return self.k[layer_idx][:, :, :end], self.v[layer_idx][:, :, :end]

    def advance(self, n: int) -> None:
        """Commit `n` newly written positions. Called once per forward pass,
        after all layers have extended their slice at the current length."""
        self._length += n

    def truncate(self, length: int) -> None:
        """Roll the cache back to `length` tokens. Used by speculative decoding
        to discard the KV of rejected draft tokens after a verification forward:
        the buffer slots stay put and are simply overwritten by the next write."""
        if not 0 <= length <= self._length:
            raise ValueError(f"cannot truncate to {length} from length {self._length}")
        self._length = length
