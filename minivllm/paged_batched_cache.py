"""Paged batched KV cache — Wave 2: PagedAttention wired into batched serving.

`BatchedKVCache` gives every slot a contiguous `[num_slots, max_seq_len, ...]`
region, so the server must reserve `num_slots * max_seq_len` worth of KV up front
— the exact worst-case over-reservation paging was built to kill (Phase 4), but
that paged pool was only ever used by single-sequence `generate(paged=True)`,
never the engine or server.

This cache is the bridge: the same `extend` / `make_mask` / `advance` /
`load_prefill` / `release` interface the batched decode loop already calls, but
storage is a **shared block pool** (one pool per layer) plus a per-slot **block
table**. A slot draws blocks from the pool as it grows and returns them on
release, so total KV memory tracks tokens actually in flight, not the worst case.
The pool can therefore be sized to a memory budget; when it is exhausted,
admission simply waits — backpressure instead of OOM.

To keep the fixed-width masked decode step unchanged (and the change low-risk),
each slot keeps one reserved block for its whole lifetime, so even idle slots
have somewhere to write the masked-out garbage token; growth blocks beyond the
first come from — and return to — the shared pool.

`extend` gathers a slot's scattered blocks into a dense `[num_slots, n_kv, width,
head_dim]` view for the attention matmul. That gather is the CPU stand-in for a
real paged-attention kernel (Phase 6); paging buys the memory flexibility now.
"""

from __future__ import annotations

import torch

from minivllm.cache import KVCache
from minivllm.config import ModelConfig


class PagedBatchedKVCache:
    def __init__(
        self,
        cfg: ModelConfig,
        num_slots: int,
        max_seq_len: int,
        block_size: int = 16,
        num_blocks: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ):
        dtype = dtype or cfg.dtype
        self.cfg = cfg
        self.num_slots = num_slots
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.device = device
        self.dtype = dtype
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.blocks_per_seq = (max_seq_len + block_size - 1) // block_size

        # Default pool = enough for every slot to reach max_seq_len (safe, never
        # OOMs). Sizing it smaller is the whole point — see scripts/serve_mem_demo.
        if num_blocks is None:
            num_blocks = num_slots * self.blocks_per_seq
        if num_blocks < num_slots:
            raise ValueError("need at least one reserved block per slot")
        self.num_blocks = num_blocks

        block_shape = (num_blocks, block_size, self.n_kv, self.head_dim)
        self.k = [
            torch.zeros(block_shape, device=device, dtype=dtype)
            for _ in range(cfg.num_hidden_layers)
        ]
        self.v = [
            torch.zeros(block_shape, device=device, dtype=dtype)
            for _ in range(cfg.num_hidden_layers)
        ]

        self._free: list[int] = list(range(num_blocks))
        # Reserve one block per slot so idle slots always have a write target.
        self.block_tables: list[list[int]] = [[self._free.pop()] for _ in range(num_slots)]
        self.lengths = torch.zeros(num_slots, dtype=torch.long, device=device)

    # -- pool bookkeeping --------------------------------------------------------

    @property
    def num_free(self) -> int:
        return len(self._free)

    def _ensure_capacity(self, slot: int, target_len: int) -> None:
        needed = (target_len + self.block_size - 1) // self.block_size
        while len(self.block_tables[slot]) < needed:
            if not self._free:
                raise RuntimeError("KV block pool exhausted")
            self.block_tables[slot].append(self._free.pop())

    def _width(self) -> int:
        return min(int(self.lengths.max().item()) + 1, self.max_seq_len)

    # -- decode interface (mirrors BatchedKVCache) -------------------------------

    def extend(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scatter each slot's new token into its block, then gather all slots'
        blocks into dense [num_slots, n_kv, width, head_dim] views."""
        kpool, vpool = self.k[layer_idx], self.v[layer_idx]
        for s in range(self.num_slots):
            pos = int(self.lengths[s].item())
            self._ensure_capacity(s, pos + 1)
            block = self.block_tables[s][pos // self.block_size]
            off = pos % self.block_size
            kpool[block, off] = k[s, :, 0, :]
            vpool[block, off] = v[s, :, 0, :]

        width = self._width()
        dense_k = torch.zeros(
            self.num_slots, self.n_kv, width, self.head_dim, device=self.device, dtype=self.dtype
        )
        dense_v = torch.zeros_like(dense_k)
        for s in range(self.num_slots):
            bt = self.block_tables[s]
            flat_k = kpool[bt].reshape(-1, self.n_kv, self.head_dim)
            flat_v = vpool[bt].reshape(-1, self.n_kv, self.head_dim)
            w = min(width, flat_k.shape[0])
            dense_k[s, :, :w, :] = flat_k[:w].transpose(0, 1)
            dense_v[s, :, :w, :] = flat_v[:w].transpose(0, 1)
        return dense_k, dense_v

    def make_mask(self, dtype: torch.dtype) -> torch.Tensor:
        width = self._width()
        positions = torch.arange(width, device=self.device)
        valid = positions[None, :] <= self.lengths[:, None]
        mask = torch.zeros_like(valid, dtype=dtype)
        mask.masked_fill_(~valid, float("-inf"))
        return mask[:, None, None, :]

    def position_ids(self) -> torch.Tensor:
        return self.lengths[:, None]

    def advance(self, active: torch.Tensor) -> None:
        self.lengths[active] += 1

    def load_prefill(self, slot: int, prefill: KVCache, prompt_len: int) -> None:
        """Copy a single-sequence prefill's K/V into `slot`'s blocks, allocating
        from the pool as needed. Block-wise copy keeps it O(prompt_len/block)."""
        self._ensure_capacity(slot, prompt_len)
        bt = self.block_tables[slot]
        for bi, block in enumerate(bt):
            start = bi * self.block_size
            if start >= prompt_len:
                break
            n = min(self.block_size, prompt_len - start)
            for layer in range(len(self.k)):
                # prefill.k[layer][0]: [n_kv, max, hd] -> slice -> [n_kv, n, hd] -> [n, n_kv, hd]
                self.k[layer][block, :n] = prefill.k[layer][0, :, start : start + n, :].transpose(
                    0, 1
                )
                self.v[layer][block, :n] = prefill.v[layer][0, :, start : start + n, :].transpose(
                    0, 1
                )
        self.lengths[slot] = prompt_len

    def release(self, slot: int) -> None:
        """Return a slot's growth blocks to the pool, keep its reserved block."""
        extra = self.block_tables[slot][1:]
        self._free.extend(extra)
        self.block_tables[slot] = self.block_tables[slot][:1]
        self.lengths[slot] = 0
