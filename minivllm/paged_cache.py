"""Paged KV cache — Phase 4, the PagedAttention idea.

The Phase 3 cache gives every sequence one contiguous buffer sized to the max
sequence length. That is simple but wasteful: a sequence that generates 100
tokens still reserves space for `max_seq_len`, and packing many sequences of
different lengths into contiguous regions fragments the pool (external
fragmentation) — exactly the failure mode that sinks naive KV allocation under
real serving load.

Paging borrows the OS virtual-memory trick. Physical KV storage is carved into
fixed-size **blocks**. A central `BlockAllocator` owns the block pool (one pool
per layer) and hands out / reclaims blocks. Each sequence keeps a **block
table** — an ordered list of physical block ids — that maps its logical token
positions to wherever they physically landed. Consequences:

  * No external fragmentation: any free block fits any sequence. A sequence's
    blocks need not be contiguous.
  * Internal fragmentation is bounded by `block_size - 1` tokens per sequence
    (the tail of its last block), independent of `max_seq_len`.
  * Many sequences share one pool; memory tracks tokens *actually* used, so far
    more sequences fit in a fixed budget. (The concurrency this unlocks is what
    Phase 5's scheduler exploits.)

`PagedKVCache` exposes the same `extend` / `advance` / `length` interface as the
contiguous cache, so the model is unchanged — only the storage underneath it is.

Note on the gather: here `extend` gathers a sequence's blocks into a contiguous
tensor for the attention math, so correctness is independent of the kernel. The
real win of avoiding that gather — attending over scattered blocks directly — is
the custom paged-attention kernel in Phase 6. Phase 4 is the memory system.
"""

from __future__ import annotations

import torch

from minivllm.config import ModelConfig


def kv_bytes_per_token(cfg: ModelConfig, dtype: torch.dtype | None = None) -> int:
    """Bytes of KV cache one token occupies across all layers (K and V)."""
    dtype = dtype or cfg.dtype
    itemsize = torch.empty(0, dtype=dtype).element_size()
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * itemsize


class BlockAllocator:
    """Owns the physical block pool (one K and one V tensor per layer) and a
    free list of block ids. Blocks are reference-free: a freed block returns to
    the pool and can back any sequence next."""

    def __init__(
        self,
        cfg: ModelConfig,
        num_blocks: int,
        block_size: int = 16,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ):
        dtype = dtype or cfg.dtype
        self.cfg = cfg
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device
        self.dtype = dtype

        block_shape = (num_blocks, block_size, cfg.num_key_value_heads, cfg.head_dim)
        # Same block id indexes into every layer's pool — the block table is
        # shared across layers, only the contents differ per layer.
        self.k = [torch.empty(block_shape, device=device, dtype=dtype) for _ in range(cfg.num_hidden_layers)]
        self.v = [torch.empty(block_shape, device=device, dtype=dtype) for _ in range(cfg.num_hidden_layers)]
        self._free: list[int] = list(range(num_blocks))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise RuntimeError(
                f"block pool exhausted ({self.num_blocks} blocks, "
                f"{self.block_size} tokens each)"
            )
        return self._free.pop()

    def free(self, block_ids: list[int]) -> None:
        self._free.extend(block_ids)


class PagedKVCache:
    """Per-sequence view over a paged pool. Drop-in for the contiguous cache."""

    def __init__(
        self,
        cfg: ModelConfig,
        max_seq_len: int,
        block_size: int = 16,
        allocator: BlockAllocator | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ):
        if allocator is None:
            # Private pool sized for exactly this one sequence.
            num_blocks = (max_seq_len + block_size - 1) // block_size
            allocator = BlockAllocator(cfg, num_blocks, block_size, device, dtype)
        elif allocator.block_size != block_size:
            raise ValueError("block_size must match the shared allocator")

        self.allocator = allocator
        self.block_size = allocator.block_size
        self.block_table: list[int] = []
        self._length = 0

    @property
    def length(self) -> int:
        return self._length

    @property
    def num_blocks(self) -> int:
        return len(self.block_table)

    def _ensure_capacity(self, target_len: int) -> None:
        """Grow the block table so it can hold `target_len` tokens."""
        needed = (target_len + self.block_size - 1) // self.block_size
        while len(self.block_table) < needed:
            self.block_table.append(self.allocator.allocate())

    def extend(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write this step's K/V into paged storage and return the full history.

        k, v: [1, n_kv_heads, new_tokens, head_dim] (post-RoPE keys). Allocation
        happens lazily here; since length only commits on `advance`, every layer
        in a step sees the same table and writes the same slots.
        """
        new = k.shape[2]
        end = self._length + new
        self._ensure_capacity(end)

        # Scatter each new token into (block, offset). For a single decode token
        # this is one slot; at prefill it spans the touched blocks.
        kpool = self.allocator.k[layer_idx]
        vpool = self.allocator.v[layer_idx]
        for i in range(new):
            pos = self._length + i
            block = self.block_table[pos // self.block_size]
            offset = pos % self.block_size
            kpool[block, offset] = k[0, :, i, :]
            vpool[block, offset] = v[0, :, i, :]

        return self._gather(layer_idx, end)

    def _gather(self, layer_idx: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather this sequence's blocks into [1, n_kv, end, head_dim] views."""
        idx = torch.tensor(self.block_table, device=self.allocator.device)
        # [n_used_blocks, block_size, n_kv, head_dim] -> [used*block_size, n_kv, head_dim]
        gk = self.allocator.k[layer_idx][idx].flatten(0, 1)[:end]
        gv = self.allocator.v[layer_idx][idx].flatten(0, 1)[:end]
        # -> [1, n_kv, end, head_dim]
        return gk.unsqueeze(0).transpose(1, 2), gv.unsqueeze(0).transpose(1, 2)

    def advance(self, n: int) -> None:
        self._length += n

    def free(self) -> None:
        """Return all blocks to the pool (e.g. when the sequence finishes)."""
        self.allocator.free(self.block_table)
        self.block_table = []
        self._length = 0
