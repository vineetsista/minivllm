"""RadixAttention-style prefix caching.

Many requests share a prompt prefix — a system prompt, a few-shot preamble, a long
document. Recomputing that prefix's KV on every request is pure waste. This caches
the KV of already-computed prefixes in a **radix tree** keyed by token content, so a
new request reuses the longest cached prefix and only prefills the *suffix*. The
win: prefill compute and time-to-first-token drop sharply on shared prompts.

The tree is at **block granularity** (block_size tokens): each node holds one block
of tokens plus that block's K/V for every layer, and a node's identity is its path
from the root — so the same block content under different prefixes is a different
node, which is exactly prefix matching. Matching walks the tree block-by-block;
inserting adds the newly computed blocks; an LRU pass evicts the least-recently
used leaves when the store is over capacity.

This is **copy-on-hit**: on a match the cached prefix K/V is copied into the
request's scratch cache (the compute/TTFT win). True zero-copy block *sharing* in
the decode gather is a further step that needs the paged-attention kernel — noted
in docs/DESIGN.md. Only the single worker thread mutates the tree (like the KV
cache), so no locks are needed on the hot path; `snapshot()` reads defensively.
"""

from __future__ import annotations

from typing import cast

import torch

from minivllm.cache import KVCache
from minivllm.config import ModelConfig
from minivllm.model import Qwen3ForCausalLM


class RadixNode:
    __slots__ = ("token_ids", "k", "v", "children", "parent", "last_access", "hits", "node_id")

    def __init__(self, token_ids, k, v, parent, node_id):
        self.token_ids: tuple[int, ...] = token_ids
        self.k: torch.Tensor | None = k  # [n_layers, n_kv, block_size, head_dim]
        self.v: torch.Tensor | None = v
        self.children: dict[tuple[int, ...], RadixNode] = {}
        self.parent: RadixNode | None = parent
        self.last_access = 0
        self.hits = 0  # times this cached block was reused
        self.node_id = node_id


class RadixPrefixCache:
    def __init__(
        self,
        cfg: ModelConfig,
        block_size: int = 16,
        capacity_blocks: int = 128,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ):
        self.cfg = cfg
        self.block_size = block_size
        self.capacity = capacity_blocks
        self.device = device
        self.dtype = dtype or cfg.dtype
        self.root = RadixNode((), None, None, None, 0)
        self.n_blocks = 0
        self._clock = 0
        self._next_id = 1
        # stats
        self.hits = 0
        self.misses = 0
        self.prefix_tokens_reused = 0
        self.prefix_tokens_total = 0
        self.evictions = 0

    # -- helpers -----------------------------------------------------------------

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _full_blocks(self, token_ids) -> list[tuple[int, ...]]:
        n = len(token_ids) // self.block_size
        bs = self.block_size
        return [tuple(token_ids[i * bs : (i + 1) * bs]) for i in range(n)]

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    # -- match / assemble / insert ----------------------------------------------

    def match(self, token_ids) -> tuple[list[RadixNode], int]:
        """Longest run of cached full blocks that prefix `token_ids`."""
        matched: list[RadixNode] = []
        node = self.root
        for block in self._full_blocks(token_ids):
            child = node.children.get(block)
            if child is None:
                break
            matched.append(child)
            child.last_access = self._tick()
            child.hits += 1
            node = child

        m = len(matched) * self.block_size
        # Never reuse the *entire* prompt: we'd have no next-token logits. Drop the
        # last block so the suffix (>=1 block) is recomputed to produce them.
        if m == len(token_ids) and matched:
            matched.pop()
            m -= self.block_size

        if m > 0:
            self.hits += 1
            self.prefix_tokens_reused += m
        else:
            self.misses += 1
        self.prefix_tokens_total += len(token_ids)
        return matched, m

    def assemble(self, matched: list[RadixNode]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Concatenate matched blocks into per-layer prefix K/V [n_kv, m, head_dim].
        Matched nodes are live (only evicted nodes have k/v cleared to None)."""
        n_layers = self.cfg.num_hidden_layers
        ks = [cast(torch.Tensor, nd.k) for nd in matched]
        vs = [cast(torch.Tensor, nd.v) for nd in matched]
        k_layers = [torch.cat([k[layer] for k in ks], dim=1) for layer in range(n_layers)]
        v_layers = [torch.cat([v[layer] for v in vs], dim=1) for layer in range(n_layers)]
        return k_layers, v_layers

    def insert(self, token_ids, tmp_cache: KVCache) -> None:
        """Insert every full block of `token_ids` whose K/V now lives in `tmp_cache`
        (positions 0..len(token_ids)-1). Existing blocks are touched, not duplicated."""
        blocks = self._full_blocks(token_ids)
        if not blocks:
            return
        n_layers = self.cfg.num_hidden_layers
        bs = self.block_size
        # [n_layers, n_kv, prompt_len, head_dim]
        k_full = torch.stack([tmp_cache.k[layer][0] for layer in range(n_layers)])
        v_full = torch.stack([tmp_cache.v[layer][0] for layer in range(n_layers)])

        node = self.root
        for bi, block in enumerate(blocks):
            child = node.children.get(block)
            if child is None:
                kb = k_full[:, :, bi * bs : (bi + 1) * bs, :].clone()
                vb = v_full[:, :, bi * bs : (bi + 1) * bs, :].clone()
                child = RadixNode(block, kb, vb, node, self._next_id)
                self._next_id += 1
                node.children[block] = child
                self.n_blocks += 1
            child.last_access = self._tick()
            node = child
        self._evict_to_fit()

    # -- eviction ----------------------------------------------------------------

    def _evict_to_fit(self) -> None:
        while self.n_blocks > self.capacity:
            leaf = self._lru_leaf()
            if leaf is None or leaf.parent is None:
                break
            del leaf.parent.children[leaf.token_ids]
            leaf.parent = None
            leaf.k = leaf.v = None
            self.n_blocks -= 1
            self.evictions += 1

    def _lru_leaf(self) -> RadixNode | None:
        best: RadixNode | None = None
        stack = [self.root]
        while stack:
            node = stack.pop()
            is_leaf = node is not self.root and not node.children
            if is_leaf and (best is None or node.last_access < best.last_access):
                best = node
            stack.extend(node.children.values())
        return best

    # -- viz ---------------------------------------------------------------------

    def snapshot(self, max_nodes: int = 200) -> list[dict]:
        """Defensive BFS snapshot for the dashboard (id, parent, depth, hits, tokens)."""
        out: list[dict] = []
        queue: list[tuple[RadixNode, int, int]] = [(self.root, 0, 0)]
        while queue and len(out) < max_nodes:
            node, depth, parent_id = queue.pop(0)
            if node is not self.root:
                out.append(
                    {
                        "id": node.node_id,
                        "parent": parent_id,
                        "depth": depth,
                        "hits": node.hits,
                        "tokens": list(node.token_ids),
                    }
                )
            for child in list(node.children.values()):
                queue.append((child, depth + 1, node.node_id))
        return out

    def stats(self) -> dict:
        return {
            "cached_blocks": self.n_blocks,
            "capacity_blocks": self.capacity,
            "hit_rate": self.hit_rate,
            "prefix_tokens_reused": self.prefix_tokens_reused,
            "prefix_tokens_total": self.prefix_tokens_total,
            "evictions": self.evictions,
        }


@torch.no_grad()
def cached_prefill(
    model: Qwen3ForCausalLM,
    prompt_ids: list[int],
    prefix_cache: RadixPrefixCache | None,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, KVCache]:
    """Prefill `prompt_ids`, reusing any cached prefix. Returns (last_token_logits,
    scratch KVCache holding the full prompt K/V). With no cache or no match this is
    a plain full-prompt forward, so behaviour is unchanged when prefix caching is off.
    """
    dtype = dtype or next(model.parameters()).dtype
    pids = list(prompt_ids)
    matched, m = prefix_cache.match(pids) if prefix_cache is not None else ([], 0)

    tmp = KVCache(model.cfg, max_seq_len=len(pids), device=device, dtype=dtype)
    if m > 0:
        assert prefix_cache is not None  # m > 0 only when a cache produced the match
        k_layers, v_layers = prefix_cache.assemble(matched)
        tmp.preload(k_layers, v_layers, m)
        suffix = pids[m:]  # non-empty: match() never reuses the whole prompt
        logits = model(torch.tensor([suffix], device=device), cache=tmp)[0, -1]
    else:
        logits = model(torch.tensor([pids], device=device), cache=tmp)[0, -1]

    if prefix_cache is not None:
        prefix_cache.insert(pids, tmp)
    return logits, tmp
