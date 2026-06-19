"""mini-vLLM: a from-scratch high-performance LLM inference engine.

Surface so far:
  * Phase 1 — a config-driven Qwen3 implementation plus a weight loader,
    validated logit-for-logit against the HuggingFace reference.
  * Phase 2 — naive autoregressive generation and a benchmark harness that
    locks in the baseline (TTFT, decode latency, throughput).
  * Phase 3 — a contiguous KV cache (O(n) decode).
  * Phase 4 — a paged (block-based) KV cache: block allocator + per-sequence
    block table, eliminating fragmentation behind the same cache interface.
  * Phase 5 — continuous (iteration-level) batching: a batched-decode cache and
    a scheduler that keeps the batch full instead of draining it.
  * Phase 7 — speculative decoding: draft + parallel verify with KV rollback,
    exact for greedy.
"""

from minivllm.cache import KVCache
from minivllm.config import ModelConfig
from minivllm.engine import ContinuousBatchingEngine, Request
from minivllm.generate import GenerationOutput, SamplingParams, generate
from minivllm.paged_cache import BlockAllocator, PagedKVCache
from minivllm.speculative import NgramDrafter, SpeculativeDecoder

__all__ = [
    "ModelConfig",
    "SamplingParams",
    "GenerationOutput",
    "generate",
    "KVCache",
    "PagedKVCache",
    "BlockAllocator",
    "ContinuousBatchingEngine",
    "Request",
    "SpeculativeDecoder",
    "NgramDrafter",
]
