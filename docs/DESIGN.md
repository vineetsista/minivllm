# Design notes

The engineering decisions behind mini-vLLM, phase by phase. The throughline:
**correctness is a gate, not a goal** — every optimization must reproduce the
previous stage token-for-token before it counts.

## Correctness methodology

Two test tiers. Fast tests use tiny random-weight models and pure logic (offline,
deterministic, run in CI). Slow "gates" download Qwen3-0.6B and compare against
the HuggingFace reference — logits in Phase 1, then greedy decode, then every
cache/scheduler/decoder variant — asserting equality token-for-token. Because an
accepted speculative token *is* the target's argmax, and a paged/batched cache
only changes *where* K/V lives, these are all exact-equality checks, not
tolerances. The lone exception is int8 quantization, which is explicitly lossy and
gated on fidelity (cosine, argmax agreement) instead.

## Phase 1 — Correct forward pass

A from-scratch Qwen3 decoder. The details that actually matter for parity:
`head_dim` is independent of `hidden_size / num_heads`; **QK-Norm** applies an
RMSNorm per head to Q and K *before* RoPE; no bias on q/k/v/o; GQA; RMSNorm in
float32. The gotcha that cost real time: transformers 5.x moved `rope_theta` under
`config.rope_parameters` (1e6 for Qwen3); silently defaulting to 10000 breaks
parity. The attention module already takes explicit `position_ids` and returns its
K/V, so a cache threads through later with minimal change.

## Phase 2 — Naive baseline

Plain autoregressive decode with no cache — every step re-runs the whole sequence
(O(n²)). Deliberately slow: it is the honest baseline every later phase beats, and
its p50→p99 latency blowup *is* the O(n²) tax made visible.

## Phase 3 — KV cache

Cache each layer's K/V so a decode step computes K/V for only the new token. The
buffer is **pre-allocated** to the max length, so the hot path does no allocation
and no `torch.cat` — it writes a slice and returns a view. The model talks to the
cache through just `extend`/`advance`/`truncate`; that narrow interface is the
seam everything else slots into. ~9× decode throughput, and the latency curve goes
flat (constant per-step cost).

## Phase 4 — Paged KV cache (PagedAttention)

Contiguous per-sequence caches fragment under real load and over-reserve. Paging
borrows OS virtual memory: a central `BlockAllocator` owns fixed-size blocks, each
sequence keeps a **block table** mapping logical positions to physical blocks. No
external fragmentation; internal waste is bounded by `block_size − 1` tokens per
sequence regardless of max length. 7.8× less KV memory and 8× more concurrent
sequences in a fixed budget. It sits behind the same `extend`/`advance` interface,
so the model is untouched.

## Phase 5 — Continuous batching

One batched decode step serves B slots; the lever is *scheduling*. `engine.py`
implements two admission policies on identical compute: **static** (admit a group,
drain it fully, then the next) vs **continuous** (refill any free slot every step).
The win is occupancy — static idles slots while a long sequence drains; continuous
keeps them full. `Attention.forward` was already written in `[B, …]` terms, so the
batched path needed only a per-slot-length mask and a thin `decode_step` entry. A
subtle but important fix: attend over the active sequence *width*, not the full
pre-allocated buffer, or every step pays attention over `max_seq_len`.

**Wave 2 coherence:** the paged pool was initially reachable only from
single-sequence generation. `PagedBatchedKVCache` wires it into the live serving
engine — one shared pool, per-slot block tables, one reserved block per slot so the
fixed-width masked decode is unchanged. Memory now tracks tokens in flight, and
pool exhaustion is backpressure rather than OOM.

## Phase 6 — Fused Triton kernel

RMSNorm's reduce→normalize→scale is memory-bound; fusing the row into one kernel
launch saves global-memory round-trips on GPU. `kernels.rms_norm` dispatches to a
Triton kernel on CUDA and a bit-identical PyTorch reference on CPU, and `RMSNorm`
delegates to it — so the kernel swaps in without touching the model, and CPU logit
parity is preserved. Validated against the reference on any CUDA box.

## Phase 7 — Speculative decoding

A cheap drafter proposes K tokens; the target verifies all K in one forward and
accepts the longest agreeing prefix plus a bonus token. For greedy this is
**exact**. The systems-interesting part is verify-and-rollback: feed `[last,
draft…]`, compare each position's argmax to the draft, then `cache.truncate` away
the rejected K/V. The default drafter is model-free n-gram (prompt-lookup); a draft
model implements the same `propose` contract. Measured insight: even on CPU it wins
wall-clock (~2.8× on repetitive text) because 0.6B decode is largely *memory-bound*
— a K-token verify forward amortizes the weight load, the same reason it wins on
GPU.

## Phase 8 + serving

A server must keep the batch full as requests *arrive*. `ServingEngine` runs a
background worker thread that owns the model and cache; requests hit a thread-safe
queue, and the worker admits/decodes/evicts every iteration. HTTP handlers submit
and await a completion event in a threadpool, so the synchronous forward never
blocks the event loop. The worker is the sole owner of cache/slots, so only the
queue needs a lock. Each generated token is pushed to a per-request stream queue,
which the **SSE** endpoints drain with incremental detokenization. The
**OpenAI-compatible** layer (`openai_api.py`) is pure translation over this engine,
applying the model's chat template. `/metrics` exposes Prometheus counters and
latency/TTFT percentiles; the dashboard polls them live.

## Quantization

int8 weight-only, symmetric per-output-channel: `scale = max|w|/127`,
`q = round(w/scale)`. Per-channel keeps error low because output channels differ in
magnitude. Tied embeddings (lm_head) are skipped. Lossy by design — 2.24× smaller
model at logit cosine 0.99943, identical greedy output on the demo prompt.

## Why these choices read as production infra

- One narrow cache interface (`extend`/`advance`) that four implementations honor,
  so optimizations compose instead of forking the model.
- Backpressure over OOM; admission control sized to a memory budget.
- Correctness gates that are exact, not approximate, wherever exactness is possible.
- Observability and a standard API surface, so it drops into existing tooling.
- CI, types, and an offline-fast test tier so the thing stays correct as it grows.
