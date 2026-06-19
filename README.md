# mini-vLLM — a from-scratch LLM inference engine

A high-performance inference engine for a real open-weight model, built from
scratch in Python (PyTorch + Triton). The point is the **optimization journey**:
load a model correctly, then climb the performance curve — KV caching, paged
attention, continuous batching, a custom kernel, speculative decoding — with a
before/after number for every step.

Target model: **Qwen/Qwen3-0.6B** (Apache 2.0, ungated). Modern transformer
stack: RoPE, RMSNorm, grouped-query attention with QK-Norm, SwiGLU.

> **Hardware note.** Phases 1–5, 7, 8 run on CPU and are developed here on a
> CPU-only laptop. The Triton kernel (Phase 6) and the headline throughput /
> GPU-memory numbers require a CUDA GPU (rented hourly: RunPod / Lambda /
> vast.ai). Numbers are labeled with the hardware they were measured on.

## Architecture

```
minivllm/
  config.py       # ModelConfig, populated from the HF config.json
  layers.py       # RMSNorm, RoPE (kernel swap-point for Phase 6)
  model.py        # from-scratch Qwen3: attention (GQA + QK-Norm), SwiGLU, decoder
  loader.py       # map HF safetensors -> our modules
  validation.py   # logit-parity check vs the HF reference
  generate.py     # naive + KV-cache autoregressive decode, sampling
  cache.py        # KVCache: contiguous per-layer K/V buffers
  paged_cache.py  # BlockAllocator + PagedKVCache (PagedAttention-style)
  batched_cache.py# BatchedKVCache: per-slot histories for batched decode
  engine.py       # continuous-batching scheduler (static + continuous policies)
  speculative.py  # speculative decoding: drafter + parallel verify with rollback
  server.py       # FastAPI serving layer + threaded streaming ServingEngine
  benchmark.py    # TTFT / decode-latency / throughput harness
scripts/
  validate_logits.py
  generate.py     # CLI: generate text
  benchmark.py    # CLI: compare decode paths (naive vs cache)
  paged_demo.py   # CLI: paged vs static KV memory / fragmentation / capacity
  batch_bench.py  # CLI: static vs continuous batching throughput
  spec_bench.py   # CLI: speculative vs greedy (acceptance, target forwards)
  loadtest.py     # CLI: async concurrent load test against the server
tests/
  test_logits.py
  test_generation.py
  test_kv_cache.py
  test_paged_cache.py
  test_continuous_batching.py
  test_speculative.py
  test_server.py
```

Design intent: clean separation between **model**, **KV-cache manager**,
**scheduler**, and **serving layer** (the latter three arrive in Phases 3–5, 8),
with Python overhead kept off the hot path.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; use source .venv/bin/activate elsewhere
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Phase status

| Phase | What | Status |
|---|---|---|
| 1 | Correct forward pass + logit parity | ✅ |
| 2 | Naive generation + baseline benchmark | ✅ |
| 3 | KV cache | ✅ |
| 4 | Paged KV cache (PagedAttention-style) | ✅ |
| 5 | Continuous (iteration-level) batching | ✅ |
| 6 | Custom Triton kernel | ⏳ (needs GPU) |
| 7 | Speculative decoding | ✅ |
| 8 | FastAPI serving + load-test dashboard | ✅ |

## Phase 1 — Correctness

We reimplement the Qwen3 forward pass and prove it matches HuggingFace
token-for-token before optimizing anything.

```bash
python -m scripts.validate_logits
python -m pytest tests/test_logits.py -v
```

The check asserts (1) argmax agreement at **every** position, (2) max absolute
logit difference within float32 tolerance, and (3) matching top-k next-token
candidates.

### Qwen3 details that matter for parity

- `head_dim` is set independently of `hidden_size // num_heads`.
- **QK-Norm**: an RMSNorm is applied per-head to queries and keys (over
  `head_dim`) *before* RoPE.
- No bias on the q/k/v/o projections.
- Grouped-query attention (`num_key_value_heads < num_attention_heads`).
- RMSNorm computed in float32; RoPE with a large theta (1e6).
- Tied input/output embeddings on the 0.6B (no separate `lm_head.weight`).

## Phase 2 — Naive generation + baseline

Plain autoregressive decode with **no KV cache**: every step re-runs the model
over the entire sequence so far. That is deliberate — it is the slow baseline
every later phase has to beat, and the recompute cost grows O(n²) in sequence
length, which is precisely what the KV cache (Phase 3) removes.

```bash
python -m scripts.generate --prompt "The capital of France is"
python -m scripts.benchmark --max-new-tokens 64 --runs 3
python -m pytest tests/test_generation.py -v
```

The harness reports time-to-first-token (TTFT, the prefill latency), per-token
decode latency (p50/p99), and throughput. A warmup run precedes the timed runs;
EOS is ignored during benchmarking so each run does a fixed amount of work.

**Correctness carries forward from Phase 1.** `test_generation.py` asserts our
greedy decode matches a manual HuggingFace greedy loop token-for-token across
many steps — so the cache-free loop is provably correct *before* we start
optimizing it.

### Baseline numbers

Measured on the dev laptop: **CPU-only** (Intel i7-1250U, 10 torch threads),
float32, Qwen3-0.6B, 5-token prompt → 64 new tokens, 3 runs.

| Phase | Decode tok/s | TTFT (ms) | Decode p50 / p99 (ms/tok) |
|---|---|---|---|
| Naive (no cache) | 1.00 | 403 | 901.1 / 2467.4 |

The decode p50→p99 spread (901 → 2467 ms) is the O(n²) tax made visible: the
last tokens of the sequence are far slower than the first because each one
re-attends over everything before it. Phase 3 should flatten that curve and lift
decode throughput sharply.

> GPU numbers (and the dramatic throughput figures) come once we rent a CUDA box
> for the later phases; the CPU baseline above is what the *algorithmic* wins in
> Phases 3–5 are measured against on the same hardware.

## Phase 3 — KV cache

The naive baseline recomputes attention over the whole sequence every decode
step. The KV cache keeps each layer's past keys and values, so a decode step
computes K/V only for the **new** token, appends it, and attends against the
cached history — decode drops from O(n²) total to O(n).

`cache.py` pre-allocates a contiguous per-layer buffer up to the max sequence
length, so the decode hot path does no allocation and no `torch.cat`: it writes
into a slice and returns a view. The model talks to it through just `extend` /
`advance`, which is the seam Phase 4 replaces with a paged (block-based)
allocator. Post-RoPE keys are cached, so positions are baked in once.

```bash
python -m scripts.benchmark --max-new-tokens 64 --runs 3   # naive vs cache
python -m pytest tests/test_kv_cache.py -v
```

`test_kv_cache.py` asserts cached greedy decode equals **both** the naive path
and the HuggingFace greedy loop, token-for-token — the cache changes cost, never
output.

### The win

Same hardware as the baseline, both paths measured back-to-back in one run
(5-token prompt → 64 new tokens):

| Phase | Decode tok/s | TTFT (ms) | Decode p50 / p99 (ms/tok) |
|---|---|---|---|
| Naive (no cache) | 0.65 | 618 | 1369.9 / 4727.2 |
| KV cache | **5.81** | 218 | 168.0 / 251.8 |

**~9× decode throughput.** Just as telling, the per-token decode latency goes
flat — p50/p99 collapse from 1370/4727 ms to 168/252 ms. The naive path's huge
p50→p99 spread *was* the O(n²) tax (later tokens re-attend over everything);
caching removes it, so every decode step costs about the same regardless of how
far into the sequence it is.

## Phase 4 — Paged KV cache

The contiguous cache gives every sequence one buffer sized to `max_seq_len`.
Under real serving load that is brutally wasteful: a short sequence still
reserves the maximum, and packing many different-length sequences fragments the
pool. Paging applies the OS virtual-memory trick — physical KV is carved into
fixed-size **blocks**, a central `BlockAllocator` hands them out and reclaims
them, and each sequence keeps a **block table** mapping its logical positions to
physical blocks (`paged_cache.py`). Any free block fits any sequence, so there is
no external fragmentation; internal waste is bounded by `block_size - 1` tokens
per sequence regardless of `max_seq_len`. This is the vLLM PagedAttention idea.

Crucially it sits behind the **same `extend` / `advance` interface** as the
contiguous cache, so `model.py` is untouched — only the storage changes.

```bash
python -m scripts.paged_demo                       # the memory win
python -m pytest tests/test_paged_cache.py -v
```

`test_paged_cache.py` checks the allocator (alloc/free/reuse, pool sharing
across sequences, exhaustion), proves the paged gather reproduces a contiguous
buffer bit-for-bit, and gates that paged greedy decode == contiguous == HF,
token-for-token (with a small block size so sequences span several blocks).

### The win — memory, not single-stream speed

Qwen3-0.6B is 28 layers × 8 KV heads × head_dim 128, so KV costs **112 KiB per
token** (float16). For 64 sequences of mixed length (avg ~264 tokens) capped at
`max_seq_len = 2048`, block size 16:

| Scheme | KV memory | Utilization | Concurrent seqs in 4 GiB |
|---|---|---|---|
| Contiguous (static) | 14.00 GiB | 12.9% | 18 |
| **Paged** | **1.80 GiB** | **100.0%** | **141** |

**~7.8× less memory and ~8× more concurrent sequences** in a fixed budget —
because memory now tracks tokens actually used, not the worst case reserved per
sequence. That concurrency is the raw material Phase 5's scheduler turns into
throughput.

Honest tradeoff: on this CPU reference path, paged single-stream decode is
~15–20% *slower* than the contiguous cache (a Python per-token scatter plus a
per-step gather to assemble blocks for the attention math). That gather is
exactly what the Phase 6 paged-attention kernel removes by attending over
scattered blocks directly — paging buys the memory/concurrency now, the kernel
buys the speed back later.

## Phase 5 — Continuous (iteration-level) batching

A single decode step can serve a whole batch at once, so the lever for
throughput is *scheduling*: when do new requests join the batch? `batched_cache.py`
keeps one `[slots, n_kv, max_len, head_dim]` buffer per layer with a per-slot
length, masking each row to its own history — so one batched matmul decodes all
slots (`Attention.forward` runs batched unchanged; the model gains only a
`decode_step` entry). `engine.py` is the scheduler, with two admission policies:

- **static** — admit a group of B, decode until *every* slot finishes, then
  admit the next group. A short sequence's slot sits idle until the longest in
  its group drains.
- **continuous** — admit a waiting request the moment *any* slot frees, every
  step. Slots stay full; no compute is wasted on idle rows.

They share the exact same compute, so the comparison isolates the scheduler.
Prefill is done per sequence into a temp cache and copied into a slot; decode is
the batched part.

```bash
python -m scripts.batch_bench --slots 4 --short 8 --long 48
python -m pytest tests/test_continuous_batching.py -v
```

`test_continuous_batching.py` gates that **both** policies reproduce, for every
request, exactly what single-sequence greedy decode produces — batching changes
the schedule, never the tokens.

### The win

12 requests, 4 slots, mixed generation lengths `[48, 8, 8, 8] × 3` (the high-
variance case where scheduling matters), same hardware:

| Policy | Wall (s) | Decode steps | Avg batch occupancy | Throughput (tok/s) |
|---|---|---|---|---|
| Static | 99.05 | 141 | 1.53 / 4 | 2.18 |
| **Continuous** | **28.28** | **61** | **3.54 / 4** | **7.64** |

**~3.5× throughput**, from **2.3× fewer decode steps** (141 → 61). The mechanism
is occupancy: static averages 1.53 live slots of 4 (three short sequences finish
and idle while a length-48 sequence drains its group); continuous refills those
slots instantly and averages 3.54 of 4. The wider the length spread and the more
requests, the larger the gap.

## Phase 7 — Speculative decoding

A cheap *drafter* proposes K tokens; the target verifies all K in **one** forward
and accepts the longest prefix it agrees with, plus a free bonus token. For
greedy decoding this is **exact** — identical output to plain target greedy,
token-for-token — because an accepted draft token is by definition the target's
own argmax, and the correction at the first disagreement is the target's argmax
too. Speculation only changes how many target forwards it takes (`speculative.py`).

The systems-interesting part is the verify-and-rollback: feed `[last, draft…]`
through the target with the KV cache, compare each position's argmax to the
draft, then `cache.truncate(...)` away the KV of rejected tokens so the next
round continues from the accepted prefix. `Drafter` is an interface; the default
`NgramDrafter` needs no second model (prompt-lookup: propose the continuation of
the longest recent token n-gram that recurs), and a draft *model* implements the
same `propose` contract via `ModelDrafter`.

```bash
python -m scripts.spec_bench --max-new-tokens 64 --k 4
python -m pytest tests/test_speculative.py -v
```

`test_speculative.py` proves speculative greedy == single-sequence greedy on a
tiny random-weight model (across k = 1, 4, 8, so the verify/rollback is exercised
at every acceptance level) and again on the real model.

### The win

64 tokens, k = 4, on a repetitive prompt (where n-gram drafting lands), same CPU:

| Decoder | Target forwards | Tokens / forward | Wall (s) |
|---|---|---|---|
| Greedy | 64 | 1.00 | 11.57 |
| **Speculative** | **15** | **4.27** | **4.07** |

**93% acceptance**, **4.3× fewer target forwards**, and **~2.8× wall-clock**. The
wall-clock win cuts against the naive "CPU is compute-bound" expectation: 0.6B
decode is in fact largely *memory-bound* (weights stream from RAM each pass), so a
5-token verify forward costs far less than five single-token forwards — the same
amortization that makes speculation a latency win on GPU, just smaller. The gain
is workload-dependent: repetitive/structured text drafts well; novel text accepts
less and gains less.

## Phase 8 — Serving layer + load test

The offline engine runs a fixed list to completion; a server must keep the batch
full as requests *arrive*. `server.py` runs `ServingEngine` — a background worker
thread that owns the model and the batched KV cache and runs the decode loop
forever, admitting queued requests into free slots every iteration and signalling
each request's completion. FastAPI handlers `submit` and await that event in a
threadpool (`asyncio.to_thread`), so the synchronous CPU-bound forward never
blocks the event loop. The worker is the sole owner of cache/slots, so only the
waiting queue needs a lock. Per-slot sampling params let requests in one batch
use different temperatures.

```bash
python -m uvicorn minivllm.server:app --port 8000   # MINIVLLM_SLOTS, MINIVLLM_MODEL via env
python -m scripts.loadtest --n 8 --concurrency 4 --max-new-tokens 32
python -m pytest tests/test_server.py -v
```

Endpoints: `POST /generate`, `GET /health`, `GET /stats`. `test_server.py`
submits concurrent requests from multiple threads against a tiny model and checks
every response equals single-sequence greedy — validating the worker, dynamic
admission, and completion signalling without a download.

### The win — throughput scales with concurrency

8 requests of 32 tokens, 4 slots, same CPU. Going from serialized to 4-way
continuous batching:

| Concurrency | Wall (s) | Latency p50 (s) | Throughput (tok/s) |
|---|---|---|---|
| 1 (serialized) | 120.65 | 18.01 | 2.12 |
| **4 (batched)** | **22.17** | **11.05** | **11.55** |

**~5.4× aggregate throughput** — and per-request latency *drops* too (18.0 → 11.0
s), because batched slots share each decode step's weight load (the memory-bound
amortization again). One subtlety found here and fixed: the batched cache must
attend only over the active sequence width, not the full pre-allocated buffer, or
every step pays attention over `max_seq_len` regardless of real lengths.

## The optimization journey

Every technique, measured on the same CPU dev box (Qwen3-0.6B, float32) against
the previous stage. Correctness is gated at every step: logits, then greedy
decode, then each cache/scheduler variant, all match the HuggingFace reference or
single-sequence decode token-for-token.

| Phase | Technique | Headline result (CPU, Qwen3-0.6B) |
|---|---|---|
| 1 | Correct forward pass | logit parity vs HF (max abs diff ~1.7e-5, argmax 100%) |
| 2 | Naive decode + baseline | ~1.0 tok/s decode; O(n²) latency growth (p99 4727 ms/tok) |
| 3 | KV cache | **~9×** decode throughput; latency flat (p99 4727 → 252 ms/tok) |
| 4 | Paged KV cache | **7.8×** less KV memory, **8×** more concurrent seqs (12.9% → 100% util) |
| 5 | Continuous batching | **~3.5×** throughput vs static (occupancy 1.5 → 3.5 of 4 slots) |
| 6 | Custom Triton kernel | deferred — requires a CUDA GPU |
| 7 | Speculative decoding | **~2.8×** wall-clock, 4.3× fewer target forwards (93% acceptance) |
| 8 | Serving + load test | **~5.4×** throughput at concurrency 4 vs serialized |

> Numbers are CPU reference figures that isolate each *algorithmic* win. The
> headline GPU throughput / memory figures — and the Phase 6 Triton kernel — come
> from a rented CUDA box; every technique above is hardware-agnostic and carries
> over.
