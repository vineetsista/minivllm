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
  cache.py        # KVCache: contiguous per-layer K/V buffers (paged in Phase 4)
  benchmark.py    # TTFT / decode-latency / throughput harness
scripts/
  validate_logits.py
  generate.py     # CLI: generate text
  benchmark.py    # CLI: compare decode paths (naive vs cache)
tests/
  test_logits.py
  test_generation.py
  test_kv_cache.py
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
| 4 | Paged KV cache (PagedAttention-style) | ⏳ |
| 5 | Continuous (iteration-level) batching | ⏳ |
| 6 | Custom Triton kernel | ⏳ (needs GPU) |
| 7 | Speculative decoding | ⏳ |
| 8 | FastAPI serving + load-test dashboard | ⏳ |

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
