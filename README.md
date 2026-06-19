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
scripts/
  validate_logits.py
tests/
  test_logits.py
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
| 2 | Naive generation + baseline benchmark | ⏳ |
| 3 | KV cache | ⏳ |
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

<!-- Benchmark numbers land here starting Phase 2. -->
