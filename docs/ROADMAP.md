# Roadmap — from working engine to flagship

Phases 1–5, 7, 8 built a correct, measured inference engine on CPU. This roadmap
takes it from "a strong pile of demos" to one coherent, modern, production-shaped
system. Each wave ships as focused commits with tests and before/after numbers,
the same discipline as the original phases.

## Wave 1 — Foundations ✅/⏳
Make the project bulletproof and professional.
- Offline, fast test tier mechanically isolated from slow HF-reference gates
  (markers + `--runslow`/`RUN_SLOW`), so `pytest` is fast and deterministic.
- GitHub Actions CI: ruff lint + format check + fast tests on every push.
- `ruff` + `mypy` configured; modern `pyproject` (deps, `[dev]`/`[gpu]` extras,
  metadata, console scripts); `LICENSE`; `.pre-commit-config.yaml`.
- One-command benchmark runner that regenerates every README number.

## Wave 2 — Architectural coherence
Turn separate demos into one vLLM-shaped engine.
- **Wire PagedAttention into the live batched serving path** — a shared block
  pool with per-sequence block tables, replacing the contiguous, max-length
  `BatchedKVCache`. This is the headline systems upgrade.
- Real per-request batched sampling (temperature/top-k/top-p/seed per slot),
  plus repetition penalty, min-p, and stop sequences.

## Wave 3 — Flagship features
The drop-in, "wait, this is from scratch?" moment.
- **Token streaming** over SSE.
- **OpenAI-compatible API** (`/v1/chat/completions`, `/v1/completions`) with chat
  templates — point the `openai` SDK / LangChain / Open WebUI at this engine.
- **Prometheus `/metrics`** + a **live web dashboard** (throughput, latency,
  KV-cache utilization, batch occupancy in real time).

## Wave 4 — Depth
- INT8 weight-only quantization (with a memory/throughput benchmark).
- bf16/fp16 paths actually exercised.
- **Phase 6 Triton kernel** (fused RMSNorm+RoPE or paged attention), CUDA-gated
  with a CPU reference and a numerical-equivalence test, so it lives in the repo
  and runs wherever a GPU is present.

## Wave 5 — Presentation
- Auto-generated benchmark **plots** (the optimization journey), embedded in the
  README.
- Mermaid architecture diagram; README overhaul with badges and a quickstart that
  includes the OpenAI-client demo.
- Dockerfile + compose for the server; `DESIGN.md` writeup of the decisions.

Correctness is non-negotiable at every step: logits match the HuggingFace
reference, and every cache/scheduler variant matches single-sequence decode
token-for-token.
