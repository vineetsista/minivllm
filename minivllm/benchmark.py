"""Benchmark harness — the Phase 2 baseline numbers every later phase beats.

We measure what matters for inference serving:
  * TTFT (time to first token) — the prefill latency
  * per-token decode latency (p50 / p99)
  * decode throughput (steady-state tokens/sec) and end-to-end tokens/sec

Methodology: a warmup run first (so lazy init, the allocator, and the CPU
threadpool spin-up don't pollute the first measurement), then N timed runs. We
pool the per-token decode latencies across all runs for the percentiles.

The benchmark ignores EOS by default so every run generates exactly
`max_new_tokens` tokens — a fixed, comparable amount of work run to run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from minivllm.config import ModelConfig
from minivllm.generate import SamplingParams, generate


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


@dataclass
class BenchmarkResult:
    model_id: str
    device: str
    dtype: str
    prompt: str
    prompt_tokens: int
    max_new_tokens: int
    n_runs: int
    torch_threads: int

    ttft_p50_s: float
    ttft_p99_s: float
    decode_latency_p50_ms: float
    decode_latency_p99_ms: float
    decode_tokens_per_s: float  # steady-state: 1 / mean decode latency
    end_to_end_tokens_per_s: float  # generated / total wall time, averaged over runs

    def as_markdown_row(self) -> str:
        """A README-pasteable row: phase, decode tok/s, TTFT, p50/p99 decode."""
        return (
            f"| Naive (no cache) | {self.decode_tokens_per_s:.2f} | "
            f"{self.ttft_p50_s * 1000:.0f} | "
            f"{self.decode_latency_p50_ms:.1f} / {self.decode_latency_p99_ms:.1f} |"
        )


@torch.no_grad()
def run_benchmark(
    model,
    tokenizer,
    cfg: ModelConfig,
    prompt: str,
    params: SamplingParams | None = None,
    n_runs: int = 3,
    warmup: int = 1,
    ignore_eos: bool = True,
    device: str = "cpu",
    model_id: str = "Qwen/Qwen3-0.6B",
) -> BenchmarkResult:
    params = params or SamplingParams(max_new_tokens=64, temperature=0.0)
    eos = None if ignore_eos else tokenizer.eos_token_id

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_tokens = int(input_ids.shape[1])

    for _ in range(warmup):
        generate(model, input_ids, params, eos_token_id=eos)

    ttfts: list[float] = []
    decode_latencies: list[float] = []  # seconds, pooled across runs
    e2e_tps: list[float] = []
    for _ in range(n_runs):
        out = generate(model, input_ids, params, eos_token_id=eos)
        ttfts.append(out.prefill_seconds)
        decode_latencies.extend(out.decode_seconds)
        if out.total_seconds > 0:
            e2e_tps.append(out.num_generated / out.total_seconds)

    mean_decode_s = float(np.mean(decode_latencies)) if decode_latencies else float("nan")

    return BenchmarkResult(
        model_id=model_id,
        device=device,
        dtype=str(cfg.dtype).replace("torch.", ""),
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        max_new_tokens=params.max_new_tokens,
        n_runs=n_runs,
        torch_threads=torch.get_num_threads(),
        ttft_p50_s=_percentile(ttfts, 50),
        ttft_p99_s=_percentile(ttfts, 99),
        decode_latency_p50_ms=_percentile([s * 1000 for s in decode_latencies], 50),
        decode_latency_p99_ms=_percentile([s * 1000 for s in decode_latencies], 99),
        decode_tokens_per_s=1.0 / mean_decode_s if mean_decode_s and mean_decode_s == mean_decode_s else float("nan"),
        end_to_end_tokens_per_s=float(np.mean(e2e_tps)) if e2e_tps else float("nan"),
    )
