"""mini-vLLM: a from-scratch high-performance LLM inference engine.

Surface so far:
  * Phase 1 — a config-driven Qwen3 implementation plus a weight loader,
    validated logit-for-logit against the HuggingFace reference.
  * Phase 2 — naive autoregressive generation and a benchmark harness that
    locks in the baseline (TTFT, decode latency, throughput).
"""

from minivllm.config import ModelConfig
from minivllm.generate import GenerationOutput, SamplingParams, generate

__all__ = ["ModelConfig", "SamplingParams", "GenerationOutput", "generate"]
