"""mini-vLLM: a from-scratch high-performance LLM inference engine.

Phase 1 surface: a config-driven Qwen3 implementation plus a weight loader,
validated logit-for-logit against the HuggingFace reference.
"""

from minivllm.config import ModelConfig

__all__ = ["ModelConfig"]
