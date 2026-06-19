"""Load HuggingFace Qwen3 weights into our from-scratch model.

Our module names mirror the HF layout (`model.layers.N.self_attn.q_proj`, ...),
so the state dict maps essentially 1:1. The one subtlety is tied embeddings:
when `tie_word_embeddings` is set, the checkpoint ships no `lm_head.weight`
(it reuses `model.embed_tokens.weight`), so a missing lm_head key is expected,
not an error.
"""

from __future__ import annotations

import glob
import os

import torch
from safetensors.torch import load_file

from minivllm.config import ModelConfig
from minivllm.model import Qwen3ForCausalLM


def _resolve_local_dir(model_id_or_path: str) -> str:
    """Return a local directory containing the model files, downloading if needed."""
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_id_or_path,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*"],
    )


def load_model(
    model_id_or_path: str = "Qwen/Qwen3-0.6B",
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> tuple[Qwen3ForCausalLM, ModelConfig]:
    """Instantiate our model and load weights from a HF checkpoint."""
    cfg = ModelConfig.from_hf(model_id_or_path, dtype=dtype)
    model = Qwen3ForCausalLM(cfg)

    local_dir = _resolve_local_dir(model_id_or_path)
    shards = sorted(glob.glob(os.path.join(local_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No .safetensors found in {local_dir}")

    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        state.update(load_file(shard))

    missing, unexpected = model.load_state_dict(state, strict=False)

    # Tolerate exactly the tied lm_head weight being absent; flag anything else.
    allowed_missing = {"lm_head.weight"} if cfg.tie_word_embeddings else set()
    real_missing = set(missing) - allowed_missing
    if real_missing:
        raise RuntimeError(f"Missing weights not present in checkpoint: {sorted(real_missing)}")
    if unexpected:
        raise RuntimeError(f"Unexpected weights in checkpoint: {sorted(unexpected)}")

    model.to(device=device, dtype=dtype)
    model.eval()
    return model, cfg
