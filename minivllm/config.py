"""Model configuration.

We keep our own config object rather than depending on a HF config at runtime
in the hot path. It is populated from the model's `config.json` so we stay
faithful to the reference, but the engine never imports `transformers` outside
of the loader and the validation harness.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ModelConfig:
    # Vocab / embedding
    vocab_size: int
    hidden_size: int
    tie_word_embeddings: bool

    # Transformer block
    num_hidden_layers: int
    intermediate_size: int

    # Attention
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int  # NOTE: Qwen3 sets this independently of hidden_size/num_heads
    rms_norm_eps: float

    # RoPE
    rope_theta: float
    max_position_embeddings: int

    # Runtime
    dtype: torch.dtype = torch.float32

    @property
    def num_kv_groups(self) -> int:
        """How many query heads share each KV head (GQA repeat factor)."""
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_hf(cls, model_id_or_path: str, dtype: torch.dtype = torch.float32) -> "ModelConfig":
        """Build our config from a HuggingFace config.json.

        Done once at load time; not on any hot path.
        """
        from transformers import AutoConfig

        hf = AutoConfig.from_pretrained(model_id_or_path)

        # Qwen3 exposes head_dim explicitly. Fall back to the classic
        # hidden_size // num_heads only if a model omits it.
        head_dim = getattr(hf, "head_dim", None)
        if head_dim is None:
            head_dim = hf.hidden_size // hf.num_attention_heads

        # RoPE theta moved around across transformers versions: a top-level
        # `rope_theta` in 4.x, nested under `rope_parameters`/`rope_scaling`
        # in 5.x. Read it wherever it lives — defaulting silently to 10000
        # here is a correctness bug (it breaks logit parity), so we look hard.
        rope_theta = getattr(hf, "rope_theta", None)
        if rope_theta is None:
            for attr in ("rope_parameters", "rope_scaling"):
                params = getattr(hf, attr, None)
                if isinstance(params, dict) and params.get("rope_theta") is not None:
                    rope_theta = params["rope_theta"]
                    break
        if rope_theta is None:
            raise ValueError("Could not determine rope_theta from the HF config")

        return cls(
            vocab_size=hf.vocab_size,
            hidden_size=hf.hidden_size,
            tie_word_embeddings=getattr(hf, "tie_word_embeddings", True),
            num_hidden_layers=hf.num_hidden_layers,
            intermediate_size=hf.intermediate_size,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=getattr(hf, "num_key_value_heads", hf.num_attention_heads),
            head_dim=head_dim,
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta=float(rope_theta),
            max_position_embeddings=getattr(hf, "max_position_embeddings", 4096),
            dtype=dtype,
        )
