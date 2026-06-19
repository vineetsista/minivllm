"""Validate our forward pass against the HuggingFace reference.

This is the Phase 1 acceptance gate: before any optimization, our logits must
match the reference. We check three things on the same input:
  1. argmax agreement at every position (the literal "token-for-token" claim)
  2. max / mean absolute logit difference (numerical closeness)
  3. agreement of the top-k next-token candidates for the final position
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from minivllm.loader import load_model


@dataclass
class ComparisonResult:
    prompt: str
    seq_len: int
    max_abs_diff: float
    mean_abs_diff: float
    argmax_match_frac: float  # fraction of positions where argmax agrees
    topk_match: bool          # do the top-k final-token candidates agree (as a set)
    topk: int
    ours_next_token: int
    ref_next_token: int

    @property
    def passed(self) -> bool:
        return self.argmax_match_frac == 1.0 and self.ours_next_token == self.ref_next_token


@torch.no_grad()
def compare_to_reference(
    model_id: str = "Qwen/Qwen3-0.6B",
    prompt: str = "The capital of France is",
    topk: int = 5,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> ComparisonResult:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)

    # Our implementation
    ours_model, _ = load_model(model_id, dtype=dtype, device=device)
    ours_logits = ours_model(input_ids)

    # HuggingFace reference, same dtype/device. (transformers>=5 renamed the
    # arg torch_dtype -> dtype; we target 5.x so we use dtype.)
    ref_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    ref_logits = ref_model(input_ids).logits

    diff = (ours_logits - ref_logits).abs()
    ours_argmax = ours_logits.argmax(dim=-1)
    ref_argmax = ref_logits.argmax(dim=-1)
    argmax_match_frac = (ours_argmax == ref_argmax).float().mean().item()

    ours_topk = set(ours_logits[0, -1].topk(topk).indices.tolist())
    ref_topk = set(ref_logits[0, -1].topk(topk).indices.tolist())

    return ComparisonResult(
        prompt=prompt,
        seq_len=input_ids.shape[1],
        max_abs_diff=diff.max().item(),
        mean_abs_diff=diff.mean().item(),
        argmax_match_frac=argmax_match_frac,
        topk_match=ours_topk == ref_topk,
        topk=topk,
        ours_next_token=int(ours_argmax[0, -1].item()),
        ref_next_token=int(ref_argmax[0, -1].item()),
    )
