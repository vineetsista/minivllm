"""Naive autoregressive generation — the Phase 2 baseline.

Deliberately *unoptimized*: every decode step re-runs the model over the entire
sequence so far (prompt + everything generated). There is no KV cache, so step t
recomputes attention over all t tokens and the work grows O(n^2) in sequence
length. That inefficiency is exactly what Phase 3 (KV cache) removes; locking in
this slow baseline is the point, so the before/after is honest.

The loop already threads explicit positions through the model the same way the
cached path will, so Phase 3 becomes a localized change rather than a rewrite.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from minivllm.model import Qwen3ForCausalLM


@dataclass
class SamplingParams:
    """How to turn logits into the next token, and when to stop."""

    max_new_tokens: int = 64
    temperature: float = 0.0  # 0.0 => greedy (argmax); >0 => sample
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


@dataclass
class GenerationOutput:
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    # Timing is recorded per forward pass so the benchmark can derive TTFT and
    # per-token decode latency without re-running anything.
    prefill_seconds: float  # time of the first forward (prompt -> first token) = TTFT
    decode_seconds: list[float]  # one entry per token generated after the first
    stopped_on_eos: bool

    @property
    def num_generated(self) -> int:
        return len(self.generated_token_ids)

    @property
    def total_seconds(self) -> float:
        return self.prefill_seconds + sum(self.decode_seconds)


def _normalize_eos(eos_token_id: int | list[int] | None) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    return set(eos_token_id)


def _select_next_token(
    logits: torch.Tensor,
    params: SamplingParams,
    generator: torch.Generator | None,
) -> int:
    """Pick the next token id from a [vocab] logit vector."""
    if params.greedy:
        return int(logits.argmax(dim=-1).item())

    logits = logits / params.temperature

    if params.top_k is not None:
        k = min(params.top_k, logits.size(-1))
        kth = torch.topk(logits, k).values[..., -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)

    if params.top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        # Keep tokens up to and including the first that crosses top_p (shift so
        # at least one token always survives).
        remove = cumulative - probs > params.top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def _prepare(input_ids, params, device):
    """Shared setup: pull out the prompt ids and the (optional) RNG."""
    if input_ids.dim() == 2:
        if input_ids.size(0) != 1:
            raise ValueError("generate handles batch size 1; batching lands in Phase 5")
        prompt_ids = input_ids[0].tolist()
    else:
        prompt_ids = input_ids.tolist()
    generator: torch.Generator | None = None
    if params.seed is not None and not params.greedy:
        generator = torch.Generator(device=device).manual_seed(params.seed)
    return prompt_ids, generator


@torch.no_grad()
def generate(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    params: SamplingParams | None = None,
    eos_token_id: int | list[int] | None = None,
    use_cache: bool = False,
) -> GenerationOutput:
    """Autoregressive decode for a single sequence.

    `input_ids` is [1, prompt_len] or [prompt_len]. Batch size > 1 is out of
    scope until Phase 5.

    use_cache=False — the naive baseline: every step re-runs the model over the
    whole growing sequence (O(n^2) total).
    use_cache=True  — Phase 3 KV cache: prefill the prompt once, then feed only
    the new token each step and attend it against the cached history (O(n)).
    Both paths produce identical tokens; only the work differs.
    """
    params = params or SamplingParams()
    device = input_ids.device
    prompt_ids, generator = _prepare(input_ids, params, device)
    eos_set = _normalize_eos(eos_token_id)

    if use_cache:
        return _generate_cached(model, prompt_ids, params, eos_set, generator, device)
    return _generate_naive(model, prompt_ids, params, eos_set, generator, device)


def _generate_naive(model, prompt_ids, params, eos_set, generator, device) -> GenerationOutput:
    seq = list(prompt_ids)
    generated: list[int] = []
    decode_seconds: list[float] = []
    stopped_on_eos = False

    # First forward processes the whole prompt -> first token. This is TTFT.
    t0 = time.perf_counter()
    logits = model(torch.tensor([seq], device=device))[0, -1]
    token = _select_next_token(logits, params, generator)
    prefill_seconds = time.perf_counter() - t0
    seq.append(token)
    generated.append(token)
    stopped_on_eos = token in eos_set

    # Remaining tokens: each re-runs the full (growing) sequence — the naive cost.
    while not stopped_on_eos and len(generated) < params.max_new_tokens:
        t = time.perf_counter()
        logits = model(torch.tensor([seq], device=device))[0, -1]
        token = _select_next_token(logits, params, generator)
        decode_seconds.append(time.perf_counter() - t)
        seq.append(token)
        generated.append(token)
        stopped_on_eos = token in eos_set

    return GenerationOutput(
        prompt_token_ids=prompt_ids,
        generated_token_ids=generated,
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        stopped_on_eos=stopped_on_eos,
    )


def _generate_cached(model, prompt_ids, params, eos_set, generator, device) -> GenerationOutput:
    from minivllm.cache import KVCache

    dtype = next(model.parameters()).dtype
    cache = KVCache(
        model.cfg,
        max_seq_len=len(prompt_ids) + params.max_new_tokens,
        device=device,
        dtype=dtype,
    )

    generated: list[int] = []
    decode_seconds: list[float] = []
    stopped_on_eos = False

    # Prefill: the whole prompt goes in once and fills the cache. TTFT.
    t0 = time.perf_counter()
    logits = model(torch.tensor([prompt_ids], device=device), cache=cache)[0, -1]
    token = _select_next_token(logits, params, generator)
    prefill_seconds = time.perf_counter() - t0
    generated.append(token)
    stopped_on_eos = token in eos_set

    # Decode: feed only the newest token; positions/mask come from cache.length.
    while not stopped_on_eos and len(generated) < params.max_new_tokens:
        t = time.perf_counter()
        logits = model(torch.tensor([[token]], device=device), cache=cache)[0, -1]
        token = _select_next_token(logits, params, generator)
        decode_seconds.append(time.perf_counter() - t)
        generated.append(token)
        stopped_on_eos = token in eos_set

    return GenerationOutput(
        prompt_token_ids=prompt_ids,
        generated_token_ids=generated,
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        stopped_on_eos=stopped_on_eos,
    )
