"""Continuous batching engine — Phase 5, the throughput multiplier.

The batched cache lets one decode step serve B slots at once. The remaining
question is *scheduling*: when do new requests enter the batch? Two policies,
run on identical compute so the comparison isolates the scheduler:

  * static    — admit a group of up to B requests, decode until *every* slot in
                the group has finished, then admit the next group. A slot that
                finishes early sits idle until the longest in its group drains.
                This is "wait for the static batch to finish."
  * continuous— admit a waiting request the moment *any* slot frees, every step.
                Slots stay full, so GPU/CPU work is never wasted on idle rows.
                This is iteration-level scheduling.

With uniform generation lengths the two are identical; the win shows up exactly
when lengths vary, which is the real-serving case. Output is byte-identical
between policies and matches single-sequence decode — only the schedule differs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from minivllm.batched_cache import BatchedKVCache
from minivllm.cache import KVCache
from minivllm.generate import SamplingParams, _normalize_eos, _select_next_token
from minivllm.model import Qwen3ForCausalLM


@dataclass
class Request:
    id: int
    prompt_ids: list[int]
    max_new_tokens: int


@dataclass
class _Slot:
    req: Request
    next_token: int  # token to feed on the next decode step
    generated: list[int]


@dataclass
class EngineStats:
    policy: str
    num_requests: int
    total_generated: int
    decode_steps: int  # number of batched forward passes
    seconds: float

    @property
    def requests_per_s(self) -> float:
        return self.num_requests / self.seconds if self.seconds else float("nan")

    @property
    def tokens_per_s(self) -> float:
        return self.total_generated / self.seconds if self.seconds else float("nan")

    @property
    def avg_batch_occupancy(self) -> float:
        """Mean live slots per decode step — how full the batch ran."""
        return self.total_generated / self.decode_steps if self.decode_steps else float("nan")


class ContinuousBatchingEngine:
    def __init__(
        self,
        model: Qwen3ForCausalLM,
        max_slots: int = 4,
        device: str = "cpu",
        eos_token_id: int | list[int] | None = None,
    ):
        self.model = model
        self.cfg = model.cfg
        self.max_slots = max_slots
        self.device = device
        self.dtype = next(model.parameters()).dtype
        self.eos = _normalize_eos(eos_token_id)

    @torch.no_grad()
    def _prefill(self, cache: BatchedKVCache, slot: int, req: Request, params, generator):
        """Run the prompt through a temp contiguous cache, copy K/V into `slot`,
        and return the first sampled token."""
        tmp = KVCache(
            self.cfg, max_seq_len=len(req.prompt_ids), device=self.device, dtype=self.dtype
        )
        ids = torch.tensor([req.prompt_ids], device=self.device)
        logits = self.model(ids, cache=tmp)[0, -1]
        cache.load_prefill(slot, tmp, len(req.prompt_ids))
        return _select_next_token(logits, params, generator)

    def _finished(self, slot: _Slot) -> bool:
        return len(slot.generated) >= slot.req.max_new_tokens or slot.next_token in self.eos

    @torch.no_grad()
    def run(
        self,
        requests: list[Request],
        params: SamplingParams | None = None,
        policy: str = "continuous",
    ) -> tuple[dict[int, list[int]], EngineStats]:
        import time

        if policy not in ("continuous", "static"):
            raise ValueError(f"unknown policy {policy!r}")
        params = params or SamplingParams(max_new_tokens=32, temperature=0.0)
        generator = None  # greedy in the batched path (deterministic comparison)

        max_len = max(len(r.prompt_ids) + r.max_new_tokens for r in requests)
        cache = BatchedKVCache(self.cfg, self.max_slots, max_len, self.device, self.dtype)

        waiting = deque(requests)
        slots: list[_Slot | None] = [None] * self.max_slots
        outputs: dict[int, list[int]] = {}
        decode_steps = 0

        def admit_into(slot_idx: int) -> None:
            req = waiting.popleft()
            first = self._prefill(cache, slot_idx, req, params, generator)
            s = _Slot(req=req, next_token=first, generated=[first])
            if self._finished(s):  # e.g. max_new_tokens == 1, or instant EOS
                outputs[req.id] = s.generated
                cache.release(slot_idx)
            else:
                slots[slot_idx] = s

        t0 = time.perf_counter()
        while waiting or any(slots):
            # Admission policy: continuous fills any free slot every step; static
            # only refills once the whole batch has drained.
            can_admit = (
                any(s is None for s in slots)
                if policy == "continuous"
                else all(s is None for s in slots)
            )
            if can_admit:
                for i in range(self.max_slots):
                    if waiting and slots[i] is None:
                        admit_into(i)

            active = [(i, s) for i, s in enumerate(slots) if s is not None]
            if not active:
                continue  # everything admitted this round finished at prefill

            # Build the batched decode step.
            input_ids = torch.zeros(self.max_slots, 1, dtype=torch.long, device=self.device)
            active_mask = torch.zeros(self.max_slots, dtype=torch.bool, device=self.device)
            for i, s in active:
                input_ids[i, 0] = s.next_token
                active_mask[i] = True

            attn_mask = cache.make_mask(self.dtype)
            logits = self.model.decode_step(input_ids, cache.position_ids(), attn_mask, cache)
            cache.advance(active_mask)
            decode_steps += 1

            for i, s in active:
                token = _select_next_token(logits[i, 0], params, generator)
                s.generated.append(token)
                s.next_token = token
                if self._finished(s):
                    outputs[s.req.id] = s.generated
                    cache.release(i)
                    slots[i] = None
        seconds = time.perf_counter() - t0

        stats = EngineStats(
            policy=policy,
            num_requests=len(requests),
            total_generated=sum(len(v) for v in outputs.values()),
            decode_steps=decode_steps,
            seconds=seconds,
        )
        return outputs, stats
