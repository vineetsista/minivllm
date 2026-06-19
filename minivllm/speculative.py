"""Speculative decoding — Phase 7.

A cheap *drafter* proposes K tokens; the target model verifies all K in a single
forward and accepts the longest prefix it agrees with, plus one free "bonus"
token. For greedy decoding this is **exact**: the output is identical to plain
target greedy decode, token-for-token — speculation only changes how many target
forward passes it takes to get there.

The lever is "tokens accepted per target forward". On a memory-bandwidth-bound
GPU decode, verifying K tokens costs about the same as decoding one, so accepting
several per forward is a direct latency win. On this CPU reference (compute-bound
at 0.6B) a K-token forward costs ~K x a single token, so wall-clock is roughly
neutral — but the algorithmic win (fewer sequential target steps, high
acceptance) is what we validate here and what converts to latency on GPU.

Drafter is an interface. The default `NgramDrafter` needs no second model — it
proposes the continuation of the longest recent token n-gram that recurs in the
history (prompt-lookup decoding), which is strong on the repetitive/structured
text where drafting pays off. A draft *model* implements the same `propose`
contract (see `ModelDrafter`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import torch

from minivllm.cache import KVCache
from minivllm.model import Qwen3ForCausalLM


class Drafter(Protocol):
    def propose(self, history: list[int], k: int) -> list[int]:
        """Propose up to k continuation tokens for `history` (may return fewer)."""
        ...


class NgramDrafter:
    """Prompt-lookup drafting: find the longest suffix n-gram of the history that
    occurred earlier, and propose the tokens that followed it last time."""

    def __init__(self, max_ngram: int = 3, min_ngram: int = 1):
        self.max_ngram = max_ngram
        self.min_ngram = min_ngram

    def propose(self, history: list[int], k: int) -> list[int]:
        n_hist = len(history)
        for n in range(min(self.max_ngram, n_hist - 1), self.min_ngram - 1, -1):
            ngram = history[-n:]
            # Search earlier occurrences, most recent first (skip the suffix itself).
            for i in range(n_hist - n - 1, -1, -1):
                if history[i : i + n] == ngram:
                    return history[i + n : i + n + k]
        return []


class ModelDrafter:
    """A draft *model* as the proposer: greedily roll out k tokens. Same
    `propose` contract as the n-gram drafter; needs a separate (smaller) draft
    checkpoint sharing the target's tokenizer, so it isn't exercised on the
    CPU-only box, but it documents the model-based path."""

    def __init__(self, draft_model: Qwen3ForCausalLM, device: str = "cpu"):
        self.model = draft_model
        self.device = device

    @torch.no_grad()
    def propose(self, history: list[int], k: int) -> list[int]:
        cache = KVCache(self.model.cfg, max_seq_len=len(history) + k, device=self.device,
                        dtype=next(self.model.parameters()).dtype)
        logits = self.model(torch.tensor([history], device=self.device), cache=cache)[0, -1]
        out = []
        token = int(logits.argmax())
        out.append(token)
        for _ in range(k - 1):
            logits = self.model(torch.tensor([[token]], device=self.device), cache=cache)[0, -1]
            token = int(logits.argmax())
            out.append(token)
        return out


@dataclass
class SpecStats:
    generated: int
    target_forwards: int  # verification forwards (excludes the prefill)
    draft_tokens: int
    accepted_tokens: int
    seconds: float

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / self.draft_tokens if self.draft_tokens else 0.0

    @property
    def tokens_per_forward(self) -> float:
        return self.generated / self.target_forwards if self.target_forwards else float("nan")


class SpeculativeDecoder:
    def __init__(
        self,
        target: Qwen3ForCausalLM,
        drafter: Drafter | None = None,
        k: int = 4,
        device: str = "cpu",
    ):
        self.target = target
        self.drafter = drafter or NgramDrafter()
        self.k = k
        self.device = device
        self.dtype = next(target.parameters()).dtype

    @torch.no_grad()
    def generate(self, prompt_ids: list[int], max_new_tokens: int = 64) -> tuple[list[int], SpecStats]:
        cache = KVCache(
            self.target.cfg,
            max_seq_len=len(prompt_ids) + max_new_tokens + self.k + 1,
            device=self.device,
            dtype=self.dtype,
        )

        t0 = time.perf_counter()
        # Prefill. The sampled `last` token is not yet in the cache (cache holds
        # exactly the prompt); it gets fed at the head of the first verification.
        logits = self.target(torch.tensor([prompt_ids], device=self.device), cache=cache)[0, -1]
        last = int(logits.argmax())
        history = list(prompt_ids) + [last]
        output = [last]

        target_forwards = 0
        draft_tokens = 0
        accepted_tokens = 0

        while len(output) < max_new_tokens:
            draft = self.drafter.propose(history, self.k)
            base = cache.length  # tokens committed before this forward

            # Feed [last, draft...] in one pass. Position i's logits predict the
            # token that should follow input token i.
            feed = [last] + draft
            logits = self.target(torch.tensor([feed], device=self.device), cache=cache)[0]
            target_preds = logits.argmax(dim=-1).tolist()  # len == len(feed)
            target_forwards += 1
            draft_tokens += len(draft)

            # Greedy accept: draft[i] is kept iff it equals the target's prediction
            # at the previous position. target_preds[0] is the token after `last`.
            m = 0
            for i in range(len(draft)):
                if draft[i] == target_preds[i]:
                    m += 1
                else:
                    break
            accepted_tokens += m

            # Accepted draft tokens + one correction/bonus (the target's own
            # prediction right after the accepted prefix).
            new_tokens = draft[:m] + [target_preds[m]]
            # Cached KV: `last` + accepted draft = base + m + 1; drop rejected.
            cache.truncate(base + m + 1)

            for t in new_tokens:
                output.append(t)
                history.append(t)
                if len(output) >= max_new_tokens:
                    break
            last = output[-1]

        seconds = time.perf_counter() - t0
        return output[:max_new_tokens], SpecStats(
            generated=len(output[:max_new_tokens]),
            target_forwards=target_forwards,
            draft_tokens=draft_tokens,
            accepted_tokens=accepted_tokens,
            seconds=seconds,
        )
