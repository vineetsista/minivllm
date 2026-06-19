"""Serving layer — Phase 8.

The offline engine (`engine.py`) runs a fixed list of requests to completion. A
server instead takes requests *as they arrive* and must keep the batch full while
new ones trickle in — continuous batching in streaming form.

`ServingEngine` runs a background worker thread that owns the model and the
batched KV cache. Requests are submitted to a thread-safe queue; every iteration
the worker admits waiting requests into free slots (prefill), runs one batched
decode step over all live slots, and signals each request's completion event when
it finishes. HTTP handlers just `submit` and await that event in a threadpool, so
the async event loop is never blocked by the (synchronous, CPU-bound) forward.

The worker is the single owner of the cache/slots, so those need no lock; the
lock guards only the waiting queue. Per-slot sampling params let different
requests use different temperatures in the same batch. The engine works on token
ids and knows nothing about tokenization — the FastAPI layer owns that.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import torch
from pydantic import BaseModel

from minivllm.batched_cache import BatchedKVCache
from minivllm.cache import KVCache
from minivllm.generate import SamplingParams, _normalize_eos, _select_next_token
from minivllm.model import Qwen3ForCausalLM


# Request/response models live at module scope: with `from __future__ import
# annotations` the annotations are strings, and FastAPI resolves them against
# module globals — a model defined inside create_app() would not resolve and
# would be misread as a query parameter.
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    generated_tokens: int
    latency_s: float


@dataclass
class _ServeReq:
    id: int
    prompt_ids: list[int]
    max_new_tokens: int
    params: SamplingParams
    event: threading.Event = field(default_factory=threading.Event)
    result: list[int] | None = None
    error: str | None = None
    queued_at: float = 0.0


@dataclass
class _Slot:
    req: _ServeReq
    next_token: int
    generator: torch.Generator | None


class ServingEngine:
    def __init__(
        self,
        model: Qwen3ForCausalLM,
        max_slots: int = 4,
        max_seq_len: int = 1024,
        device: str = "cpu",
        eos_token_id: int | list[int] | None = None,
        paged: bool = False,
        block_size: int = 16,
        num_blocks: int | None = None,
    ):
        self.model = model
        self.cfg = model.cfg
        self.max_slots = max_slots
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = next(model.parameters()).dtype
        self.eos = _normalize_eos(eos_token_id)

        self.paged = paged
        if paged:
            from minivllm.paged_batched_cache import PagedBatchedKVCache

            self.cache: BatchedKVCache | PagedBatchedKVCache = PagedBatchedKVCache(
                self.cfg, max_slots, max_seq_len, block_size, num_blocks, device, self.dtype
            )
        else:
            self.cache = BatchedKVCache(self.cfg, max_slots, max_seq_len, device, self.dtype)
        self.slots: list[_Slot | None] = [None] * max_slots

        self._waiting: deque[_ServeReq] = deque()
        self._cond = threading.Condition()
        self._running = True
        self._next_id = 0
        # Lightweight stats (worker-only writes except counters under lock).
        self.completed = 0
        self.generated_tokens = 0

        self._worker = threading.Thread(target=self._loop, name="minivllm-engine", daemon=True)
        self._worker.start()

    # -- public API --------------------------------------------------------------

    def submit(
        self, prompt_ids: list[int], max_new_tokens: int, params: SamplingParams
    ) -> _ServeReq:
        if len(prompt_ids) + max_new_tokens > self.max_seq_len:
            raise ValueError(
                f"prompt({len(prompt_ids)}) + max_new({max_new_tokens}) exceeds "
                f"server max_seq_len {self.max_seq_len}"
            )
        with self._cond:
            req = _ServeReq(
                self._next_id,
                list(prompt_ids),
                max_new_tokens,
                params,
                queued_at=time.perf_counter(),
            )
            self._next_id += 1
            self._waiting.append(req)
            self._cond.notify()
        return req

    def shutdown(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify_all()
        self._worker.join(timeout=5.0)

    # -- worker ------------------------------------------------------------------

    @torch.no_grad()
    def _prefill(self, slot_idx: int, req: _ServeReq) -> tuple[int, torch.Generator | None]:
        tmp = KVCache(
            self.cfg, max_seq_len=len(req.prompt_ids), device=self.device, dtype=self.dtype
        )
        ids = torch.tensor([req.prompt_ids], device=self.device)
        logits = self.model(ids, cache=tmp)[0, -1]
        self.cache.load_prefill(slot_idx, tmp, len(req.prompt_ids))
        gen = None
        if req.params.seed is not None and not req.params.greedy:
            gen = torch.Generator(device=self.device).manual_seed(req.params.seed)
        first = _select_next_token(logits, req.params, gen)
        return first, gen

    def _finish(self, slot_idx: int, generated: list[int], req: _ServeReq) -> None:
        req.result = generated
        self.cache.release(slot_idx)
        self.slots[slot_idx] = None
        with self._cond:
            self.completed += 1
            self.generated_tokens += len(generated)
        req.event.set()

    def _is_finished(self, generated: list[int], next_token: int, req: _ServeReq) -> bool:
        return len(generated) >= req.max_new_tokens or next_token in self.eos

    @torch.no_grad()
    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._running and not self._waiting and all(s is None for s in self.slots):
                    self._cond.wait()
                if not self._running:
                    return
                admit = []
                for i in range(self.max_slots):
                    if self.slots[i] is None and self._waiting:
                        admit.append((i, self._waiting.popleft()))

            # Prefill admitted requests (outside the lock — heavy torch work).
            for i, req in admit:
                try:
                    first, gen = self._prefill(i, req)
                except Exception as exc:  # surface prefill failures to the caller
                    req.error = str(exc)
                    req.event.set()
                    continue
                generated = [first]
                if self._is_finished(generated, first, req):
                    self._finish(i, generated, req)
                else:
                    self.slots[i] = _Slot(req=req, next_token=first, generator=gen)
                    # track partial generation on the slot via req.result staging
                    req.result = generated

            active = [(i, s) for i, s in enumerate(self.slots) if s is not None]
            if not active:
                continue

            input_ids = torch.zeros(self.max_slots, 1, dtype=torch.long, device=self.device)
            active_mask = torch.zeros(self.max_slots, dtype=torch.bool, device=self.device)
            for i, slot in active:
                input_ids[i, 0] = slot.next_token
                active_mask[i] = True

            attn_mask = self.cache.make_mask(self.dtype)
            logits = self.model.decode_step(
                input_ids, self.cache.position_ids(), attn_mask, self.cache
            )
            self.cache.advance(active_mask)

            for i, slot in active:
                token = _select_next_token(logits[i, 0], slot.req.params, slot.generator)
                assert slot.req.result is not None  # set at admission
                slot.req.result.append(token)
                slot.next_token = token
                if self._is_finished(slot.req.result, token, slot.req):
                    self._finish(i, slot.req.result, slot.req)


# --- FastAPI app ----------------------------------------------------------------


def create_app(model_id: str = "Qwen/Qwen3-0.6B", max_slots: int = 4, max_seq_len: int = 1024):

    from fastapi import FastAPI, HTTPException

    state: dict = {}

    @asynccontextmanager
    async def lifespan(app):
        from transformers import AutoTokenizer

        from minivllm.loader import load_model

        model, _ = load_model(model_id)
        tok = AutoTokenizer.from_pretrained(model_id)
        state["tok"] = tok
        state["engine"] = ServingEngine(
            model, max_slots=max_slots, max_seq_len=max_seq_len, eos_token_id=tok.eos_token_id
        )
        state["model_id"] = model_id
        yield
        state["engine"].shutdown()

    app = FastAPI(title="mini-vLLM", lifespan=lifespan)

    @app.get("/health")
    async def health():
        engine = state.get("engine")
        return {
            "status": "ok" if engine else "starting",
            "model": state.get("model_id"),
            "slots": engine.max_slots if engine else None,
        }

    @app.get("/stats")
    async def stats():
        engine = state["engine"]
        return {
            "completed_requests": engine.completed,
            "generated_tokens": engine.generated_tokens,
            "queue_depth": len(engine._waiting),
            "active_slots": sum(s is not None for s in engine.slots),
        }

    @app.post("/generate", response_model=GenerateResponse)
    async def generate_endpoint(body: GenerateRequest):
        import asyncio

        engine, tok = state["engine"], state["tok"]
        prompt_ids = tok(body.prompt, return_tensors="pt").input_ids[0].tolist()
        params = SamplingParams(
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
            top_p=body.top_p,
            seed=body.seed,
        )
        try:
            req = engine.submit(prompt_ids, body.max_new_tokens, params)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        t0 = time.perf_counter()
        await asyncio.to_thread(req.event.wait)  # park the blocking wait off the event loop
        if req.error:
            raise HTTPException(status_code=500, detail=req.error)
        latency = time.perf_counter() - t0

        return GenerateResponse(
            text=tok.decode(req.result, skip_special_tokens=True),
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(req.result),
            latency_s=latency,
        )

    return app


# `uvicorn minivllm.server:app` — model id / slots come from env.
def _app_from_env():
    import os

    return create_app(
        model_id=os.environ.get("MINIVLLM_MODEL", "Qwen/Qwen3-0.6B"),
        max_slots=int(os.environ.get("MINIVLLM_SLOTS", "4")),
        max_seq_len=int(os.environ.get("MINIVLLM_MAX_SEQ_LEN", "1024")),
    )


app = _app_from_env()
