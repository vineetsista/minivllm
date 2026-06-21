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

import queue
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import torch
from pydantic import BaseModel

from minivllm.batched_cache import BatchedKVCache
from minivllm.generate import SamplingParams, _normalize_eos, _select_next_token
from minivllm.model import Qwen3ForCausalLM
from minivllm.prefix_cache import RadixPrefixCache, cached_prefill

# Sentinel pushed to a request's token stream to mark completion.
_STREAM_DONE = None


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
    stream: bool = False


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
    # Live token stream: the worker pushes each generated id as it is produced,
    # then _STREAM_DONE. Endpoints that don't stream just wait on `event`.
    stream_q: queue.Queue = field(default_factory=queue.Queue)
    result: list[int] | None = None
    error: str | None = None
    queued_at: float = 0.0
    first_token_at: float = 0.0


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
        prefix_cache: bool = False,
        prefix_cache_blocks: int = 128,
    ):
        self.model = model
        self.cfg = model.cfg
        self.max_slots = max_slots
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = next(model.parameters()).dtype
        self.eos = _normalize_eos(eos_token_id)

        self.prefix_cache: RadixPrefixCache | None = None
        if prefix_cache:
            self.prefix_cache = RadixPrefixCache(
                self.cfg, block_size, prefix_cache_blocks, device, self.dtype
            )

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
        # Metrics (worker writes under the lock; readers snapshot).
        self.completed = 0
        self.generated_tokens = 0
        self.tokens_emitted = 0  # live, incremented per token (for throughput)
        self.prompt_tokens = 0
        self.decode_steps = 0
        self._recent_latency: deque[float] = deque(maxlen=256)  # end-to-end seconds
        self._recent_ttft: deque[float] = deque(maxlen=256)  # submit -> first token

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
        # Reuse any cached prompt prefix; tmp holds the full prompt K/V either way.
        logits, tmp = cached_prefill(
            self.model, req.prompt_ids, self.prefix_cache, self.device, self.dtype
        )
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
        req.stream_q.put(_STREAM_DONE)
        now = time.perf_counter()
        with self._cond:
            self.completed += 1
            self.generated_tokens += len(generated)
            self.prompt_tokens += len(req.prompt_ids)
            self._recent_latency.append(now - req.queued_at)
            if req.first_token_at:
                self._recent_ttft.append(req.first_token_at - req.queued_at)
        req.event.set()

    def metrics(self) -> dict:
        """Snapshot of serving metrics (thread-safe enough for a dashboard)."""
        with self._cond:
            lat = sorted(self._recent_latency)
            ttft = sorted(self._recent_ttft)
            completed = self.completed
            gen = self.generated_tokens
            emitted = self.tokens_emitted
            prompt = self.prompt_tokens
            steps = self.decode_steps

        def pct(xs: list[float], q: float) -> float:
            if not xs:
                return 0.0
            return xs[min(len(xs) - 1, int(q / 100 * len(xs)))]

        cap = getattr(self.cache, "num_free", None)  # paged pool only
        return {
            "completed_requests": completed,
            "generated_tokens": gen,
            "tokens_emitted": emitted,
            "prompt_tokens": prompt,
            "decode_steps": steps,
            "tokens_per_decode_step": (emitted / steps) if steps else 0.0,
            "queue_depth": len(self._waiting),
            "active_slots": sum(s is not None for s in self.slots),
            "max_slots": self.max_slots,
            "latency_p50_s": pct(lat, 50),
            "latency_p99_s": pct(lat, 99),
            "ttft_p50_s": pct(ttft, 50),
            "ttft_p99_s": pct(ttft, 99),
            "kv_blocks_free": cap,
            "kv_blocks_total": getattr(self.cache, "num_blocks", None),  # paged pool only
            "paged": self.paged,
            "prefix_cache": self.prefix_cache is not None,
            **(
                {
                    "prefix_hit_rate": self.prefix_cache.hit_rate,
                    "prefix_tokens_reused": self.prefix_cache.prefix_tokens_reused,
                    "prefix_cached_blocks": self.prefix_cache.n_blocks,
                }
                if self.prefix_cache is not None
                else {}
            ),
        }

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
                req.first_token_at = time.perf_counter()
                req.result = generated  # staging area the decode loop appends to
                req.stream_q.put(first)
                self.tokens_emitted += 1
                if self._is_finished(generated, first, req):
                    self._finish(i, generated, req)
                else:
                    self.slots[i] = _Slot(req=req, next_token=first, generator=gen)

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
            with self._cond:
                self.decode_steps += 1

            for i, slot in active:
                token = _select_next_token(logits[i, 0], slot.req.params, slot.generator)
                assert slot.req.result is not None  # set at admission
                slot.req.result.append(token)
                slot.next_token = token
                slot.req.stream_q.put(token)
                self.tokens_emitted += 1
                if self._is_finished(slot.req.result, token, slot.req):
                    self._finish(i, slot.req.result, slot.req)


# --- streaming helpers ----------------------------------------------------------


async def iter_token_deltas(req: _ServeReq, tok):
    """Async generator of decoded text deltas as the worker produces tokens.

    Uses incremental detokenization (decode the running id list, emit the new
    suffix) so multi-token characters and BPE merges render correctly. The
    blocking queue read is parked off the event loop with asyncio.to_thread.
    """
    import asyncio

    ids: list[int] = []
    prev = ""
    while True:
        token = await asyncio.to_thread(req.stream_q.get)
        if token is _STREAM_DONE:
            break
        ids.append(token)
        full = tok.decode(ids, skip_special_tokens=True)
        delta = full[len(prev) :]
        prev = full
        if delta:
            yield delta


def params_from_request(
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> SamplingParams:
    return SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )


# --- FastAPI app ----------------------------------------------------------------


def create_app(
    model_id: str = "Qwen/Qwen3-0.6B",
    max_slots: int = 4,
    max_seq_len: int = 1024,
    paged: bool = False,
    prefix_cache: bool = False,
):
    import asyncio
    import json
    from pathlib import Path

    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse

    state: dict = {}

    @asynccontextmanager
    async def lifespan(app):
        from transformers import AutoTokenizer

        from minivllm.loader import load_model

        model, _ = load_model(model_id)
        tok = AutoTokenizer.from_pretrained(model_id)
        state["tok"] = tok
        state["engine"] = ServingEngine(
            model,
            max_slots=max_slots,
            max_seq_len=max_seq_len,
            eos_token_id=tok.eos_token_id,
            paged=paged,
            prefix_cache=prefix_cache,
        )
        state["model_id"] = model_id
        yield
        state["engine"].shutdown()

    app = FastAPI(title="mini-vLLM", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    def _submit(prompt_ids, params, max_new):
        try:
            return state["engine"].submit(prompt_ids, max_new, params)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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
        return state["engine"].metrics()

    @app.get("/cache")
    async def cache_tree():
        engine, tok = state["engine"], state["tok"]
        pc = engine.prefix_cache
        if pc is None:
            return {"enabled": False, "nodes": [], "stats": {}}
        nodes = pc.snapshot()
        for n in nodes:  # decode a short text preview per cached block
            n["preview"] = tok.decode(n.pop("tokens")[:8], skip_special_tokens=True)[:24]
        return {"enabled": True, "nodes": nodes, "stats": pc.stats()}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        m = state["engine"].metrics()
        lines = [
            "# mini-vLLM serving metrics (Prometheus exposition)",
            f"minivllm_completed_requests_total {m['completed_requests']}",
            f"minivllm_generated_tokens_total {m['generated_tokens']}",
            f"minivllm_prompt_tokens_total {m['prompt_tokens']}",
            f"minivllm_decode_steps_total {m['decode_steps']}",
            f"minivllm_tokens_per_decode_step {m['tokens_per_decode_step']:.4f}",
            f"minivllm_queue_depth {m['queue_depth']}",
            f"minivllm_active_slots {m['active_slots']}",
            f"minivllm_max_slots {m['max_slots']}",
            f'minivllm_latency_seconds{{quantile="0.5"}} {m["latency_p50_s"]:.4f}',
            f'minivllm_latency_seconds{{quantile="0.99"}} {m["latency_p99_s"]:.4f}',
            f'minivllm_ttft_seconds{{quantile="0.5"}} {m["ttft_p50_s"]:.4f}',
            f'minivllm_ttft_seconds{{quantile="0.99"}} {m["ttft_p99_s"]:.4f}',
        ]
        if m["kv_blocks_free"] is not None:
            lines.append(f"minivllm_kv_blocks_free {m['kv_blocks_free']}")
        return "\n".join(lines) + "\n"

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        html = Path(__file__).parent / "dashboard.html"
        return html.read_text(encoding="utf-8")

    @app.post("/generate", response_model=None)
    async def generate_endpoint(body: GenerateRequest):
        tok = state["tok"]
        prompt_ids = tok(body.prompt, return_tensors="pt").input_ids[0].tolist()
        params = params_from_request(
            body.max_new_tokens, body.temperature, body.top_k, body.top_p, body.seed
        )
        req = _submit(prompt_ids, params, body.max_new_tokens)

        if body.stream:

            async def sse():
                async for delta in iter_token_deltas(req, tok):
                    yield f"data: {json.dumps({'text': delta})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        t0 = time.perf_counter()
        await asyncio.to_thread(req.event.wait)
        if req.error:
            raise HTTPException(status_code=500, detail=req.error)
        return GenerateResponse(
            text=tok.decode(req.result, skip_special_tokens=True),
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(req.result or []),
            latency_s=time.perf_counter() - t0,
        )

    # OpenAI-compatible /v1 routes (chat/completions, completions, models).
    from minivllm.openai_api import register_openai_routes

    register_openai_routes(app, state, _submit)

    return app


# `uvicorn minivllm.server:app` — model id / slots come from env.
def _app_from_env():
    import os

    return create_app(
        model_id=os.environ.get("MINIVLLM_MODEL", "Qwen/Qwen3-0.6B"),
        max_slots=int(os.environ.get("MINIVLLM_SLOTS", "4")),
        max_seq_len=int(os.environ.get("MINIVLLM_MAX_SEQ_LEN", "1024")),
        paged=os.environ.get("MINIVLLM_PAGED", "0") == "1",
        prefix_cache=os.environ.get("MINIVLLM_PREFIX", "0") == "1",
    )


app = _app_from_env()
