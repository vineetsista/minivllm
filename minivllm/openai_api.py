"""OpenAI-compatible API — point the `openai` SDK (or LangChain, Open WebUI,
curl examples) at this from-scratch engine.

Implements the subset that matters: `/v1/models`, `/v1/completions`, and
`/v1/chat/completions`, each in both non-streaming and SSE-streaming form, in
OpenAI's exact response shapes. Chat requests are rendered through the model's
own chat template. Tokens flow from the same continuous-batching ServingEngine
that backs the native API — the OpenAI layer is pure translation.

Pydantic models live at module scope so FastAPI resolves their (string)
annotations against module globals; a model defined inside the registrar would be
misread as a query parameter under `from __future__ import annotations`.
"""

from __future__ import annotations

import asyncio
import json
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from minivllm.server import build_constraint, iter_token_deltas, params_from_request, regex_for


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "minivllm"
    messages: list[ChatMessage]
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float | None = None
    seed: int | None = None
    stream: bool = False
    response_format: dict | None = None  # {"type": "json_object"} or "json_schema"


class CompletionRequest(BaseModel):
    model: str = "minivllm"
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float | None = None
    seed: int | None = None
    stream: bool = False
    response_format: dict | None = None


def _finish_reason(n_generated: int, max_new: int) -> str:
    return "length" if n_generated >= max_new else "stop"


def register_openai_routes(app: FastAPI, state: dict, submit) -> None:
    """Attach the /v1 routes. `submit(prompt_ids, params, max_new) -> _ServeReq`
    is the server's shared admission helper; `state` holds the tokenizer."""

    def _prompt_ids_for_chat(messages: list[ChatMessage]) -> list[int]:
        tok = state["tok"]
        rendered = tok.apply_chat_template(
            [m.model_dump() for m in messages], tokenize=False, add_generation_prompt=True
        )
        return tok(rendered, return_tensors="pt").input_ids[0].tolist()

    @app.get("/v1/models")
    async def list_models():
        mid = state.get("model_id", "minivllm")
        return {
            "object": "list",
            "data": [{"id": mid, "object": "model", "owned_by": "minivllm"}],
        }

    # -- chat completions --------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        tok = state["tok"]
        prompt_ids = _prompt_ids_for_chat(body.messages)
        params = params_from_request(
            body.max_tokens, body.temperature, top_p=body.top_p, seed=body.seed
        )
        params.constraint = build_constraint(state, regex_for(response_format=body.response_format))
        req = submit(prompt_ids, params, body.max_tokens)
        created = int(time.time())
        cid = f"chatcmpl-{req.id}"

        if body.stream:

            async def sse():
                head = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body.model,
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(head)}\n\n"
                async for delta in iter_token_deltas(req, tok):
                    chunk = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": body.model,
                        "choices": [
                            {"index": 0, "delta": {"content": delta}, "finish_reason": None}
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                tail = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(tail)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        await asyncio.to_thread(req.event.wait)
        if req.error:
            raise HTTPException(status_code=500, detail=req.error)
        result = req.result or []
        text = tok.decode(result, skip_special_tokens=True)
        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": _finish_reason(len(result), body.max_tokens),
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(result),
                "total_tokens": len(prompt_ids) + len(result),
            },
        }

    # -- text completions --------------------------------------------------------

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest):
        tok = state["tok"]
        prompt_ids = tok(body.prompt, return_tensors="pt").input_ids[0].tolist()
        params = params_from_request(
            body.max_tokens, body.temperature, top_p=body.top_p, seed=body.seed
        )
        params.constraint = build_constraint(state, regex_for(response_format=body.response_format))
        req = submit(prompt_ids, params, body.max_tokens)
        created = int(time.time())
        cid = f"cmpl-{req.id}"

        if body.stream:

            async def sse():
                async for delta in iter_token_deltas(req, tok):
                    chunk = {
                        "id": cid,
                        "object": "text_completion",
                        "created": created,
                        "model": body.model,
                        "choices": [{"index": 0, "text": delta, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        await asyncio.to_thread(req.event.wait)
        if req.error:
            raise HTTPException(status_code=500, detail=req.error)
        result = req.result or []
        return {
            "id": cid,
            "object": "text_completion",
            "created": created,
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "text": tok.decode(result, skip_special_tokens=True),
                    "finish_reason": _finish_reason(len(result), body.max_tokens),
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(result),
                "total_tokens": len(prompt_ids) + len(result),
            },
        }
