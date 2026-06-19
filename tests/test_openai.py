"""HTTP-layer gates for the OpenAI-compatible API (slow: loads the real model).

Named with the `real_model` hint so conftest skips them unless --runslow. These
drive the FastAPI app through Starlette's TestClient (which runs the lifespan, so
the real tokenizer + chat template are exercised), covering the request/response
serialization the unit tests can't reach.
"""

from __future__ import annotations

import json


def _client():
    from fastapi.testclient import TestClient

    from minivllm.server import create_app

    return TestClient(create_app(max_slots=2, max_seq_len=256))


def test_openai_chat_completion_real_model():
    with _client() as client:
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["usage"]["completion_tokens"] >= 1
        assert body["choices"][0]["finish_reason"] in {"stop", "length"}


def test_openai_chat_streaming_real_model():
    with _client() as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say hi."}],
                "max_tokens": 8,
                "stream": True,
            },
        )
        assert r.status_code == 200
        chunks = [
            ln[6:]
            for ln in r.text.splitlines()
            if ln.startswith("data: ") and ln[6:].strip() != "[DONE]"
        ]
        assert chunks, "expected streamed chunks"
        first = json.loads(chunks[0])
        assert first["object"] == "chat.completion.chunk"
        assert r.text.rstrip().endswith("[DONE]")


def test_openai_completions_and_metrics_real_model():
    with _client() as client:
        r = client.post("/v1/completions", json={"prompt": "Hello", "max_tokens": 6})
        assert r.status_code == 200
        assert r.json()["object"] == "text_completion"

        models = client.get("/v1/models").json()
        assert models["data"][0]["object"] == "model"

        metrics = client.get("/metrics").text
        assert "minivllm_completed_requests_total" in metrics
