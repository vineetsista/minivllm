"""Drop-in demo: drive the from-scratch engine with the official OpenAI SDK.

Start the server, then run this. It points the real `openai` client at the local
engine — no OpenAI account or key needed (the key is ignored) — and runs a
non-streaming and a streaming chat completion.

    pip install openai
    python -m uvicorn minivllm.server:app --port 8000
    python -m scripts.openai_client_demo
"""

from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--prompt", default="Explain KV caching in one sentence.")
    args = ap.parse_args()

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    print(">> non-streaming /v1/chat/completions")
    r = client.chat.completions.create(
        model="minivllm",
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=96,
    )
    print(r.choices[0].message.content)
    print(f"[usage] {r.usage.total_tokens} tokens\n")

    print(">> streaming")
    for chunk in client.chat.completions.create(
        model="minivllm",
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=96,
        stream=True,
    ):
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
    print("\n\nThe official OpenAI SDK just talked to a from-scratch engine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
