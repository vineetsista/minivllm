"""CLI: async load test against the mini-vLLM server — the Phase 8 numbers.

Fires N requests at a target concurrency and reports end-to-end latency
percentiles and throughput. This is the serving-side view of everything the
engine does: continuous batching means concurrent requests share decode steps,
so aggregate throughput climbs with concurrency until the slots saturate.

Start the server first:
    python -m uvicorn minivllm.server:app --port 8000
Then:
    python -m scripts.loadtest --n 16 --concurrency 8 --max-new-tokens 32
"""

from __future__ import annotations

import argparse
import asyncio
import time

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

_PROMPTS = [
    "The capital of France is",
    "Write one sentence about the ocean:",
    "List three primary colors:",
    "Explain gravity in one line:",
]


async def _one(client, url, prompt, max_new):
    t0 = time.perf_counter()
    r = await client.post(url, json={"prompt": prompt, "max_new_tokens": max_new}, timeout=120.0)
    r.raise_for_status()
    data = r.json()
    return time.perf_counter() - t0, data["generated_tokens"]


async def _run(args) -> int:
    import httpx

    url = f"{args.base_url}/generate"
    sem = asyncio.Semaphore(args.concurrency)
    latencies: list[float] = []
    tokens = 0

    async with httpx.AsyncClient() as client:
        # Readiness check.
        for _ in range(60):
            try:
                h = await client.get(f"{args.base_url}/health", timeout=5.0)
                if h.status_code == 200 and h.json().get("status") == "ok":
                    break
            except Exception:
                pass
            await asyncio.sleep(1.0)
        else:
            console.print("[red]server not ready[/red]")
            return 1

        async def task(i):
            nonlocal tokens
            async with sem:
                lat, gen = await _one(client, url, _PROMPTS[i % len(_PROMPTS)], args.max_new_tokens)
                latencies.append(lat)
                tokens += gen

        console.print(
            f"[bold]Load test[/bold] · {args.n} requests · concurrency {args.concurrency} "
            f"· {args.max_new_tokens} new tokens"
        )
        t0 = time.perf_counter()
        await asyncio.gather(*(task(i) for i in range(args.n)))
        wall = time.perf_counter() - t0

    lat = np.asarray(latencies)
    table = Table(title="Phase 8 — serving load test")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("requests", str(args.n))
    table.add_row("concurrency", str(args.concurrency))
    table.add_row("wall time", f"{wall:.2f} s")
    table.add_row("latency p50", f"{np.percentile(lat, 50):.2f} s")
    table.add_row("latency p99", f"{np.percentile(lat, 99):.2f} s")
    table.add_row("request throughput", f"{args.n / wall:.2f} req/s")
    table.add_row("token throughput", f"{tokens / wall:.2f} tok/s")
    console.print(table)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
