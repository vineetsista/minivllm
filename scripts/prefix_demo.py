"""CLI: RadixAttention prefix caching — the TTFT win on shared prompts.

Many requests share a long system prompt / preamble. Without prefix caching each
one recomputes that prefix's KV; with it, the prefix is computed once and reused.
This times prefill for a batch of requests that share a long system prompt, with
the cache off vs on, and reports the speedup, hit-rate, and prefix tokens reused.

Usage:
    python -m scripts.prefix_demo --shared 400 --requests 8
"""

from __future__ import annotations

import argparse
import time

from rich.console import Console
from rich.table import Table

from minivllm.loader import load_model
from minivllm.prefix_cache import RadixPrefixCache, cached_prefill

console = Console()

_SYSTEM = (
    "You are an expert assistant. Follow the instructions carefully, think step by "
    "step, be concise and accurate, cite assumptions, and never fabricate facts. "
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=16)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    console.print(f"[bold]Loading[/bold] {args.model} ...")
    model, cfg = load_model(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)

    # A long shared system prompt + distinct user questions => shared prefix.
    shared = _SYSTEM * 6
    prompts = [
        shared + f"\n\nUser question {i}: explain concept number {i}." for i in range(args.requests)
    ]
    token_lists = [tok(p, return_tensors="pt").input_ids[0].tolist() for p in prompts]
    shared_len = len(tok(shared, return_tensors="pt").input_ids[0])
    console.print(
        f"{args.requests} requests · ~{shared_len} shared prefix tokens · block {args.block_size}\n"
    )

    def run(prefix_cache: RadixPrefixCache | None) -> tuple[float, list[float]]:
        per = []
        t0 = time.perf_counter()
        for ids in token_lists:
            s = time.perf_counter()
            cached_prefill(model, ids, prefix_cache, dtype=cfg.dtype)
            per.append(time.perf_counter() - s)
        return time.perf_counter() - t0, per

    # warmup
    cached_prefill(model, token_lists[0], None, dtype=cfg.dtype)

    off_total, off_per = run(None)
    pc = RadixPrefixCache(cfg, block_size=args.block_size, capacity_blocks=256)
    on_total, on_per = run(pc)

    table = Table(title="Prefill time per request (shared system prompt)")
    table.add_column("request", justify="right")
    table.add_column("no prefix cache (ms)", justify="right")
    table.add_column("with prefix cache (ms)", justify="right")
    for i, (a, b) in enumerate(zip(off_per, on_per, strict=False)):
        table.add_row(str(i), f"{a * 1000:.0f}", f"{b * 1000:.0f}")
    console.print(table)

    st = pc.stats()
    console.print(
        f"\n[bold]Total prefill:[/bold] {off_total * 1000:.0f} ms -> {on_total * 1000:.0f} ms "
        f"([bold green]{off_total / on_total:.1f}x faster[/bold green])\n"
        f"[bold]Cache hit-rate:[/bold] {100 * st['hit_rate']:.0f}% · "
        f"prefix tokens reused: {st['prefix_tokens_reused']} · "
        f"cached blocks: {st['cached_blocks']}\n"
        f"[dim]The first request pays full prefill and warms the cache; the rest reuse "
        f"the shared prefix and only forward their unique suffix. On a server this is a "
        f"direct time-to-first-token drop under shared-prompt load.[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
