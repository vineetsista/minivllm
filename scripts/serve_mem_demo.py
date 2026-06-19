"""CLI: what paging buys the *server* — KV memory and concurrency flexibility.

The contiguous batched cache reserves `num_slots * max_seq_len` of KV up front,
sized for the worst case (every slot full to the maximum length) whether or not
that ever happens. The paged server instead sizes a shared block pool to a memory
budget and admits requests until the pool is full (backpressure, not OOM), so the
same memory serves far more real traffic.

Weight-free (reads config.json only), so it runs instantly.

Usage:
    python -m scripts.serve_mem_demo --slots 16 --max-seq-len 4096 --pool-gib 2
"""

from __future__ import annotations

import argparse

import torch
from rich.console import Console
from rich.table import Table

from minivllm.config import ModelConfig
from minivllm.paged_cache import kv_bytes_per_token

console = Console()


def _gib(n_bytes: float) -> float:
    return n_bytes / (1024**3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--pool-gib", type=float, default=2.0, help="paged pool budget")
    ap.add_argument("--typical-len", type=int, default=256, help="typical in-flight length")
    args = ap.parse_args()

    cfg = ModelConfig.from_hf(args.model, dtype=torch.float16)
    bpt = kv_bytes_per_token(cfg, torch.float16)

    contiguous_tokens = args.slots * args.max_seq_len
    contiguous_mem = contiguous_tokens * bpt
    pool_tokens = int(args.pool_gib * (1024**3) / bpt)

    console.print(
        f"[bold]{args.model}[/bold] · float16 · KV {bpt / 1024:.0f} KiB/token · "
        f"{args.slots} slots · max_seq_len {args.max_seq_len}\n"
    )

    t = Table(title="Server KV memory: contiguous batched vs paged pool")
    t.add_column("scheme")
    t.add_column("reserved KV", justify="right")
    t.add_column("max concurrent (full-length) seqs", justify="right")
    t.add_column(f"or typical ({args.typical_len}-tok) seqs", justify="right")
    t.add_row(
        "contiguous (BatchedKVCache)",
        f"{_gib(contiguous_mem):.2f} GiB",
        str(args.slots),
        str(args.slots),  # still capped at num_slots; the rest of the reservation is wasted
    )
    t.add_row(
        f"paged pool ({args.pool_gib:.0f} GiB budget)",
        f"{args.pool_gib:.2f} GiB",
        str(pool_tokens // args.max_seq_len),
        str(pool_tokens // args.typical_len),
    )
    console.print(t)

    console.print(
        f"\nContiguous must reserve [red]{_gib(contiguous_mem):.1f} GiB[/red] for "
        f"{args.slots} slots regardless of real lengths. A paged pool of "
        f"[green]{args.pool_gib:.0f} GiB[/green] serves up to "
        f"[bold]{pool_tokens // args.typical_len}[/bold] typical "
        f"({args.typical_len}-token) sequences concurrently — memory tracks tokens "
        f"actually in flight, and the pool is sized to budget instead of worst case. "
        f"When it fills, admission waits (backpressure) rather than over-allocating.\n"
        f"[dim]Wired into the server via ServingEngine(paged=True); decode output is "
        f"identical to the contiguous path (tests/test_paged_batched.py).[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
