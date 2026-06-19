"""CLI: the Phase 4 win — paged vs contiguous KV memory.

Paging's payoff is memory, not single-stream speed, so we measure memory.
We take a serving workload (many sequences of varying lengths sharing one KV
pool, capped at max_seq_len) and compare:

  * Contiguous/static: every sequence reserves `max_seq_len` up front.
  * Paged: every sequence uses ceil(len / block_size) blocks; waste per
    sequence is at most block_size - 1 tokens, regardless of max_seq_len.

Accounting uses float16 (the planned GPU serving dtype) and the real Qwen3-0.6B
shape, read from config.json — no weights are loaded, so this runs instantly.

Usage:
    python -m scripts.paged_demo
    python -m scripts.paged_demo --max-seq-len 4096 --block-size 16 --budget-gib 8
"""

from __future__ import annotations

import argparse
import math

import torch
from rich.console import Console
from rich.table import Table

from minivllm.config import ModelConfig
from minivllm.paged_cache import kv_bytes_per_token

console = Console()

# A deterministic, mixed-length serving batch (short prompts, varied gens). The
# point is variance: static must reserve max_seq_len for even the shortest.
_LENGTH_PATTERN = [128, 256, 512, 192, 96, 320, 160, 448]


def _workload(n_seqs: int) -> list[int]:
    return [_LENGTH_PATTERN[i % len(_LENGTH_PATTERN)] for i in range(n_seqs)]


def _gib(n_bytes: float) -> float:
    return n_bytes / (1024**3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--n-seqs", type=int, default=64)
    ap.add_argument("--budget-gib", type=float, default=4.0)
    args = ap.parse_args()

    cfg = ModelConfig.from_hf(args.model, dtype=torch.float16)
    bpt = kv_bytes_per_token(cfg, torch.float16)  # bytes per token (K+V, all layers)

    lengths = _workload(args.n_seqs)
    used = sum(lengths)
    static = args.n_seqs * args.max_seq_len
    blk = args.block_size
    paged = sum(math.ceil(length / blk) * blk for length in lengths)

    console.print(
        f"[bold]{args.model}[/bold] · float16 · {cfg.num_hidden_layers} layers · "
        f"{cfg.num_key_value_heads} KV heads · head_dim {cfg.head_dim}"
    )
    console.print(
        f"KV cache per token: [bold]{bpt / 1024:.1f} KiB[/bold] "
        f"(= 2 x {cfg.num_hidden_layers} x {cfg.num_key_value_heads} x {cfg.head_dim} x 2 bytes)\n"
    )

    t = Table(
        title=f"KV memory for {args.n_seqs} sequences (max_seq_len {args.max_seq_len}, block {blk})"
    )
    t.add_column("scheme")
    t.add_column("reserved tokens", justify="right")
    t.add_column("KV memory", justify="right")
    t.add_column("utilization", justify="right")
    t.add_row(
        "contiguous (static)",
        f"{static:,}",
        f"{_gib(static * bpt):.2f} GiB",
        f"{100 * used / static:.1f}%",
    )
    t.add_row("paged", f"{paged:,}", f"{_gib(paged * bpt):.2f} GiB", f"{100 * used / paged:.1f}%")
    console.print(t)

    waste_static = (static - used) / static
    waste_paged = (paged - used) / paged
    console.print(
        f"\nWasted (reserved-but-unused) KV: static [red]{100 * waste_static:.1f}%[/red] "
        f"vs paged [green]{100 * waste_paged:.1f}%[/green]. "
        f"Paged internal fragmentation is bounded by block_size-1 = {blk - 1} tokens/seq."
    )
    console.print(
        f"Memory to serve this batch: "
        f"[bold]{_gib(static * bpt):.2f} GiB -> {_gib(paged * bpt):.2f} GiB[/bold] "
        f"({static / paged:.1f}x less)."
    )

    # Concurrent capacity under a fixed KV budget.
    budget = args.budget_gib * (1024**3)
    avg_paged = paged / args.n_seqs
    cap_static = int(budget // (args.max_seq_len * bpt))
    cap_paged = int(budget // (avg_paged * bpt))
    console.print(
        f"\nIn a [bold]{args.budget_gib:.0f} GiB[/bold] KV budget, concurrent sequences: "
        f"static [red]{cap_static}[/red] (each reserves max_seq_len) vs "
        f"paged [green]{cap_paged}[/green] (each uses ~{avg_paged:.0f} tokens) "
        f"-> [bold]{cap_paged / max(cap_static, 1):.0f}x[/bold] more.\n"
        f"[dim]This concurrency is what the Phase 5 scheduler turns into throughput.[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
