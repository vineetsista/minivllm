"""CLI: benchmark the decode paths and show the win.

By default runs both the naive (no-cache) baseline and the KV-cache path on the
same prompt and prints a side-by-side comparison with the speedup.

Usage:
    python -m scripts.benchmark
    python -m scripts.benchmark --max-new-tokens 128 --runs 5
    python -m scripts.benchmark --only cache
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from minivllm.benchmark import run_benchmark
from minivllm.generate import SamplingParams
from minivllm.loader import load_model

console = Console()


def _fmt(r) -> list[str]:
    return [
        r.label,
        f"{r.ttft_p50_s * 1000:.0f}",
        f"{r.decode_latency_p50_ms:.1f}",
        f"{r.decode_latency_p99_ms:.1f}",
        f"{r.decode_tokens_per_s:.2f}",
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument(
        "--only", choices=["naive", "cache"], default=None, help="run just one path instead of both"
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    console.print(f"[bold]Loading[/bold] {args.model} ...")
    model, cfg = load_model(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)

    params = SamplingParams(max_new_tokens=args.max_new_tokens, temperature=0.0)
    configs = []
    if args.only != "cache":
        configs.append(False)
    if args.only != "naive":
        configs.append(True)

    results = []
    for use_cache in configs:
        console.print(
            f"[bold]Benchmarking[/bold] {'KV cache' if use_cache else 'naive'} "
            f"· {args.runs} runs (+{args.warmup} warmup) · {args.max_new_tokens} new tokens ..."
        )
        results.append(
            run_benchmark(
                model,
                tok,
                cfg,
                prompt=args.prompt,
                params=params,
                n_runs=args.runs,
                warmup=args.warmup,
                use_cache=use_cache,
                model_id=args.model,
            )
        )

    table = Table(title="Phase 3 — decode path comparison")
    table.add_column("path")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("decode p50 (ms/tok)", justify="right")
    table.add_column("decode p99 (ms/tok)", justify="right")
    table.add_column("decode tok/s", justify="right")
    for r in results:
        table.add_row(*_fmt(r))
    console.print(table)

    if len(results) == 2:
        naive, cached = results
        speedup = cached.decode_tokens_per_s / naive.decode_tokens_per_s
        console.print(
            f"\n[bold green]KV-cache speedup:[/bold green] "
            f"{speedup:.1f}x decode throughput "
            f"({naive.decode_tokens_per_s:.2f} -> {cached.decode_tokens_per_s:.2f} tok/s)"
        )

    console.print("\n[dim]README rows:[/dim]")
    for r in results:
        console.print(r.as_markdown_row())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
