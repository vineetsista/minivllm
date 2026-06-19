"""CLI: run the Phase 2 baseline benchmark and print the numbers.

Usage:
    python -m scripts.benchmark
    python -m scripts.benchmark --max-new-tokens 128 --runs 5
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from minivllm.benchmark import run_benchmark
from minivllm.generate import SamplingParams
from minivllm.loader import load_model

console = Console()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    console.print(f"[bold]Loading[/bold] {args.model} ...")
    model, cfg = load_model(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)

    console.print(
        f"[bold]Benchmarking[/bold] naive decode · {args.runs} runs "
        f"(+{args.warmup} warmup) · {args.max_new_tokens} new tokens ..."
    )
    result = run_benchmark(
        model,
        tok,
        cfg,
        prompt=args.prompt,
        params=SamplingParams(max_new_tokens=args.max_new_tokens, temperature=0.0),
        n_runs=args.runs,
        warmup=args.warmup,
        model_id=args.model,
    )

    table = Table(title="Phase 2 baseline — naive autoregressive decode")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("device / dtype", f"{result.device} / {result.dtype}")
    table.add_row("torch threads", str(result.torch_threads))
    table.add_row("prompt tokens", str(result.prompt_tokens))
    table.add_row("new tokens / run", str(result.max_new_tokens))
    table.add_row("runs", str(result.n_runs))
    table.add_section()
    table.add_row("TTFT p50", f"{result.ttft_p50_s * 1000:.0f} ms")
    table.add_row("TTFT p99", f"{result.ttft_p99_s * 1000:.0f} ms")
    table.add_row("decode latency p50", f"{result.decode_latency_p50_ms:.1f} ms/tok")
    table.add_row("decode latency p99", f"{result.decode_latency_p99_ms:.1f} ms/tok")
    table.add_section()
    table.add_row("decode throughput", f"{result.decode_tokens_per_s:.2f} tok/s")
    table.add_row("end-to-end throughput", f"{result.end_to_end_tokens_per_s:.2f} tok/s")
    console.print(table)

    console.print("\n[dim]README row:[/dim]")
    console.print(result.as_markdown_row())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
