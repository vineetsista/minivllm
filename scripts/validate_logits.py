"""CLI: prove our forward pass matches the HuggingFace reference.

Usage:
    python -m scripts.validate_logits
    python -m scripts.validate_logits --model Qwen/Qwen3-0.6B --prompt "Hello, world"
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from minivllm.validation import compare_to_reference

console = Console()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    console.print(f"[bold]Loading + comparing[/bold] {args.model} ...")
    r = compare_to_reference(args.model, prompt=args.prompt, topk=args.topk)

    table = Table(title="Logit parity vs HuggingFace reference")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("prompt", repr(r.prompt))
    table.add_row("seq len", str(r.seq_len))
    table.add_row("max abs logit diff", f"{r.max_abs_diff:.3e}")
    table.add_row("mean abs logit diff", f"{r.mean_abs_diff:.3e}")
    table.add_row("argmax match (all positions)", f"{r.argmax_match_frac * 100:.2f}%")
    table.add_row(f"top-{r.topk} final-token set match", "yes" if r.topk_match else "no")
    table.add_row("our next token id", str(r.ours_next_token))
    table.add_row("ref next token id", str(r.ref_next_token))
    console.print(table)

    if r.passed:
        console.print("[bold green]PASS[/bold green] — logits match token-for-token.")
        return 0
    console.print("[bold red]FAIL[/bold red] — divergence from reference.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
