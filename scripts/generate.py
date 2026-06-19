"""CLI: generate text with the naive (Phase 2) decode loop.

Usage:
    python -m scripts.generate --prompt "The capital of France is"
    python -m scripts.generate --prompt "Write a haiku" --temperature 0.8 --top-p 0.95 --seed 0
"""

from __future__ import annotations

import argparse

from rich.console import Console

from minivllm.generate import SamplingParams, generate
from minivllm.loader import load_model

console = Console()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    console.print(f"[bold]Loading[/bold] {args.model} ...")
    model, _ = load_model(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)

    input_ids = tok(args.prompt, return_tensors="pt").input_ids
    params = SamplingParams(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )

    out = generate(model, input_ids, params, eos_token_id=tok.eos_token_id)
    text = tok.decode(out.generated_token_ids, skip_special_tokens=True)

    console.print(f"\n[dim]{args.prompt}[/dim][bold]{text}[/bold]")
    console.print(
        f"\n[dim]{out.num_generated} tokens · TTFT {out.prefill_seconds * 1000:.0f} ms · "
        f"{out.num_generated / out.total_seconds:.2f} tok/s end-to-end"
        f"{' · hit EOS' if out.stopped_on_eos else ''}[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
