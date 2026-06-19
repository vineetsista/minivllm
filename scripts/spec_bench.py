"""CLI: speculative decoding vs plain greedy — the Phase 7 numbers.

Plain greedy takes one target forward per token. Speculative decoding verifies K
drafted tokens per forward and accepts the longest agreeing prefix, so it covers
the same tokens in fewer *sequential* target steps. We report the algorithmic
lever (target forwards, acceptance, tokens/forward) and wall time.

Note: decode is largely memory-bound (the weights stream from RAM/VRAM each
pass), so a K-token verify forward costs far less than K single-token forwards —
that holds even on CPU, so a high acceptance rate yields a real wall-clock win
here, amplified further on GPU. The gain is workload-dependent: repetitive text
drafts well (high acceptance), novel text less so.

Usage:
    python -m scripts.spec_bench --max-new-tokens 64 --k 4
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from minivllm.generate import SamplingParams, generate
from minivllm.loader import load_model
from minivllm.speculative import NgramDrafter, SpeculativeDecoder

console = Console()

# Repetitive/structured text is where n-gram drafting shines.
_PROMPT = (
    "List the planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune. "
    "Now list them again: Mercury, Venus, Earth,"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompt", default=_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    console.print(f"[bold]Loading[/bold] {args.model} ...")
    model, _ = load_model(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    prompt_ids = tok(args.prompt, return_tensors="pt").input_ids[0].tolist()
    params = SamplingParams(max_new_tokens=args.max_new_tokens, temperature=0.0)

    # Warm up.
    generate(model, tok(args.prompt, return_tensors="pt").input_ids,
             SamplingParams(max_new_tokens=4, temperature=0.0), eos_token_id=None, use_cache=True)

    import time
    t0 = time.perf_counter()
    base = generate(model, tok(args.prompt, return_tensors="pt").input_ids, params,
                    eos_token_id=None, use_cache=True)
    base_secs = time.perf_counter() - t0
    base_forwards = base.num_generated  # one target forward per token

    spec_tokens, stats = SpeculativeDecoder(model, drafter=NgramDrafter(), k=args.k).generate(
        prompt_ids, args.max_new_tokens
    )
    assert spec_tokens == base.generated_token_ids, "speculative output must equal greedy"

    table = Table(title=f"Phase 7 — speculative (k={args.k}) vs greedy")
    table.add_column("decoder")
    table.add_column("tokens", justify="right")
    table.add_column("target forwards", justify="right")
    table.add_column("tokens / forward", justify="right")
    table.add_column("wall (s)", justify="right")
    table.add_row("greedy", str(base_forwards), str(base_forwards), "1.00", f"{base_secs:.2f}")
    table.add_row(
        f"speculative", str(stats.generated), str(stats.target_forwards),
        f"{stats.tokens_per_forward:.2f}", f"{stats.seconds:.2f}",
    )
    console.print(table)
    wall_speedup = base_secs / stats.seconds if stats.seconds else float("nan")
    console.print(
        f"\n[bold green]Acceptance rate:[/bold green] {100 * stats.acceptance_rate:.0f}% "
        f"({stats.accepted_tokens}/{stats.draft_tokens} drafted tokens accepted)\n"
        f"[bold]Target forwards:[/bold] {base_forwards} -> {stats.target_forwards} "
        f"({base_forwards / stats.target_forwards:.1f}x fewer), "
        f"[bold]wall:[/bold] {base_secs:.2f}s -> {stats.seconds:.2f}s "
        f"({wall_speedup:.1f}x).\n"
        f"[dim]Even on CPU decode is partly memory-bound (the 0.6B weights stream from RAM "
        f"each pass), so verifying {args.k} tokens in one forward amortizes that load — hence "
        f"a real wall-clock win when acceptance is high. The gain is workload-dependent "
        f"(repetitive text drafts well) and larger on GPU.[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
