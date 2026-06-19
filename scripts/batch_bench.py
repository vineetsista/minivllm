"""CLI: continuous vs static batching throughput — the Phase 5 win.

Same model, same batched-decode compute, same requests; the only difference is
the admission policy. The workload mixes short and long generations on purpose:
static batching wastes a slot whenever a short sequence shares a group with a
long one, while continuous batching refills that slot immediately.

Usage:
    python -m scripts.batch_bench
    python -m scripts.batch_bench --slots 4 --short 8 --long 48
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from minivllm.engine import ContinuousBatchingEngine, Request
from minivllm.generate import SamplingParams
from minivllm.loader import load_model

console = Console()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--n-requests", type=int, default=12)
    ap.add_argument("--short", type=int, default=8)
    ap.add_argument("--long", type=int, default=48)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    console.print(f"[bold]Loading[/bold] {args.model} ...")
    model, _ = load_model(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    prompt_ids = tok(args.prompt, return_tensors="pt").input_ids[0].tolist()

    # High-variance lengths: one long sequence per group of `slots`, rest short.
    lengths = [args.long if i % args.slots == 0 else args.short for i in range(args.n_requests)]
    reqs = [
        Request(id=i, prompt_ids=list(prompt_ids), max_new_tokens=n) for i, n in enumerate(lengths)
    ]
    console.print(
        f"{args.n_requests} requests · {args.slots} slots · "
        f"gen lengths {lengths} (mean {sum(lengths) / len(lengths):.0f})\n"
    )

    engine = ContinuousBatchingEngine(model, max_slots=args.slots, eos_token_id=None)
    params = SamplingParams(temperature=0.0)

    # Warm the threadpool/allocator so the first policy isn't penalized.
    engine.run(reqs[: args.slots], params, policy="continuous")

    results = {}
    for policy in ("static", "continuous"):
        console.print(f"[bold]Running[/bold] {policy} ...")
        _, stats = engine.run(reqs, params, policy=policy)
        results[policy] = stats

    table = Table(title="Phase 5 — static vs continuous batching")
    table.add_column("policy")
    table.add_column("wall (s)", justify="right")
    table.add_column("decode steps", justify="right")
    table.add_column("avg batch occupancy", justify="right")
    table.add_column("throughput (tok/s)", justify="right")
    for policy in ("static", "continuous"):
        s = results[policy]
        table.add_row(
            policy,
            f"{s.seconds:.2f}",
            str(s.decode_steps),
            f"{s.avg_batch_occupancy:.2f} / {args.slots}",
            f"{s.tokens_per_s:.2f}",
        )
    console.print(table)

    st, ct = results["static"], results["continuous"]
    console.print(
        f"\n[bold green]Continuous batching:[/bold green] "
        f"{ct.tokens_per_s / st.tokens_per_s:.1f}x throughput, "
        f"{st.decode_steps / ct.decode_steps:.1f}x fewer decode steps "
        f"({st.decode_steps} -> {ct.decode_steps}).\n"
        f"[dim]Occupancy {st.avg_batch_occupancy:.2f} -> {ct.avg_batch_occupancy:.2f} of "
        f"{args.slots} slots: static leaves slots idle while a long sequence drains; "
        f"continuous keeps them full.[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
