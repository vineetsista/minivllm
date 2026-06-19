"""Run the whole benchmark suite — regenerates every README number in one shot.

Each phase benchmark loads the model in its own subprocess for isolation. The
serving load test (Phase 8) needs a running server, so it is listed but not
launched here; run it separately per the README.

Usage:
    python -m scripts.bench_all
    python -m scripts.bench_all --max-new-tokens 32   # quicker pass
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from rich.console import Console

console = Console()

# (label, module, extra args)
_STEPS = [
    ("Phase 1 — logit parity", "scripts.validate_logits", []),
    ("Phase 2/3 — naive vs KV cache", "scripts.benchmark", []),
    ("Phase 4 — paged KV memory", "scripts.paged_demo", []),
    ("Phase 5 — static vs continuous batching", "scripts.batch_bench", []),
    ("Phase 7 — speculative vs greedy", "scripts.spec_bench", []),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    for label, module, extra in _STEPS:
        console.rule(f"[bold]{label}[/bold]")
        cmd = [sys.executable, "-m", module, *extra]
        # Pass --max-new-tokens only to scripts that accept it.
        if module in {"scripts.benchmark", "scripts.spec_bench"}:
            cmd += ["--max-new-tokens", str(args.max_new_tokens)]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            console.print(f"[red]{label} failed (exit {result.returncode})[/red]")
            return result.returncode

    console.rule("[bold]Phase 8 — serving load test[/bold]")
    console.print(
        "Start the server, then load-test it:\n"
        "  python -m uvicorn minivllm.server:app --port 8000\n"
        "  python -m scripts.loadtest --n 8 --concurrency 4"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
