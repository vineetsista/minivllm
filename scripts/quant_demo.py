"""CLI: int8 weight-only quantization — memory saved vs quality cost.

Loads the real model in float32, measures its memory and next-token logits on a
prompt, then quantizes the Linear weights to int8 and measures both again. Unlike
the rest of the engine this is lossy, so we report the fidelity (logit cosine,
argmax agreement) alongside the memory saving, and show generation still reads
sensibly.

Usage:
    python -m scripts.quant_demo --prompt "The capital of France is"
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

from minivllm.generate import SamplingParams, generate
from minivllm.loader import load_model
from minivllm.quantization import model_nbytes, quantize_linears

console = Console()


def _gib(n: int) -> float:
    return n / (1024**3)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    console.print(f"[bold]Loading[/bold] {args.model} (float32) ...")
    model, _ = load_model(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(args.prompt, return_tensors="pt").input_ids

    fp32_bytes = model_nbytes(model)
    fp32_logits = model(ids)[0, -1]
    fp32_gen = generate(
        model,
        ids,
        SamplingParams(max_new_tokens=args.max_new_tokens, temperature=0.0),
        eos_token_id=tok.eos_token_id,
        use_cache=True,
    ).generated_token_ids

    console.print("[bold]Quantizing[/bold] Linear weights to int8 ...")
    quantize_linears(model)
    int8_bytes = model_nbytes(model)
    int8_logits = model(ids)[0, -1]
    int8_gen = generate(
        model,
        ids,
        SamplingParams(max_new_tokens=args.max_new_tokens, temperature=0.0),
        eos_token_id=tok.eos_token_id,
        use_cache=True,
    ).generated_token_ids

    cos = F.cosine_similarity(fp32_logits, int8_logits, dim=0).item()
    max_abs = (fp32_logits - int8_logits).abs().max().item()
    same_argmax = int(fp32_logits.argmax() == int8_logits.argmax())

    t = Table(title="int8 weight-only quantization")
    t.add_column("metric")
    t.add_column("value", justify="right")
    t.add_row("model memory (fp32)", f"{_gib(fp32_bytes):.3f} GiB")
    t.add_row("model memory (int8 weights)", f"{_gib(int8_bytes):.3f} GiB")
    t.add_row("reduction", f"{fp32_bytes / int8_bytes:.2f}x")
    t.add_section()
    t.add_row("next-token logit cosine", f"{cos:.5f}")
    t.add_row("max abs logit diff", f"{max_abs:.3e}")
    t.add_row("same next token", "yes" if same_argmax else "no")
    console.print(t)

    console.print(f"\n[dim]fp32:[/dim] {tok.decode(fp32_gen, skip_special_tokens=True)!r}")
    console.print(f"[dim]int8:[/dim] {tok.decode(int8_gen, skip_special_tokens=True)!r}")
    console.print(
        "\n[dim]int8 stores weights at 1 byte vs 4; embeddings stay fp32 (tied to "
        "lm_head), so the whole-model reduction is below 4x. On GPU this also speeds "
        "memory-bound decode.[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
