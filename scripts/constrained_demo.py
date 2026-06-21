"""CLI: constrained decoding — guaranteed-valid JSON from a 0.6B model.

Generates the same prompt twice: unconstrained (the model usually rambles or emits
broken JSON) and constrained to a JSON schema (every token masked to the grammar,
so the output always parses). The first constrained call compiles the grammar
(builds the vocab automaton, a few seconds); after that masking is near-free.

Usage:
    python -m scripts.constrained_demo
"""

from __future__ import annotations

import argparse
import json

import torch
from rich.console import Console

from minivllm.constraints import build_vocab, json_schema_to_regex, make_grammar_from_regex
from minivllm.generate import SamplingParams, generate
from minivllm.loader import load_model

console = Console()

_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "is_student": {"type": "boolean"},
    },
}


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompt", default="Describe a person as a JSON object.")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    console.print(f"[bold]Loading[/bold] {args.model} ...")
    model, _ = load_model(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(args.prompt, return_tensors="pt").input_ids

    def parses(text: str) -> bool:
        try:
            json.loads(text)
            return True
        except Exception:
            return False

    # Unconstrained.
    free = generate(
        model,
        ids,
        SamplingParams(max_new_tokens=80, temperature=0.0),
        eos_token_id=tok.eos_token_id,
        use_cache=True,
    )
    free_text = tok.decode(free.generated_token_ids, skip_special_tokens=True)

    # Constrained to the schema.
    console.print("[dim]compiling grammar (builds the vocab automaton) ...[/dim]")
    grammar = make_grammar_from_regex(
        json_schema_to_regex(_SCHEMA), *build_vocab(tok), tok.eos_token_id
    )
    params = SamplingParams(max_new_tokens=80, temperature=0.0)
    params.constraint = grammar.new_fsm()
    con = generate(model, ids, params, eos_token_id=tok.eos_token_id, use_cache=True)
    con_text = tok.decode(con.generated_token_ids, skip_special_tokens=True)

    free_preview = repr(free_text[:160])
    console.print(f"\n[bold]schema:[/bold] {json.dumps(_SCHEMA['properties'])}")
    console.print(
        f"\n[bold red]unconstrained[/bold red] (valid JSON: {parses(free_text)}):\n  {free_preview}"
    )
    console.print(
        f"\n[bold green]constrained[/bold green] (valid JSON: {parses(con_text)}):\n  {con_text!r}"
    )
    if parses(con_text):
        console.print(f"\n[dim]parsed -> {json.loads(con_text)}[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
