"""Phase B: constrained decoding — the FSM must permit exactly the valid tokens.

Fast tests use a tiny synthetic vocabulary (token id -> string), so the regex
engine, the JSON-schema compiler, and the vocab-trie ∩ NFA masking are all tested
with no model download. A slow gate (real model) lives in test_constraints_real
(see the *_real_model name -> auto-skipped unless --runslow).
"""

from __future__ import annotations

import json

import torch

from minivllm.constraints import (
    Grammar,
    _TrieNode,
    compile_regex_nfa,
    json_schema_to_regex,
    make_grammar_from_regex,
)


def _vocab(strings: list[str]):
    """Build a trie + id->str map from a synthetic vocabulary. id 0 = EOS."""
    root = _TrieNode()
    id_to_str = {}
    for tid, s in enumerate(strings):
        if tid == 0:
            continue  # reserve 0 as EOS
        id_to_str[tid] = s
        node = root
        for ch in s:
            node = node.children.setdefault(ch, _TrieNode())
        node.token_id = tid
    return root, id_to_str


# --- regex engine ----------------------------------------------------------------


def _matches(pattern: str, s: str) -> bool:
    nfa = compile_regex_nfa(pattern)
    cfg = nfa.start_config()
    for ch in s:
        cfg = nfa.move(cfg, ch)
        if not cfg:
            return False
    return nfa.is_accepting(cfg)


def test_regex_engine_basics():
    assert _matches(r"-?[0-9]+", "42")
    assert _matches(r"-?[0-9]+", "-7")
    assert not _matches(r"-?[0-9]+", "4a")
    assert _matches(r"(true|false)", "true")
    assert not _matches(r"(true|false)", "tru")
    assert _matches(r'"[^"]*"', '"hello world"')
    assert _matches(r"-?[0-9]+(\.[0-9]+)?", "3.14")


def test_json_schema_to_regex_matches_valid_json():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    }
    rx = json_schema_to_regex(schema)
    assert _matches(rx, '{"name": "Ada", "age": 36}')
    assert _matches(rx, '{"name":"Ada","age":36}')
    assert not _matches(rx, '{"name": "Ada"}')  # missing required field
    assert not _matches(rx, '{"age": 36, "name": "Ada"}')  # wrong key order


# --- FSM masking over a synthetic vocab -----------------------------------------


def test_fsm_allows_only_valid_tokens():
    # Vocab with a mix of valid and invalid continuations.
    strings = ["<eos>", "{", "}", '"', "a", "b", "5", "xyz", "true"]
    trie, id_to_str = _vocab(strings)
    g = make_grammar_from_regex(r'\{"[a-z]+":[0-9]+\}', trie, id_to_str, eos_id=0)
    fsm = g.new_fsm()

    # Start: only "{" is valid.
    allowed = set(
        fsm.mask_logits(torch.zeros(len(strings))).isfinite().nonzero().flatten().tolist()
    )
    assert allowed == {strings.index("{")}

    # Drive a full valid string and confirm it never dead-ends, ends accepting.
    for ch_token in ["{", '"', "a", "b", '"', ":", "5", "}"]:
        if ch_token == ":":  # ':' isn't its own vocab token here; advance via a 1-char id
            tid = max(id_to_str) + 1
            id_to_str[tid] = ":"
        else:
            tid = strings.index(ch_token)
        fsm.advance(tid if ch_token != ":" else tid)
    assert fsm.is_accepting()


def test_grammar_caches_allowed_sets():
    strings = ["<eos>", "{", "}", '"', "k", ":", "1"]
    trie, id_to_str = _vocab(strings)
    g: Grammar = make_grammar_from_regex(r'\{"k":[0-9]\}', trie, id_to_str, eos_id=0)
    fsm = g.new_fsm()
    fsm.mask_logits(torch.zeros(len(strings)))
    assert g._cache  # the start config's allowed set was memoized


def test_masked_greedy_builds_valid_json_with_synthetic_lm():
    # A deterministic "model": always prefers an invalid token; the mask must force
    # validity anyway. Tokens are single characters so any path is reachable.
    chars = ["<eos>"] + list('{}":,abcdefghijklmnopqrstuvwxyz0123456789 ')
    trie, id_to_str = _vocab(chars)
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    g = make_grammar_from_regex(json_schema_to_regex(schema), trie, id_to_str, eos_id=0)
    fsm = g.new_fsm()

    out = ""
    for _ in range(40):
        logits = torch.zeros(len(chars))  # uniform -> argmax picks first allowed id
        masked = fsm.mask_logits(logits)
        tid = int(masked.argmax())
        if tid == 0:  # EOS
            break
        out += id_to_str[tid]
        fsm.advance(tid)
    # The forced output parses and has the required field.
    obj = json.loads(out)
    assert "n" in obj and isinstance(obj["n"], int)


# --- slow gate: real model is forced into valid JSON ----------------------------


def test_constrained_decoding_real_model():
    import torch
    from transformers import AutoTokenizer

    from minivllm.constraints import build_vocab, json_schema_to_regex, make_grammar_from_regex
    from minivllm.generate import SamplingParams, generate
    from minivllm.loader import load_model

    model, _ = load_model("Qwen/Qwen3-0.6B")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    trie, id_to_str = build_vocab(tok)
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}, "count": {"type": "integer"}},
    }
    grammar = make_grammar_from_regex(
        json_schema_to_regex(schema), trie, id_to_str, tok.eos_token_id
    )

    prompt = tok("Respond with a JSON object: ", return_tensors="pt").input_ids
    params = SamplingParams(max_new_tokens=48, temperature=0.0)
    params.constraint = grammar.new_fsm()
    out = generate(model, prompt, params, eos_token_id=tok.eos_token_id, use_cache=True)
    text = tok.decode(out.generated_token_ids, skip_special_tokens=True)

    obj = json.loads(text)  # the load-bearing assertion: it parses
    assert set(obj) == {"ok", "count"}
    assert isinstance(obj["ok"], bool) and isinstance(obj["count"], int)
