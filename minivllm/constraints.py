r"""Constrained / structured decoding — guarantee valid output by masking logits.

At every decode step the model proposes a distribution over the whole vocabulary.
If we mask out every token that would make the output violate a grammar — a JSON
schema, a regex — *before* sampling, the model can only ever emit a conforming
sequence. Even a 0.6B model then produces 100%-parseable JSON.

The machinery (the Outlines / XGrammar idea, from scratch):
  1. Compile a regex (or a JSON schema → regex) into an **NFA** (Thompson
     construction). We simulate it lazily — a "config" is the epsilon-closed set
     of NFA states, i.e. an on-the-fly DFA.
  2. Build a **vocabulary trie** once: every token's decoded string inserted as a
     path of characters, the leaf tagged with the token id.
  3. The allowed tokens for a config = walk the trie and the NFA together; a token
     is allowed iff its character path keeps the NFA in a non-empty config. This
     is bounded by token length, not vocab size, and configs are memoized.
  4. EOS is allowed only in an accepting config (so generation can't stop early
     mid-structure, and stops as soon as the structure is complete).

Scope: a regex subset sufficient for JSON (literals, `[...]` classes incl. `^`
negation and ranges, `\d \w \s`, `.`, `* + ?`, `|`, grouping) and flat JSON object
schemas (string/integer/number/boolean fields, fixed key order). Full CFGs are out
of scope (noted in docs/DESIGN.md).
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------------
# Regex -> NFA (Thompson construction) with lazy simulation.
# ---------------------------------------------------------------------------------


class _Nfa:
    """An NFA over single characters. States are ints; transitions carry a matcher
    predicate `str -> bool`. Simulated lazily (no DFA built up front)."""

    def __init__(self) -> None:
        self.eps: list[list[int]] = []
        self.trans: list[list[tuple]] = []  # list of (matcher, target)
        self.start = 0
        self.accept = 0
        # (config, char) -> next config. The trie ∩ NFA walk hits the same pairs
        # millions of times over a 151k-token vocab; memoizing is the difference
        # between a ~2-minute first request and a sub-second one.
        self._move_cache: dict[tuple[frozenset[int], str], frozenset[int]] = {}

    def _new(self) -> int:
        self.eps.append([])
        self.trans.append([])
        return len(self.eps) - 1

    def closure(self, states) -> frozenset[int]:
        stack = list(states)
        seen = set(states)
        while stack:
            s = stack.pop()
            for t in self.eps[s]:
                if t not in seen:
                    seen.add(t)
                    stack.append(t)
        return frozenset(seen)

    def move(self, config: frozenset[int], ch: str) -> frozenset[int]:
        key = (config, ch)
        cached = self._move_cache.get(key)
        if cached is not None:
            return cached
        nxt: set[int] = set()
        for s in config:
            for matcher, t in self.trans[s]:
                if matcher(ch):
                    nxt.add(t)
        result = self.closure(nxt) if nxt else frozenset()
        self._move_cache[key] = result
        return result

    def start_config(self) -> frozenset[int]:
        return self.closure([self.start])

    def is_accepting(self, config: frozenset[int]) -> bool:
        return self.accept in config


def _class_matcher(body: str):
    """Compile the inside of a [...] character class to a predicate."""
    negate = body.startswith("^")
    if negate:
        body = body[1:]
    singles: set[str] = set()
    ranges: list[tuple[str, str]] = []
    preds: list = []
    i = 0
    while i < len(body):
        c = body[i]
        if c == "\\" and i + 1 < len(body):
            singles |= _escape_set(body[i + 1])
            i += 2
            continue
        if i + 2 < len(body) and body[i + 1] == "-":
            ranges.append((c, body[i + 2]))
            i += 3
            continue
        singles.add(c)
        i += 1

    def matches(ch: str) -> bool:
        hit = ch in singles or any(lo <= ch <= hi for lo, hi in ranges) or any(p(ch) for p in preds)
        return (not hit) if negate else hit

    return matches


def _escape_set(c: str):
    if c == "d":
        return {chr(d) for d in range(48, 58)}
    if c == "w":
        return (
            {chr(x) for x in range(48, 58)}
            | {chr(x) for x in range(65, 91)}
            | {chr(x) for x in range(97, 123)}
            | {"_"}
        )
    if c == "s":
        return set(" \t\n\r")
    if c == "t":
        return {"\t"}
    if c == "n":
        return {"\n"}
    if c == "r":
        return {"\r"}
    return {c}  # escaped literal


def _escape_matcher(c: str):
    if c in "dwstnr":
        s = _escape_set(c)
        return lambda ch: ch in s
    lit = c
    return lambda ch: ch == lit


class _RegexParser:
    """Recursive-descent parser building NFA fragments (start, accept) directly."""

    def __init__(self, nfa: _Nfa, pattern: str):
        self.nfa = nfa
        self.p = pattern
        self.i = 0

    def parse(self):
        frag = self._alt()
        self.nfa.start, self.nfa.accept = frag
        return self.nfa

    def _alt(self):
        frag = self._concat()
        while self._peek() == "|":
            self.i += 1
            right = self._concat()
            s, a = self.nfa._new(), self.nfa._new()
            self.nfa.eps[s] += [frag[0], right[0]]
            self.nfa.eps[frag[1]].append(a)
            self.nfa.eps[right[1]].append(a)
            frag = (s, a)
        return frag

    def _concat(self):
        frags = []
        while self._peek() not in (None, "|", ")"):
            frags.append(self._quant())
        if not frags:
            s = self.nfa._new()
            return (s, s)
        start = frags[0][0]
        for x, y in zip(frags, frags[1:], strict=False):
            self.nfa.eps[x[1]].append(y[0])
        return (start, frags[-1][1])

    def _quant(self):
        frag = self._atom()
        q = self._peek()
        if q == "*":
            self.i += 1
            s, a = self.nfa._new(), self.nfa._new()
            self.nfa.eps[s] += [frag[0], a]
            self.nfa.eps[frag[1]] += [frag[0], a]
            return (s, a)
        if q == "+":
            self.i += 1
            s, a = self.nfa._new(), self.nfa._new()
            self.nfa.eps[s].append(frag[0])
            self.nfa.eps[frag[1]] += [frag[0], a]
            return (s, a)
        if q == "?":
            self.i += 1
            s, a = self.nfa._new(), self.nfa._new()
            self.nfa.eps[s] += [frag[0], a]
            self.nfa.eps[frag[1]].append(a)
            return (s, a)
        return frag

    def _atom(self):
        c = self._peek()
        if c == "(":
            self.i += 1
            frag = self._alt()
            assert self._peek() == ")", "unbalanced ("
            self.i += 1
            return frag
        if c == "[":
            j = self.p.index("]", self.i + 1)
            while self.p[j - 1] == "\\":
                j = self.p.index("]", j + 1)
            body = self.p[self.i + 1 : j]
            self.i = j + 1
            return self._char(_class_matcher(body))
        if c == "\\":
            self.i += 2
            return self._char(_escape_matcher(self.p[self.i - 1]))
        if c == ".":
            self.i += 1
            return self._char(lambda ch: True)
        self.i += 1
        return self._char(lambda ch, lit=c: ch == lit)

    def _char(self, matcher):
        s, a = self.nfa._new(), self.nfa._new()
        self.nfa.trans[s].append((matcher, a))
        return (s, a)

    def _peek(self):
        return self.p[self.i] if self.i < len(self.p) else None


def compile_regex_nfa(pattern: str) -> _Nfa:
    return _RegexParser(_Nfa(), pattern).parse()


# ---------------------------------------------------------------------------------
# JSON schema -> regex.
# ---------------------------------------------------------------------------------

_WS = r"[ \t\n\r]*"


def _rep(atom: str, lo: int, hi: int) -> str:
    """Regex matching lo..hi repetitions of `atom` (a single regex unit), built
    without `{n,m}` (which the engine doesn't parse). Bounding the count is what
    stops a greedy model emitting digits/characters forever and never closing."""
    opt = ""
    for _ in range(hi - lo):
        opt = "(" + atom + opt + ")?"
    return atom * lo + opt


_VALUE = {
    "string": '"' + _rep(r'[^"]', 0, 48) + '"',
    "integer": "-?" + _rep(r"[0-9]", 1, 12),
    "number": "-?" + _rep(r"[0-9]", 1, 12) + r"(\." + _rep(r"[0-9]", 1, 8) + ")?",
    "boolean": r"(true|false)",
}


def json_schema_to_regex(schema: dict) -> str:
    """Flat object schema -> regex. Fields are emitted in `properties` order."""
    props = schema.get("properties", {})
    parts = [r"\{", _WS]
    for idx, (key, spec) in enumerate(props.items()):
        if idx:
            parts += [_WS, ",", _WS]
        val = _VALUE.get(spec.get("type", "string"), _VALUE["string"])
        parts += [r'"' + key + r'"', _WS, ":", _WS, val]
    parts += [_WS, r"\}"]
    return "".join(parts)


def generic_object_regex() -> str:
    """A flat (non-recursive) JSON object: string keys, primitive values. Used for
    OpenAI `response_format={"type":"json_object"}` when no schema is given."""
    key = '"' + _rep(r'[^"]', 1, 32) + '"'
    val = "(" + _VALUE["string"] + "|" + _VALUE["number"] + "|" + _VALUE["boolean"] + ")"
    pair = key + _WS + ":" + _WS + val
    return r"\{" + _WS + "(" + pair + "(" + _WS + "," + _WS + pair + ")*)?" + _WS + r"\}"


# ---------------------------------------------------------------------------------
# Vocabulary trie + the per-request FSM that masks logits.
# ---------------------------------------------------------------------------------


class _TrieNode:
    __slots__ = ("children", "token_id")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.token_id: int | None = None


def build_vocab(tokenizer) -> tuple[_TrieNode, dict[int, str]]:
    """Build the char trie of decoded token strings + an id->string map. Done once
    per tokenizer (cache it). Special tokens are excluded (EOS is handled separately)."""
    root = _TrieNode()
    id_to_str: dict[int, str] = {}
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    vocab_size = len(tokenizer)
    for tid in range(vocab_size):
        if tid in special:
            continue
        s = tokenizer.decode([tid], skip_special_tokens=True)
        if not s:
            continue
        id_to_str[tid] = s
        node = root
        for ch in s:
            node = node.children.setdefault(ch, _TrieNode())
        node.token_id = tid
    return root, id_to_str


class Grammar:
    """A compiled grammar (immutable, shared across requests): the NFA + the shared
    vocab trie + a memoized config->allowed-token-ids cache."""

    def __init__(self, nfa: _Nfa, trie: _TrieNode, id_to_str: dict[int, str], eos_id: int):
        self.nfa = nfa
        self.trie = trie
        self.id_to_str = id_to_str
        self.eos_id = eos_id
        self._cache: dict[frozenset[int], torch.Tensor] = {}

    def allowed_idx(self, config: frozenset[int]) -> torch.Tensor:
        cached = self._cache.get(config)
        if cached is not None:
            return cached
        allowed: set[int] = set()
        stack = [(self.trie, config)]
        while stack:
            node, cfg = stack.pop()
            if node.token_id is not None:
                allowed.add(node.token_id)
            for ch, child in node.children.items():
                ncfg = self.nfa.move(cfg, ch)
                if ncfg:
                    stack.append((child, ncfg))
        if self.nfa.is_accepting(config):
            allowed.add(self.eos_id)
        if not allowed:  # dead end — let it stop rather than mask everything to -inf
            allowed.add(self.eos_id)
        idx = torch.tensor(sorted(allowed), dtype=torch.long)
        self._cache[config] = idx
        return idx

    def new_fsm(self) -> ConstraintFSM:
        return ConstraintFSM(self)


class ConstraintFSM:
    """Per-request decoding state: the current NFA config, advanced as tokens are
    chosen. Shares the heavy Grammar (NFA + trie + cache)."""

    def __init__(self, grammar: Grammar):
        self.grammar = grammar
        self.config = grammar.nfa.start_config()

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        idx = self.grammar.allowed_idx(self.config).to(logits.device)
        masked = torch.full_like(logits, float("-inf"))
        masked[idx] = logits[idx]
        return masked

    def advance(self, token_id: int) -> None:
        if token_id == self.grammar.eos_id:
            return
        s = self.grammar.id_to_str.get(token_id, "")
        for ch in s:
            self.config = self.grammar.nfa.move(self.config, ch)

    def is_accepting(self) -> bool:
        return self.grammar.nfa.is_accepting(self.config)


# ---------------------------------------------------------------------------------
# Builders used by the serving layer.
# ---------------------------------------------------------------------------------


def make_grammar_from_regex(pattern: str, trie, id_to_str, eos_id: int) -> Grammar:
    return Grammar(compile_regex_nfa(pattern), trie, id_to_str, eos_id)


def make_grammar_from_schema(schema: dict, trie, id_to_str, eos_id: int) -> Grammar:
    return make_grammar_from_regex(json_schema_to_regex(schema), trie, id_to_str, eos_id)
