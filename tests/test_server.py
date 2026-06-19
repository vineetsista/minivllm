"""Phase 8 correctness: the streaming serving engine schedules concurrent
requests without changing their output.

Fast test only (no model download, no HTTP): a tiny random-weight model behind
the threaded `ServingEngine`. Several requests of different lengths are submitted
concurrently from multiple threads; each result must equal what single-sequence
greedy decode produces. This exercises the background worker, dynamic admission
into free slots, and per-request completion signalling.
"""

from __future__ import annotations

import threading

import torch

from minivllm.config import ModelConfig
from minivllm.generate import SamplingParams, generate
from minivllm.model import Qwen3ForCausalLM
from minivllm.server import ServingEngine


def _tiny_model() -> Qwen3ForCausalLM:
    cfg = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        tie_word_embeddings=True,
        num_hidden_layers=2,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).eval()


def _greedy(model, prompt, n):
    return generate(
        model,
        torch.tensor([prompt]),
        SamplingParams(max_new_tokens=n, temperature=0.0),
        eos_token_id=None,
        use_cache=True,
    ).generated_token_ids


def test_serving_engine_concurrent_matches_greedy():
    model = _tiny_model()
    engine = ServingEngine(model, max_slots=2, max_seq_len=128, eos_token_id=None)
    try:
        specs = [([1, 2, 3], 5), ([4, 5], 12), ([6, 7, 8, 9], 7), ([2, 4], 9), ([1, 1, 1], 6)]
        expected = [_greedy(model, p, n) for p, n in specs]

        results: dict[int, list[int]] = {}
        reqs = []

        def submit_one(idx, prompt, n):
            req = engine.submit(prompt, n, SamplingParams(max_new_tokens=n, temperature=0.0))
            reqs.append((idx, req))

        # Submit concurrently from several threads to stress the queue/worker.
        threads = [
            threading.Thread(target=submit_one, args=(i, p, n)) for i, (p, n) in enumerate(specs)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for idx, req in reqs:
            assert req.event.wait(timeout=30.0), f"request {idx} timed out"
            assert req.error is None, f"request {idx} errored: {req.error}"
            results[idx] = req.result

        for i in range(len(specs)):
            assert results[i] == expected[i], (
                f"request {i} diverged:\n single={expected[i]}\n serve ={results[i]}"
            )
        assert engine.completed == len(specs)
    finally:
        engine.shutdown()
