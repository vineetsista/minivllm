"""Shared pytest configuration.

The suite has two tiers:
  * fast  — pure logic and tiny in-memory random-weight models; no network, runs
            in seconds. This is what CI and a bare `pytest` run execute.
  * slow  — correctness gates that download Qwen3-0.6B and call the HuggingFace
            reference. These are the ground-truth checks, but they need the
            network and ~minutes, so they are skipped unless explicitly enabled.

Slow tests are detected by name (they are named after what they compare against
the reference) and auto-marked, so individual test files need no decorators.
Enable them with `pytest --runslow` or `RUN_SLOW=1 pytest`.
"""

from __future__ import annotations

import os

import pytest

# Substrings that identify the model-downloading / HF-reference gates. Kept
# precise so tiny-model tests (which may share words like "single_sequence")
# are not caught: the real gates all compare against the HF "_reference", use
# the "real_model", or run "both_policies" of the real-model engine.
_SLOW_NAME_HINTS = ("_reference", "real_model", "both_policies")


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests that download Qwen3-0.6B and call the HF reference",
    )


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--runslow") or os.environ.get("RUN_SLOW") == "1"
    skip_slow = pytest.mark.skip(reason="slow/network gate; pass --runslow or set RUN_SLOW=1")
    for item in items:
        if any(hint in item.name for hint in _SLOW_NAME_HINTS):
            item.add_marker(pytest.mark.slow)
            if not run_slow:
                item.add_marker(skip_slow)
