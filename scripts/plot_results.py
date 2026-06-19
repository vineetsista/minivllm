"""Render the optimization-journey charts into docs/assets/.

The numbers are the measured CPU results from the per-phase benchmark scripts
(scripts/benchmark.py, batch_bench.py, paged_demo.py, spec_bench.py, loadtest.py),
recorded here so the figures regenerate deterministically without re-running every
heavy benchmark. Re-run a given benchmark to refresh its row, then this script.

Usage:
    pip install matplotlib
    python -m scripts.plot_results
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

BG = "#0b0e14"
FG = "#e6e9ef"
ACCENT = "#5cc8ff"
ACCENT2 = "#9d8cff"
GOOD = "#5ad19a"
WARN = "#ff7b72"


def _style(ax, title):
    ax.set_title(title, color=FG, fontsize=13, pad=12, weight="bold")
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color("#2a3346")
    ax.yaxis.label.set_color(FG)
    ax.xaxis.label.set_color(FG)


def _fig():
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    fig.patch.set_facecolor(BG)
    return fig, ax


def _bars(ax, labels, values, colors, fmt="{:.0f}"):
    bars = ax.bar(labels, values, color=colors, width=0.62)
    for b, v in zip(bars, values, strict=False):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v,
            fmt.format(v),
            ha="center",
            va="bottom",
            color=FG,
            fontsize=10,
        )


def kv_cache():
    fig, ax = _fig()
    _style(ax, "Phase 3 — KV cache flattens decode latency (CPU)")
    labels = ["naive p50", "naive p99", "cache p50", "cache p99"]
    vals = [1369.9, 4727.2, 168.0, 251.8]
    _bars(ax, labels, vals, [WARN, WARN, GOOD, GOOD], "{:.0f}")
    ax.set_ylabel("ms / token")
    fig.tight_layout()
    fig.savefig(OUT / "kv_cache_latency.png", dpi=130, facecolor=BG)
    plt.close(fig)


def paged_memory():
    fig, ax = _fig()
    _style(ax, "Phase 4 — Paged KV: serve far more in the same memory")
    labels = ["contiguous\n(static)", "paged\npool"]
    vals = [14.0, 1.8]
    _bars(ax, labels, vals, [WARN, GOOD], "{:.1f} GiB")
    ax.set_ylabel("KV memory (GiB), 64 seqs")
    fig.tight_layout()
    fig.savefig(OUT / "paged_memory.png", dpi=130, facecolor=BG)
    plt.close(fig)


def batching():
    fig, ax = _fig()
    _style(ax, "Phase 5 — Continuous vs static batching (CPU)")
    labels = ["static", "continuous"]
    vals = [2.18, 7.64]
    _bars(ax, labels, vals, [WARN, ACCENT], "{:.2f}")
    ax.set_ylabel("throughput (tok/s)")
    fig.tight_layout()
    fig.savefig(OUT / "batching_throughput.png", dpi=130, facecolor=BG)
    plt.close(fig)


def serving_scaling():
    fig, ax = _fig()
    _style(ax, "Phase 8 — Serving throughput scales with concurrency (CPU)")
    labels = ["1 (serial)", "4 (batched)"]
    vals = [2.12, 11.55]
    _bars(ax, labels, vals, [WARN, ACCENT], "{:.2f}")
    ax.set_ylabel("throughput (tok/s)")
    fig.tight_layout()
    fig.savefig(OUT / "serving_scaling.png", dpi=130, facecolor=BG)
    plt.close(fig)


def journey():
    fig, ax = _fig()
    _style(ax, "The optimization journey — relative speedup vs previous stage")
    labels = ["KV cache\n(P3)", "Cont. batch\n(P5)", "Spec. decode\n(P7)", "Serving@4\n(P8)"]
    vals = [9.0, 3.5, 2.8, 5.4]
    _bars(ax, labels, vals, [ACCENT, ACCENT2, GOOD, ACCENT], "{:.1f}x")
    ax.set_ylabel("speedup (x)")
    ax.axhline(1.0, color="#2a3346", lw=1, ls="--")
    fig.tight_layout()
    fig.savefig(OUT / "journey.png", dpi=130, facecolor=BG)
    plt.close(fig)


def main() -> int:
    for fn in (kv_cache, paged_memory, batching, serving_scaling, journey):
        fn()
    print(f"wrote charts to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
