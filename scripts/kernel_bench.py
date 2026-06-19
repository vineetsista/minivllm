"""CLI: benchmark the fused Triton RMSNorm against the PyTorch reference.

On CUDA this runs the Triton kernel vs the unfused reference and reports the
speedup (the Phase 6 number). On CPU there is no Triton, so both paths are the
reference and it simply confirms correctness and prints ~1x — run it on a free
Colab/Kaggle T4 (see notebooks/minivllm_colab.ipynb) to get the real kernel win.

Usage:
    python -m scripts.kernel_bench
    python -m scripts.kernel_bench --rows 8192 --dim 1024
"""

from __future__ import annotations

import argparse
import time

import torch

from minivllm.kernels import _ref_rms_norm, rms_norm, triton_available


def _bench(fn, x, w, eps, iters: int) -> float:
    for _ in range(10):  # warmup
        fn(x, w, eps)
    if x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x, w, eps)
    if x.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # microseconds / call


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    x = torch.randn(args.rows, args.dim, device=device)
    w = torch.randn(args.dim, device=device)
    eps = 1e-6

    # Correctness first.
    out = rms_norm(x, w, eps)
    ref = _ref_rms_norm(x, w, eps)
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)

    fused_us = _bench(rms_norm, x, w, eps, args.iters)
    ref_us = _bench(_ref_rms_norm, x, w, eps, args.iters)

    print(f"device            : {device}")
    print(f"triton active     : {triton_available()}")
    print(f"shape             : [{args.rows}, {args.dim}]")
    print(f"correct vs ref    : {ok}")
    print(f"reference (unfused): {ref_us:8.2f} us/call")
    print(f"rms_norm (dispatch): {fused_us:8.2f} us/call")
    if triton_available():
        print(f"kernel speedup    : {ref_us / fused_us:.2f}x")
    else:
        print("kernel speedup    : n/a (no CUDA/Triton here — both paths are the reference)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
