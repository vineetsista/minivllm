"""Custom kernels — Phase 6.

A fused RMSNorm written in Triton. On a GPU, RMSNorm's reduction + normalize +
scale is memory-bound and benefits from fusing the whole row into one kernel
launch (one read, one write) instead of several PyTorch ops each round-tripping
through global memory. `rms_norm` dispatches to the Triton kernel on CUDA and to
a numerically identical PyTorch reference everywhere else, so the engine runs
unchanged on CPU and the kernel activates automatically on GPU.

This is the seam `layers.py` always pointed at: RMSNorm now calls `rms_norm`, so
swapping in the kernel touched nothing else. A CUDA + Triton box validates the
kernel against the reference (tests/test_kernels.py, skipped without CUDA).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # CPU-only dev box: no triton
    _HAS_TRITON = False


def _ref_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """PyTorch reference — must match minivllm.layers.RMSNorm bit-for-bit:
    normalize in float32, cast back, then apply the learned weight."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return weight * x.to(input_dtype)


if _HAS_TRITON:

    @triton.jit
    def _rms_norm_fwd(X, W, Y, stride, N, eps, BLOCK: tl.constexpr):
        # One program per row; BLOCK is the next power of two >= N, so a row is
        # processed in a single vectorized pass.
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(Y + row * stride + cols, y, mask=mask)

    def _triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        n = x.shape[-1]
        x2 = x.reshape(-1, n).contiguous()
        y = torch.empty_like(x2, dtype=torch.float32)
        block = triton.next_power_of_2(n)
        _rms_norm_fwd[(x2.shape[0],)](x2, weight, y, x2.stride(0), n, eps, BLOCK=block, num_warps=4)
        return y.reshape(x.shape).to(x.dtype)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Fused RMSNorm on CUDA (Triton), reference everywhere else."""
    if _HAS_TRITON and x.is_cuda:
        return _triton_rms_norm(x, weight, eps)
    return _ref_rms_norm(x, weight, eps)


def triton_available() -> bool:
    return _HAS_TRITON and torch.cuda.is_available()
