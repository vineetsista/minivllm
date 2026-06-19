"""Wave 4 / Phase 6: the fused RMSNorm kernel and its dispatch.

On CPU `rms_norm` must equal the RMSNorm reference exactly (so logit parity is
untouched). The Triton kernel itself is validated against the reference only when
CUDA is available — skipped on the CPU dev box, but the test ships so the kernel
is checked on any GPU box.
"""

from __future__ import annotations

import pytest
import torch

from minivllm.kernels import _ref_rms_norm, rms_norm, triton_available
from minivllm.layers import RMSNorm


def test_dispatch_matches_module_on_cpu():
    torch.manual_seed(0)
    x = torch.randn(4, 32)
    norm = RMSNorm(32, eps=1e-6)
    norm.weight.data.normal_()
    # The module delegates to rms_norm; both must equal the explicit reference.
    expected = _ref_rms_norm(x, norm.weight, 1e-6)
    assert torch.allclose(norm(x), expected, atol=1e-6)
    assert torch.allclose(rms_norm(x, norm.weight, 1e-6), expected, atol=1e-6)


@pytest.mark.skipif(not triton_available(), reason="needs CUDA + Triton")
def test_triton_kernel_matches_reference_on_cuda():
    torch.manual_seed(0)
    x = torch.randn(8, 1024, device="cuda")
    w = torch.randn(1024, device="cuda")
    out = rms_norm(x, w, 1e-6)  # routes to the Triton kernel
    ref = _ref_rms_norm(x, w, 1e-6)
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
