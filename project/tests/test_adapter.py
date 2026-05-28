import torch
import pytest


def test_adapter_forward_shape():
    from adapter_module import Adapter3DResNet

    B, T, S, C = 2, 8, 49, 1024  # batch, time, spatial tokens, channels
    L = T * S
    x = torch.randn(B, L, C)
    adapter = Adapter3DResNet(channels=C, reduction=4, t=T, h=7, w=7)

    out = adapter(x)
    assert out.shape == x.shape, f"expected {x.shape}, got {out.shape}"


def test_adapter_residual_preserves_zero_init():
    """A freshly-initialized adapter with the up-proj zeroed should be ~identity."""
    from adapter_module import Adapter3DResNet
    B, T, S, C = 2, 8, 49, 1024
    L = T * S
    x = torch.randn(B, L, C)
    adapter = Adapter3DResNet(channels=C, reduction=4, t=T, h=7, w=7, zero_init_up=True)
    out = adapter(x)
    assert torch.allclose(out, x, atol=1e-5), "zero-init adapter must be identity"
