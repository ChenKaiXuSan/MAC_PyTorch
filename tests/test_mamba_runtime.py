# pyright: reportMissingImports=false

import pytest
import torch
from typing import Any

from project.models.skeleton_model import SkeletonMambaClassifier


mamba_ssm = pytest.importorskip("mamba_ssm")


def _build_mamba_block(d_model: int = 128) -> Any:
    # Prefer Mamba2; fallback to Mamba for older installations.
    if hasattr(mamba_ssm, "Mamba2"):
        return mamba_ssm.Mamba2(d_model=d_model, d_state=16, d_conv=4, expand=2)
    if hasattr(mamba_ssm, "Mamba"):
        return mamba_ssm.Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
    pytest.skip("mamba_ssm has neither Mamba2 nor Mamba")
    raise RuntimeError("unreachable")


def test_mamba_library_forward_only():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available; skip low-level mamba runtime check")

    device = "cuda"
    block = _build_mamba_block(d_model=128).to(device)
    x = torch.randn(2, 32, 128, device=device)

    try:
        y = block(x)
    except TypeError as exc:
        if "NoneType" in str(exc):
            pytest.fail(
                "mamba_ssm low-level runtime is unavailable (NoneType callable). "
                "Please reinstall/align mamba_ssm + causal-conv1d + triton."
            )
        raise

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("device", ["cuda" if torch.cuda.is_available() else "cpu"])
def test_skeleton_mamba_forward_smoke(device: str):
    model = SkeletonMambaClassifier(
        num_joints=70,
        num_classes=52,
        d_model=128,
        d_state=16,
        d_conv=4,
        expand=2,
    ).to(device)

    x = torch.randn(2, 16, 70, 3, device=device)

    try:
        logits = model(x)
    except TypeError as exc:
        # Known runtime issue when mamba custom op is not correctly loaded.
        if "NoneType" in str(exc):
            pytest.fail(
                "Mamba runtime kernel is unavailable (NoneType callable). "
                "Please check mamba_ssm / causal-conv1d / triton installation."
            )
        raise

    assert logits.shape == (2, 52)
    assert torch.isfinite(logits).all()


def test_skeleton_mamba_input_shape_check():
    model = SkeletonMambaClassifier(num_joints=70, num_classes=52)
    bad_x = torch.randn(2, 70, 3)  # missing T dimension

    with pytest.raises(ValueError, match="Expected input shape"):
        _ = model(bad_x)
