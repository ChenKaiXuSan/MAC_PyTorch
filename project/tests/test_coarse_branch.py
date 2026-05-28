import pytest
import torch


@pytest.mark.slow
def test_coarse_branch_forward(device):
    from video_model import CoarseBranch
    model = CoarseBranch(num_coarse=7, num_frames=16).to(device)
    x = torch.randn(1, 16, 3, 224, 224, device=device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 7)
    # adapter should be bound after first forward
    assert model._bound
    # only adapters + head should be trainable
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    for n in trainable:
        assert 'adapters' in n or 'head' in n or 'norm' in n, f"unexpected trainable: {n}"
