import torch


def test_ensemble_basic():
    """Soft ensembling: fine probs reweighted by coarse probs through fine2coarse map."""
    from utils import ensemble_fine_with_coarse

    # 4 fine classes, 2 coarse classes; fine [0,1]→coarse 0, fine [2,3]→coarse 1
    p_fine = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    p_coarse = torch.tensor([[0.9, 0.1]])
    fine2coarse = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    out = ensemble_fine_with_coarse(p_fine, p_coarse, fine2coarse)
    # raw: p_fine * coarse_w = [0.4*0.9, 0.3*0.9, 0.2*0.1, 0.1*0.1]
    #    = [0.36, 0.27, 0.02, 0.01], sum = 0.66
    # normalized: [0.5455, 0.4091, 0.0303, 0.0152]
    expected = torch.tensor([[0.5455, 0.4091, 0.0303, 0.0152]])
    assert torch.allclose(out, expected, atol=1e-3), f"got {out}"
    # sums to 1
    assert torch.allclose(out.sum(-1), torch.ones(1), atol=1e-5)


def test_ensemble_returns_per_sample():
    """Verify ensembling handles batched input correctly."""
    from utils import ensemble_fine_with_coarse

    p_fine = torch.tensor([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])
    p_coarse = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    fine2coarse = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    out = ensemble_fine_with_coarse(p_fine, p_coarse, fine2coarse)
    # Sample 0: coarse class 0, fine class 2/3 zeroed → [0.5, 0.5, 0.0, 0.0]
    # Sample 1: coarse class 1, fine class 0/1 zeroed → [0.0, 0.0, 0.5, 0.5]
    expected = torch.tensor([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])
    assert torch.allclose(out, expected, atol=1e-5), f"got {out}"
