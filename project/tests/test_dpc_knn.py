import torch


def test_dpc_knn_finds_three_clusters():
    """Three well-separated Gaussian clusters → DPC-KNN picks one center per cluster."""
    from tc_module import dpc_knn

    torch.manual_seed(0)
    centers = torch.tensor([[0., 0.], [10., 0.], [0., 10.]])
    pts = torch.cat([c + 0.1 * torch.randn(30, 2) for c in centers], dim=0)  # (90, 2)
    pts = pts.unsqueeze(0)  # (B=1, N=90, D=2)

    summary, center_idx = dpc_knn(pts, k=3, knn=5)

    assert summary.shape == (1, 3, 2)
    # the three center indices should land one in each cluster (0-29, 30-59, 60-89)
    buckets = sorted({(idx.item() // 30) for idx in center_idx[0]})
    assert buckets == [0, 1, 2], f"clusters not all represented: {buckets}"
