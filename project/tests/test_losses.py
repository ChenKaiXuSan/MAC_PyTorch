import torch


def test_wce_smoke():
    from losses import WeightedCrossEntropyLoss
    n_classes = 5
    counts = torch.tensor([100., 50., 20., 10., 5.])
    loss = WeightedCrossEntropyLoss(class_counts=counts, gamma=0.5)
    logits = torch.randn(8, n_classes)
    targets = torch.randint(0, n_classes, (8,))
    v = loss(logits, targets)
    assert v.dim() == 0 and v.item() > 0


def test_wce_rare_class_weighted_higher():
    """Loss on a rare-class sample should be > loss on a head-class sample for the same logits."""
    from losses import WeightedCrossEntropyLoss
    counts = torch.tensor([100., 10.])
    loss = WeightedCrossEntropyLoss(class_counts=counts, gamma=1.0)
    logits = torch.tensor([[0.1, 0.0]])  # slightly favors class 0
    l_head = loss(logits, torch.tensor([0]))
    l_rare = loss(logits, torch.tensor([1]))
    assert l_rare > l_head


def test_focal_smoke():
    from losses import ClassBalancedFocalLoss
    counts = torch.tensor([100., 50., 20., 10., 5.])
    loss = ClassBalancedFocalLoss(class_counts=counts, gamma_w=0.5, gamma_f=2.0)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    v = loss(logits, targets)
    assert v.dim() == 0 and v.item() > 0
