import torch
from torch.utils.data import WeightedRandomSampler


def test_rare_resampler_weights_rare_classes_higher():
    from my_dataset import compute_rare_resampler_weights

    # 52 fake classes; class i has count proportional to 52 - i (heavy head, sparse tail)
    counts = torch.tensor([52.0 - i for i in range(52)])
    weights = compute_rare_resampler_weights(class_counts=counts,
                                              fine_labels=[0]*52 + [51]*5,
                                              rare_threshold_rank=40)
    # the 5 tail samples (class 51) should have higher sample weight than the 52 head samples
    head_w = sum(weights[:52]) / 52
    tail_w = sum(weights[52:]) / 5
    assert tail_w > head_w * 5  # tail samples weighted at least 5x head
