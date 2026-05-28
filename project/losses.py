"""Long-tail-aware losses for Stage 2 head retraining."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """L_WCE = -Sigma_c  w_c . y_c . log(p_c),  w_c = 1 / N_c^gamma.

    gamma in [0, 1]. gamma=0 reduces to plain CE; gamma=1 is inverse-frequency.
    """

    def __init__(self, class_counts: torch.Tensor, gamma: float = 0.5):
        super().__init__()
        assert 0.0 <= gamma <= 1.0
        w = 1.0 / class_counts.pow(gamma)
        w = w * (len(class_counts) / w.sum())  # normalize so mean weight ~ 1
        self.register_buffer('weights', w)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weights)


class ClassBalancedFocalLoss(nn.Module):
    """Class-balanced focal:  L = -Sigma_c  (1 - p_c)^gamma_f . w_c . y_c . log(p_c).

    gamma_w controls class-frequency weighting (0..1)
    gamma_f controls hard-sample focusing  (>=0; 2.0 is standard).
    """

    def __init__(self, class_counts: torch.Tensor,
                 gamma_w: float = 0.5, gamma_f: float = 2.0):
        super().__init__()
        w = 1.0 / class_counts.pow(gamma_w)
        w = w * (len(class_counts) / w.sum())
        self.register_buffer('weights', w)
        self.gamma_f = gamma_f

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)               # (B, C)
        p = log_p.exp()
        log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        w_t = self.weights[targets]                          # (B,)
        loss = -((1 - p_t).pow(self.gamma_f)) * w_t * log_p_t
        return loss.mean()
