import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    from mamba_ssm import Mamba2 as MambaBlock
except ImportError:  # pragma: no cover
    from mamba_ssm import Mamba as MambaBlock


class SkeletonNormalizer(nn.Module):
    """Normalize 3D keypoints by root-centering and optional scale normalization."""

    def __init__(
        self,
        root_idx: int = 0,
        scale_joints: Optional[Tuple[int, int]] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.root_idx = root_idx
        self.scale_joints = scale_joints
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, T, J, C), got {tuple(x.shape)}")
        if x.size(-1) < 3:
            raise ValueError("Expected at least 3 coordinates per joint.")

        root = x[:, :, self.root_idx : self.root_idx + 1, :3]
        x = x[:, :, :, :3] - root

        if self.scale_joints is not None:
            joint_a, joint_b = self.scale_joints
            scale = torch.norm(x[:, :, joint_a] - x[:, :, joint_b], dim=-1, keepdim=True)
            scale = scale.mean(dim=1, keepdim=True)
            x = x / (scale.unsqueeze(-1) + self.eps)

        return x


class MotionFeatureBuilder(nn.Module):
    """Build position, velocity and acceleration features."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = x
        vel = torch.zeros_like(pos)
        vel[:, 1:] = pos[:, 1:] - pos[:, :-1]
        acc = torch.zeros_like(pos)
        acc[:, 1:] = vel[:, 1:] - vel[:, :-1]
        return torch.cat([pos, vel, acc], dim=-1)


class SkeletonMambaClassifier(nn.Module):
    """Minimal and stable skeleton classification baseline for 3D keypoints.

    Input shape:
        (B, T, J, 3)

    Pipeline:
        normalize -> motion features -> flatten joints -> linear embed -> Mamba -> pooling -> classifier
    """

    def __init__(
        self,
        num_joints: int,
        num_classes: int,
        num_coarse_classes: int = 7,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        root_idx: int = 0,
        scale_joints: Optional[Tuple[int, int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.num_classes = num_classes
        self.num_coarse_classes = num_coarse_classes
        self.normalizer = SkeletonNormalizer(root_idx=root_idx, scale_joints=scale_joints)
        self.motion_builder = MotionFeatureBuilder()
        self.embed = nn.Linear(num_joints * 9, d_model)
        self.pre_norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.post_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fine_classifier = nn.Linear(d_model, num_classes)
        self.coarse_classifier = nn.Linear(d_model, num_coarse_classes)

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, T, J, 3), got {tuple(x.shape)}")

        x = self.normalizer(x)

        # branch 1: fine classification with mamba temporal modeling
        feat = self.motion_builder(x)  # (B, T, J, 9)
        batch_size, time_steps, num_joints, channels = feat.shape
        feat = feat.reshape(batch_size, time_steps, num_joints * channels)
        feat = self.embed(feat)
        feat = feat + self.mamba(self.pre_norm(feat))
        feat = self.post_norm(feat)
        feat = feat.mean(dim=1)
        feat = self.dropout(feat)
        fine_logits = self.fine_classifier(feat)
        coarse_logits = self.coarse_classifier(feat)

        if return_dict:
            return {
                "fine_logits": fine_logits,
                "coarse_logits": coarse_logits,
            }

        return fine_logits


def build_skeleton_mamba_model(
    num_joints: int,
    num_classes: int,
    num_coarse_classes: int = 7,
    d_model: int = 256,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    root_idx: int = 0,
    scale_joints: Optional[Tuple[int, int]] = None,
    dropout: float = 0.0,
) -> SkeletonMambaClassifier:
    return SkeletonMambaClassifier(
        num_joints=num_joints,
        num_classes=num_classes,
        num_coarse_classes=num_coarse_classes,
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        root_idx=root_idx,
        scale_joints=scale_joints,
        dropout=dropout,
    )
