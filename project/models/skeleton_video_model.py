import torch
import torch.nn as nn
from typing import Optional, Tuple, cast
from transformers import AutoImageProcessor, AutoModel

try:
    from mamba_ssm import Mamba2 as MambaBlock
except ImportError:  # pragma: no cover
    from mamba_ssm import Mamba as MambaBlock


DINOV3_DIMS = {
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m": 768,
    "facebook/dinov3-convnext-small-pretrain-lvd1689m": 768,
    "facebook/dinov3-convnext-base-pretrain-lvd1689m": 1024,
    "facebook/dinov3-convnext-large-pretrain-lvd1689m": 1536,
}


class DINOv3ConvNeXtBackbone(nn.Module):
    def __init__(self, model_name: str, freeze: bool = True):
        super().__init__()
        processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(1, 1, 3, 1, 1)
        std = torch.tensor(processor.image_std, dtype=torch.float32).view(1, 1, 3, 1, 1)
        self.register_buffer("dino_mean", mean, persistent=False)
        self.register_buffer("dino_std", std, persistent=False)

        if freeze:
            self.backbone.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        if x.dtype != torch.float32:
            x = x.float()

        # Support both raw pixels and already-normalized tensors.
        # 1) uint8/0-255 -> scale to [0, 1]
        if x.max() > 1.5:
            x = x / 255.0

        # 2) only normalize if tensor still looks like non-normalized image range
        # (typically in [0, 1]); skip to avoid double-normalization.
        if x.min() >= 0.0 and x.max() <= 1.5:
            dino_mean = cast(torch.Tensor, self.dino_mean)
            dino_std = cast(torch.Tensor, self.dino_std)
            x = (x - dino_mean) / (dino_std + 1e-6)

        out = self.backbone(x.view(b * t, c, h, w), return_dict=True)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        else:
            feat = out.last_hidden_state.mean(dim=1)
        return feat.view(b, t, -1)


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


class SkeletonVideoMambaClassifier(nn.Module):
    """Two-branch model.

    Fine branch: skeleton -> motion features -> Mamba -> fine classifier.
    Coarse branch: DINO per-frame features -> Mamba -> coarse classifier.
    """

    def __init__(
        self,
        num_joints: int,
        num_classes: int,
        num_coarse_classes: int = 7,
        dino_model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        dino_freeze: bool = True,
        d_model: int = 256,
        dino_d_model: int = 256,
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

        # fine branch (skeleton + mamba)
        self.normalizer = SkeletonNormalizer(root_idx=root_idx, scale_joints=scale_joints)
        self.motion_builder = MotionFeatureBuilder()
        self.skeleton_embed = nn.Linear(num_joints * 9, d_model)
        self.skeleton_pre_norm = nn.LayerNorm(d_model)
        self.skeleton_mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.skeleton_post_norm = nn.LayerNorm(d_model)

        # coarse branch (dino + mamba)
        if dino_model_name not in DINOV3_DIMS:
            raise ValueError(f"Unsupported dino_model_name: {dino_model_name}")
        self.dino_backbone = DINOv3ConvNeXtBackbone(model_name=dino_model_name, freeze=dino_freeze)
        self.dino_proj = nn.Linear(DINOV3_DIMS[dino_model_name], dino_d_model)
        self.dino_pre_norm = nn.LayerNorm(dino_d_model)
        self.dino_mamba = MambaBlock(
            d_model=dino_d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dino_post_norm = nn.LayerNorm(dino_d_model)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fine_classifier = nn.Linear(d_model, num_classes)
        self.coarse_classifier = nn.Linear(dino_d_model, num_coarse_classes)

    def forward(self, kpt_3d: torch.Tensor, frames: Optional[torch.Tensor] = None, return_dict: bool = False):
        if kpt_3d.ndim != 4:
            raise ValueError(f"Expected kpt_3d shape (B, T, J, 3), got {tuple(kpt_3d.shape)}")

        x = self.normalizer(kpt_3d)

        # fine branch: skeleton + mamba
        feat = self.motion_builder(x)  # (B, T, J, 9)
        batch_size, time_steps, num_joints, channels = feat.shape
        feat = feat.reshape(batch_size, time_steps, num_joints * channels)
        feat = self.skeleton_embed(feat)
        feat = feat + self.skeleton_mamba(self.skeleton_pre_norm(feat))
        feat = self.skeleton_post_norm(feat)
        feat = self.dropout(feat.mean(dim=1))
        fine_logits = self.fine_classifier(feat)

        # coarse branch: dino + mamba
        dino_feat = self.dino_backbone(frames) # (B, T, dino_d_model)
        dino_feat = self.dino_proj(dino_feat)
        dino_feat = dino_feat + self.dino_mamba(self.dino_pre_norm(dino_feat))
        dino_feat = self.dino_post_norm(dino_feat)
        dino_feat = self.dropout(dino_feat.mean(dim=1))
        coarse_logits = self.coarse_classifier(dino_feat)

        if return_dict:
            return {
                "fine_logits": fine_logits,
                "coarse_logits": coarse_logits,
            }

        return fine_logits


def build_skeleton_video_model(
    num_joints: int,
    num_classes: int,
    num_coarse_classes: int = 7,
    dino_model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    dino_freeze: bool = True,
    d_model: int = 256,
    dino_d_model: int = 256,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    root_idx: int = 0,
    scale_joints: Optional[Tuple[int, int]] = None,
    dropout: float = 0.0,
) -> SkeletonVideoMambaClassifier:
    return SkeletonVideoMambaClassifier(
        num_joints=num_joints,
        num_classes=num_classes,
        num_coarse_classes=num_coarse_classes,
        dino_model_name=dino_model_name,
        dino_freeze=dino_freeze,
        d_model=d_model,
        dino_d_model=dino_d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        root_idx=root_idx,
        scale_joints=scale_joints,
        dropout=dropout,
    )
