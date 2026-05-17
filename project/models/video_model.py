import torch
import torch.nn as nn
from contextlib import nullcontext
from typing import Optional
from transformers import AutoImageProcessor, AutoModel

from mamba_ssm import Mamba2

# ── feature dimensions ─────────────────────────────────────────────────────────

DINOV3_DIMS = {
    'facebook/dinov3-convnext-tiny-pretrain-lvd1689m': 768,
    'facebook/dinov3-convnext-small-pretrain-lvd1689m': 768,
    'facebook/dinov3-convnext-base-pretrain-lvd1689m': 1024,
    'facebook/dinov3-convnext-large-pretrain-lvd1689m': 1536,
}


# ── backbone loaders ───────────────────────────────────────────────────────────

class DINOv3ConvNeXtBackbone(nn.Module):
    """DINOv3 ConvNeXt backbone with controllable forward behavior."""

    def __init__(self, model_name: str, freeze: bool = True):
        super().__init__()
        processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("dino_mean", mean, persistent=False)
        self.register_buffer("dino_std", std, persistent=False)

        if freeze:
            self.backbone.requires_grad_(False)

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
    ):
    
        out = self.backbone(x, return_dict=True)

        if return_dict:
            return out

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state.mean(dim=1)


VMAE_DIMS = {
    'OpenGVLab/VideoMAEv2-Base':  768,
    'OpenGVLab/VideoMAEv2-Large': 1024,
    'OpenGVLab/VideoMAEv2-Huge':  1280,
    'OpenGVLab/VideoMAEv2-giant': 1408,
}


class VideoMAEv2Backbone(nn.Module):
    """VideoMAEv2 backbone with controllable forward behavior."""

    def __init__(self, model_path: str, freeze: bool = True):
        super().__init__()
        processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )

        mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("vmae_mean", mean, persistent=False)
        self.register_buffer("vmae_std", std, persistent=False)

        if freeze:
            self.backbone.requires_grad_(False)

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
        cls_only: bool = True,
    ):
        
        out = self.backbone(x, return_dict=True)

        if return_dict:
            return out

        if not cls_only:
            return out.last_hidden_state

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0]

# ── TCN temporal module (used by DINOv3 branch only) ─────────────────────────

class TCNTemporalBlock(nn.Module):
    """Residual temporal block for sequence modeling on (B, D, T)."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class SpatialTemporalTCN(nn.Module):
    """Stack of TCN layers. Input (B, T, D) → output (B, D)."""

    def __init__(self, d_model: int, n_layers: int = 2, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TCNTemporalBlock(
                    channels=d_model,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        # Global temporal pooling -> (B, D)
        return x.mean(dim=-1)

# ── main model ─────────────────────────────────────────────────────────────────

class DualBranchVideo(nn.Module):
    """Asymmetric dual-branch video model.

    Branch 1 (fine, 52 classes):
        frozen DINOv3-ConvNeXt  →  per-frame features (B*T, D1)
        → TCN temporal modeling →  (B, D1)
        → fine_head             →  (B, num_fine)

    Branch 2 (coarse, 7 classes):
        frozen VideoMAEv2  →  CLS token (B, D2)   [video-native, no extra temporal needed]
        → coarse_head      →  (B, num_coarse)
    """

    def __init__(
        self,
        convnext: nn.Module,   # DINOv3-ConvNeXt backbone
        vmae: nn.Module,       # VideoMAEv2
        convnext_dim: int,
        vmae_dim: int,
        num_fine: int = 52,
        num_coarse: int = 7,
        kernel_size: int = 3,
        n_layers: int = 1,
        dropout: float = 0.0,
        no_grad_convnext: bool = True,
        no_grad_vmae: bool = True,
    ):
        super().__init__()
        self.convnext = convnext
        self.vmae     = vmae
        self.no_grad_convnext = no_grad_convnext
        self.no_grad_vmae = no_grad_vmae

        self.temporal    = SpatialTemporalTCN(
            d_model=convnext_dim,
            n_layers=n_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.fine_head   = nn.Linear(convnext_dim, num_fine)
        self.coarse_head = nn.Linear(vmae_dim, num_coarse)

    def train(self, mode: bool = True):
        super().train(mode)
        self.convnext.eval()
        self.vmae.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        no_grad_convnext: Optional[bool] = None,
        no_grad_vmae: Optional[bool] = None,
    ):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        if no_grad_convnext is None:
            no_grad_convnext = self.no_grad_convnext
        if no_grad_vmae is None:
            no_grad_vmae = self.no_grad_vmae

        # ── Branch 1: DINOv3-ConvNeXt + TCN ─────────────────────────────────
        f1 = self.convnext(
            x.view(B * T, C, H, W),
            use_no_grad=no_grad_convnext,
            return_dict=False,
        )
        f1 = self.temporal(f1.view(B, T, -1))  # (B, D1)
        fine_logits = self.fine_head(f1)          # (B, num_fine)

        # ── Branch 2: VideoMAEv2 (full clip) ─────────────────────────────────
        # VideoMAEv2 expects (B, C, T, H, W); our x is (B, T, C, H, W)
        f2 = self.vmae(
            x.permute(0, 2, 1, 3, 4),
            use_no_grad=no_grad_vmae,
            return_dict=False,
            cls_only=True,
        )
        coarse_logits = self.coarse_head(f2)       # (B, num_coarse)

        return fine_logits, coarse_logits


# ── factory ────────────────────────────────────────────────────────────────────

def build_dual_video_model(
    # DINOv3-ConvNeXt (branch 1, fine)
    model_name_1: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    # VideoMAEv2 (branch 2, coarse)
    vmae_path: str = "OpenGVLab/VideoMAEv2-Base",
    # heads
    num_fine: int = 52,
    num_coarse: int = 7,
    # TCN (branch 1 only)
    kernel_size: int = 3,
    n_layers: int = 1,
    dropout: float = 0.0,
) -> DualBranchVideo:
    convnext = DINOv3ConvNeXtBackbone(model_name=model_name_1, freeze=True)
    vmae = VideoMAEv2Backbone(model_path=vmae_path, freeze=True)
    return DualBranchVideo(
        convnext, vmae,
        convnext_dim=DINOV3_DIMS[model_name_1],
        vmae_dim=VMAE_DIMS[vmae_path],
        num_fine=num_fine,
        num_coarse=num_coarse,
        kernel_size=kernel_size,
        n_layers=n_layers,
        dropout=dropout,
    )
