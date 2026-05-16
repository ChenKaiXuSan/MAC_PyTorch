import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


# ── feature dimensions ─────────────────────────────────────────────────────────

DINOV3_DIMS = {
    'dinov3_convnext_tiny':  768,
    'dinov3_convnext_small': 768,
    'dinov3_convnext_base':  1024,
    'dinov3_convnext_large': 1536,
    'facebook/dinov3-convnext-tiny-pretrain-lvd1689m': 768,
    'facebook/dinov3-convnext-small-pretrain-lvd1689m': 768,
    'facebook/dinov3-convnext-base-pretrain-lvd1689m': 1024,
    'facebook/dinov3-convnext-large-pretrain-lvd1689m': 1536,
}


# ── backbone loaders ───────────────────────────────────────────────────────────

def load_dinov3_convnext(model_name: str) -> nn.Module:
    """Load DINOv3-pretrained ConvNeXt from a local .pth file."""

    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    dino_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    dino_model.requires_grad_(False)

    mean = torch.tensor(
        processor.image_mean, dtype=torch.float32
    ).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, dtype=torch.float32).view(
        1, 3, 1, 1
    )
    dino_model.register_buffer("dino_mean", mean, persistent=False)
    dino_model.register_buffer("dino_std", std, persistent=False)


    return dino_model


VMAE_DIMS = {
    'OpenGVLab/VideoMAEv2-Base':  768,
    'OpenGVLab/VideoMAEv2-Large': 1024,
    'OpenGVLab/VideoMAEv2-Huge':  1280,
    'OpenGVLab/VideoMAEv2-giant': 1408,
}


def load_videomae_v2(model_path: str):
    """Load VideoMAEv2 from HuggingFace hub or a local directory.

    VideoMAEv2 uses custom code so requires AutoModel + trust_remote_code=True.
    VideoMAEv1 (MCG-NJU/videomae-*) works with VideoMAEModel directly.

    Args:
        model_path: HuggingFace model ID (e.g. 'OpenGVLab/VideoMAEv2-Base')
                    or local path to a saved model directory.

    Returns:
        model:    frozen model
        feat_dim: hidden size (768 for Base, 1024 for Large, etc.)
    """
    processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )

    mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
    model.register_buffer("vmae_mean", mean, persistent=False)
    model.register_buffer("vmae_std", std, persistent=False)

    model.requires_grad_(False)
    return model

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


# ── helper ─────────────────────────────────────────────────────────────────────

def _freeze(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()
    return module


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
    ):
        super().__init__()
        self.convnext = _freeze(convnext)
        self.vmae     = _freeze(vmae)

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

    def forward(self, x: torch.Tensor):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # ── Branch 1: DINOv3-ConvNeXt + TCN ─────────────────────────────────
        with torch.no_grad():
            f1 = self.convnext(x.view(B * T, C, H, W))  # (B*T, D1)
        f1 = self.temporal(f1.pooler_output.view(B, T, -1))  # (B, D1)
        fine_logits = self.fine_head(f1)          # (B, num_fine)

        # ── Branch 2: VideoMAEv2 (full clip) ─────────────────────────────────
        # VideoMAEv2 expects (B, C, T, H, W); our x is (B, T, C, H, W)
        with torch.no_grad():
            f2 = self.vmae(x.permute(0, 2, 1, 3, 4))  # (B, D2)
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
    convnext = load_dinov3_convnext(model_name_1)
    vmae = load_videomae_v2(vmae_path)
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
