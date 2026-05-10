import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from model import convnext_tiny, convnext_small, convnext_base, convnext_large


FEAT_DIMS = {
    'dinov3_convnext_tiny':  768,
    'dinov3_convnext_small': 768,
    'dinov3_convnext_base':  1024,
    'dinov3_convnext_large': 1536,
}

_BUILDERS = {
    'dinov3_convnext_tiny':  convnext_tiny,
    'dinov3_convnext_small': convnext_small,
    'dinov3_convnext_base':  convnext_base,
    'dinov3_convnext_large': convnext_large,
}


def load_dinov3_convnext(model_name: str, weights_path: str) -> nn.Module:
    """Build ConvNeXt and load DINOv3-pretrained weights from a local .pth file.

    The checkpoint is a bare backbone state dict (no head).
    DINOv3 adds norms.* keys absent in model.py — loaded with strict=False.
    """
    backbone = _BUILDERS[model_name](num_classes=1000)
    state = torch.load(weights_path, map_location='cpu')
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    # expected missing: head.weight / head.bias
    # expected unexpected: norms.3.weight / norms.3.bias  (DINOv3 extra norm)
    non_head_missing = [k for k in missing if 'head' not in k]
    if non_head_missing:
        print(f'[WARN] {model_name}: unexpected missing keys: {non_head_missing}')
    print(f'Loaded {model_name} from {weights_path}  '
          f'(missing={len(missing)}, unexpected={len(unexpected)})')
    return backbone


# ── Mamba2 temporal modules ────────────────────────────────────────────────────

class Mamba2Temporal(nn.Module):
    """Single Mamba2 layer with pre-norm and residual. Input: (B, T, C)."""

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.mamba = Mamba2(d_model=d_model, d_state=d_state,
                            d_conv=d_conv, expand=expand)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.mamba(self.norm(x))
        return residual + self.dropout(x)


class SpatialTemporalMamba(nn.Module):
    """Stack of Mamba2Temporal layers. Input (B, T, D) → output (B, D)."""

    def __init__(self, d_model: int, d_state: int = 64,
                 n_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            Mamba2Temporal(d_model=d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x.mean(dim=1)  # (B, D)


# ── helpers ────────────────────────────────────────────────────────────────────

def _freeze(backbone: nn.Module) -> nn.Module:
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()
    return backbone


# ── main model ─────────────────────────────────────────────────────────────────

class DualDINOv3Video(nn.Module):
    """Two frozen DINOv3-ConvNeXt branches with Mamba2 temporal scan.

    Branch 1: backbone_1 → SpatialTemporalMamba_1 → fine_head   (num_fine  classes)
    Branch 2: backbone_2 → SpatialTemporalMamba_2 → coarse_head (num_coarse classes)

    Trainable: temporal modules + classification heads only.
    """

    def __init__(
        self,
        backbone_1: nn.Module,
        backbone_2: nn.Module,
        feat_dim_1: int,
        feat_dim_2: int,
        num_fine: int = 52,
        num_coarse: int = 7,
        d_state: int = 64,
        n_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone_1 = _freeze(backbone_1)
        self.backbone_2 = _freeze(backbone_2)

        self.temporal_1 = SpatialTemporalMamba(feat_dim_1, d_state, n_layers, dropout)
        self.temporal_2 = SpatialTemporalMamba(feat_dim_2, d_state, n_layers, dropout)

        self.fine_head   = nn.Linear(feat_dim_1, num_fine)
        self.coarse_head = nn.Linear(feat_dim_2, num_coarse)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone_1.eval()
        self.backbone_2.eval()
        return self

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        with torch.no_grad():
            f1 = self.backbone_1.forward_features(x_flat)  # (B*T, D1)
            f2 = self.backbone_2.forward_features(x_flat)  # (B*T, D2)

        f1 = self.temporal_1(f1.view(B, T, -1))  # (B, D1)
        f2 = self.temporal_2(f2.view(B, T, -1))  # (B, D2)

        return self.fine_head(f1), self.coarse_head(f2)  # (B,52), (B,7)


# ── factory ────────────────────────────────────────────────────────────────────

def build_dual_video_model(
    model_name_1: str,
    weights_1: str,
    model_name_2: str,
    weights_2: str,
    num_fine: int = 52,
    num_coarse: int = 7,
    d_state: int = 64,
    n_layers: int = 1,
    dropout: float = 0.0,
) -> DualDINOv3Video:
    backbone_1 = load_dinov3_convnext(model_name_1, weights_1)
    backbone_2 = load_dinov3_convnext(model_name_2, weights_2)
    return DualDINOv3Video(
        backbone_1, backbone_2,
        feat_dim_1=FEAT_DIMS[model_name_1],
        feat_dim_2=FEAT_DIMS[model_name_2],
        num_fine=num_fine,
        num_coarse=num_coarse,
        d_state=d_state,
        n_layers=n_layers,
        dropout=dropout,
    )
