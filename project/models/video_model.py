import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from model import convnext_tiny, convnext_small, convnext_base, convnext_large


# ── feature dimensions ─────────────────────────────────────────────────────────

DINOV3_DIMS = {
    'dinov3_convnext_tiny':  768,
    'dinov3_convnext_small': 768,
    'dinov3_convnext_base':  1024,
    'dinov3_convnext_large': 1536,
}

_CONVNEXT_BUILDERS = {
    'dinov3_convnext_tiny':  convnext_tiny,
    'dinov3_convnext_small': convnext_small,
    'dinov3_convnext_base':  convnext_base,
    'dinov3_convnext_large': convnext_large,
}


# ── backbone loaders ───────────────────────────────────────────────────────────

def load_dinov3_convnext(model_name: str, weights_path: str) -> nn.Module:
    """Load DINOv3-pretrained ConvNeXt from a local .pth file."""
    backbone = _CONVNEXT_BUILDERS[model_name](num_classes=1000)
    state = torch.load(weights_path, map_location='cpu')
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    non_head_missing = [k for k in missing if 'head' not in k]
    if non_head_missing:
        print(f'[WARN] {model_name}: unexpected missing keys: {non_head_missing}')
    print(f'Loaded {model_name} from {weights_path}  '
          f'(missing={len(missing)}, unexpected={len(unexpected)})')
    return backbone


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
    from transformers import AutoModel, AutoConfig
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                      low_cpu_mem_usage=False)

    # Recompute any remaining meta tensors (pos_embed is sinusoidal, not in checkpoint)
    _rematerialize_meta(model)
    _restore_sinusoidal_pos_embed(model)

    # resolve feat_dim: prefer config, fall back to lookup table
    try:
        feat_dim = model.config.hidden_size
    except AttributeError:
        feat_dim = VMAE_DIMS.get(model_path)
        if feat_dim is None:
            raise ValueError(
                f"Cannot determine hidden_size for '{model_path}'. "
                f"Pass one of: {list(VMAE_DIMS.keys())}"
            )
    print(f'Loaded VideoMAEv2 from {model_path}  (hidden_size={feat_dim})')
    return model, feat_dim


def _restore_sinusoidal_pos_embed(model: nn.Module):
    """Recompute sinusoidal pos_embed for VisionTransformer (not saved in checkpoint)."""
    import numpy as np
    vit = getattr(model, 'model', None)
    if vit is None or not hasattr(vit, 'pos_embed'):
        return
    pe = vit.pos_embed
    if not (isinstance(pe, torch.Tensor) and pe.shape[0] == 1):
        return
    _, n_pos, d_hid = pe.shape

    def angle_vec(pos):
        return [pos / np.power(10000, 2 * (j // 2) / d_hid) for j in range(d_hid)]

    table = np.array([angle_vec(i) for i in range(n_pos)])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    real_pe = torch.tensor(table, dtype=torch.float32).unsqueeze(0)  # (1, n_pos, d_hid)

    if isinstance(vit.pos_embed, nn.Parameter):
        vit.pos_embed = nn.Parameter(real_pe, requires_grad=False)
    else:
        vit.register_buffer('pos_embed', real_pe)


def _rematerialize_meta(module: nn.Module):
    """Replace any meta tensors with zero-initialized real tensors on CPU."""
    for name, buf in list(module.named_buffers(recurse=False)):
        if buf.is_meta:
            module.register_buffer(name, torch.zeros(buf.shape, dtype=buf.dtype))
    for name, param in list(module.named_parameters(recurse=False)):
        if param.is_meta:
            param.data = torch.zeros(param.shape, dtype=param.dtype)
    # Also catch plain tensor attributes (like pos_embed before our register_buffer fix)
    for name, attr in list(vars(module).items()):
        if isinstance(attr, torch.Tensor) and attr.is_meta:
            setattr(module, name, torch.zeros(attr.shape, dtype=attr.dtype))
    for child in module.children():
        _rematerialize_meta(child)


# ── Mamba2 temporal module (used by DINOv3 branch only) ───────────────────────

class Mamba2Temporal(nn.Module):
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
    """Stack of Mamba2 layers. Input (B, T, D) → output (B, D)."""

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
        → Mamba2 temporal scan  →  (B, D1)
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
        d_state: int = 64,
        n_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.convnext = _freeze(convnext)
        self.vmae     = _freeze(vmae)

        self.temporal    = SpatialTemporalMamba(convnext_dim, d_state, n_layers, dropout)
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

        # ── Branch 1: DINOv3-ConvNeXt + Mamba2 ──────────────────────────────
        with torch.no_grad():
            f1 = self.convnext.forward_features(x.view(B * T, C, H, W))  # (B*T, D1)
        f1 = self.temporal(f1.view(B, T, -1))   # (B, D1)
        fine_logits = self.fine_head(f1)          # (B, num_fine)

        # ── Branch 2: VideoMAEv2 (full clip) ─────────────────────────────────
        # VideoMAEv2 expects (B, C, T, H, W); our x is (B, T, C, H, W)
        with torch.no_grad():
            f2 = self.vmae.model.forward_features(x.permute(0, 2, 1, 3, 4))  # (B, D2)
        coarse_logits = self.coarse_head(f2)       # (B, num_coarse)

        return fine_logits, coarse_logits


# ── factory ────────────────────────────────────────────────────────────────────

def build_dual_video_model(
    # DINOv3-ConvNeXt (branch 1, fine)
    model_name_1: str = 'dinov3_convnext_tiny',
    weights_1: str = '',
    # VideoMAEv2 (branch 2, coarse)
    vmae_path: str = 'MCG-NJU/videomae-v2-base',
    # heads
    num_fine: int = 52,
    num_coarse: int = 7,
    # Mamba2 (branch 1 only)
    d_state: int = 64,
    n_layers: int = 1,
    dropout: float = 0.0,
) -> DualBranchVideo:
    convnext = load_dinov3_convnext(model_name_1, weights_1)
    vmae, vmae_dim = load_videomae_v2(vmae_path)
    return DualBranchVideo(
        convnext, vmae,
        convnext_dim=DINOV3_DIMS[model_name_1],
        vmae_dim=vmae_dim,
        num_fine=num_fine,
        num_coarse=num_coarse,
        d_state=d_state,
        n_layers=n_layers,
        dropout=dropout,
    )
