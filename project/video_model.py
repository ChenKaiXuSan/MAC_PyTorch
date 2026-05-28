"""Dual-branch micro-action recognition model.

This file will grow over the four milestones:
  M1: CoarseBranch (VideoMAE v2 frozen + 3D-ResNet Adapter + head)
  M2: FineBranch   (TC-InternVideo2 + head)
  M3: DualBranchVideo (combines both)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from adapter_module import Adapter3DResNet


# ── VideoMAE v2 loader ─────────────────────────────────────────────────────────

def load_videomae_v2(model_path: str = 'OpenGVLab/VideoMAEv2-Large'):
    """Load VideoMAE v2 from HuggingFace (custom code, trust_remote_code=True)."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                      low_cpu_mem_usage=False)
    _rematerialize_meta(model)
    _restore_sinusoidal_pos_embed(model)
    feat_dim = getattr(model.config, 'hidden_size', None) or {
        'OpenGVLab/VideoMAEv2-Base': 768,
        'OpenGVLab/VideoMAEv2-Large': 1024,
        'OpenGVLab/VideoMAEv2-Huge': 1280,
    }.get(model_path)
    if feat_dim is None:
        raise ValueError(f"Cannot determine hidden_size for '{model_path}'")
    print(f'Loaded {model_path}  (hidden_size={feat_dim})')
    return model, feat_dim


def _restore_sinusoidal_pos_embed(model: nn.Module):
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
    real_pe = torch.tensor(table, dtype=torch.float32).unsqueeze(0)
    if isinstance(vit.pos_embed, nn.Parameter):
        vit.pos_embed = nn.Parameter(real_pe, requires_grad=False)
    else:
        vit.register_buffer('pos_embed', real_pe)


def _rematerialize_meta(module: nn.Module):
    for name, buf in list(module.named_buffers(recurse=False)):
        if buf.is_meta:
            module.register_buffer(name, torch.zeros(buf.shape, dtype=buf.dtype))
    for name, param in list(module.named_parameters(recurse=False)):
        if param.is_meta:
            param.data = torch.zeros(param.shape, dtype=param.dtype)
    for name, attr in list(vars(module).items()):
        if isinstance(attr, torch.Tensor) and attr.is_meta:
            setattr(module, name, torch.zeros(attr.shape, dtype=attr.dtype))
    for child in module.children():
        _rematerialize_meta(child)


def _freeze(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()
    return module


# ── Coarse branch (M1) ─────────────────────────────────────────────────────────

class CoarseBranch(nn.Module):
    """VideoMAE v2 (frozen) + 3D-ResNet Adapter + linear head → 7-class coarse logits.

    The adapter is hooked AFTER each VideoMAE transformer block via a forward-hook.
    Only adapter params + head are trainable.
    """

    def __init__(self, vmae_path: str = 'OpenGVLab/VideoMAEv2-Large',
                 num_coarse: int = 7,
                 num_frames: int = 16,
                 spatial_tokens_per_frame: int = 14 * 14,  # for 224 input, patch=16
                 adapter_reduction: int = 4):
        super().__init__()
        self.vmae, self.dim = load_videomae_v2(vmae_path)
        self.vmae = _freeze(self.vmae)

        # VideoMAEv2 tubelet size is typically 2 along time → effective T = num_frames/2
        # We resolve t/h/w lazily on first forward, then bind adapters.
        self.num_frames = num_frames
        self.adapter_reduction = adapter_reduction
        self.adapters: Optional[nn.ModuleList] = None
        self._hooks: List = []
        self._bound = False

        self.norm = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, num_coarse)

    def _bind_adapters(self, t: int, h: int, w: int):
        blocks = self.vmae.model.blocks  # VideoMAEv2 attribute
        n = len(blocks)
        self.adapters = nn.ModuleList([
            Adapter3DResNet(channels=self.dim, reduction=self.adapter_reduction,
                            t=t, h=h, w=w, zero_init_up=True)
            for _ in range(n)
        ])
        # Move adapters to the same device/dtype as the backbone
        ref = next(self.vmae.parameters())
        self.adapters.to(device=ref.device, dtype=ref.dtype)

        for i, block in enumerate(blocks):
            adapter = self.adapters[i]
            def make_hook(adapter):
                def hook(module, inputs, output):
                    # output: (B, L, C) for VideoMAE block
                    return adapter(output)
                return hook
            h_handle = block.register_forward_hook(make_hook(adapter))
            self._hooks.append(h_handle)
        self._bound = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) → VideoMAE expects (B, C, T, H, W)
        B, T, Cc, H, W = x.shape
        x_vmae = x.permute(0, 2, 1, 3, 4)

        if not self._bound:
            # Probe: run one block to learn token layout
            with torch.no_grad():
                feat = self.vmae.model.patch_embed(x_vmae)
            # feat shape: (B, L, C)
            L = feat.shape[1]
            # Try standard VideoMAEv2: tubelet=2 → t=T/2; spatial = (H/16)*(W/16)
            t = T // 2
            spatial = L // t
            h = int(spatial ** 0.5)
            w = spatial // h
            assert h * w == spatial, f"non-square spatial layout: spatial={spatial}"
            self._bind_adapters(t=t, h=h, w=w)

        # Full forward through VideoMAE (hooks inject adapters automatically)
        feat = self.vmae.model.forward_features(x_vmae)  # (B, L, C) or (B, C)
        if feat.dim() == 3:
            feat = feat.mean(dim=1)  # global pool
        logits = self.head(self.norm(feat))
        return logits


# ── InternVideo2 loader ────────────────────────────────────────────────────────

def load_internvideo2(model_path: str = 'OpenGVLab/InternVideo2-Stage1-L14'):
    """Load InternVideo2 from HuggingFace. Model id may need adjustment based on HF availability;
    fallback to a smaller variant or local download if needed.
    """
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                      low_cpu_mem_usage=False)
    _rematerialize_meta(model)
    # InternVideo2 hidden size: L14=1024, B14=768
    feat_dim = getattr(model.config, 'hidden_size', None) or {
        'OpenGVLab/InternVideo2-Stage1-B14': 768,
        'OpenGVLab/InternVideo2-Stage1-L14': 1024,
    }.get(model_path)
    if feat_dim is None:
        raise ValueError(f"hidden_size unknown for '{model_path}'")
    print(f'Loaded {model_path}  (hidden_size={feat_dim})')
    return model, feat_dim


# ── Fine branch (M2) ───────────────────────────────────────────────────────────

class FineBranch(nn.Module):
    """InternVideo2 (e2e fine-tuned) with TC injection → 52-class fine logits.

    Replaces each transformer block's self-attention with TCMHSA. The TC state
    is reset at the start of every forward call.
    """

    def __init__(self, intv2_path: str = 'OpenGVLab/InternVideo2-Stage1-L14',
                 num_fine: int = 52, num_frames: int = 16,
                 n_select: int = 16, k_summary: int = 8,
                 spatial_tokens_per_frame: int = 14 * 14):
        super().__init__()
        self.intv2, self.dim = load_internvideo2(intv2_path)
        self.num_frames = num_frames
        self.spatial_tokens = spatial_tokens_per_frame
        self.n_select = n_select
        self.k_summary = k_summary

        self._patch_tc()

        self.norm = nn.LayerNorm(self.dim)
        # 1-layer divided ST-attention head: temporal-then-spatial attn over (B, T, D) tokens
        self.head_norm = nn.LayerNorm(self.dim)
        self.head_attn = nn.MultiheadAttention(self.dim, num_heads=8, batch_first=True)
        self.head_fc = nn.Linear(self.dim, num_fine)

    def _patch_tc(self):
        """Replace each transformer block's attention with TCMHSA.

        InternVideo2 block layout varies; assume model.blocks[i].attn is the MHSA.
        """
        from tc_module import TCMHSA
        blocks = getattr(self.intv2, 'blocks', None) or getattr(self.intv2.model, 'blocks')
        for i, blk in enumerate(blocks):
            inner = blk.attn
            blk.attn = TCMHSA(inner_attn=inner, dim=self.dim, num_heads=8,
                              n_select=self.n_select, k_summary=self.k_summary,
                              n_frames=self.num_frames // 2,  # tubelet=2 typically
                              spatial_tokens=self.spatial_tokens)
        # Patch each block's forward to thread TCState through
        # NOTE: This is implementation-dependent on InternVideo2. The block's forward
        # must be edited to pass `state=self._tc_state` into TCMHSA. We rebind via:
        for i, blk in enumerate(blocks):
            orig_forward = blk.forward
            def make_fwd(blk_local, attn):
                def fwd(x, *args, **kwargs):
                    # delegate to attention with state
                    return attn(x, state=self._tc_state) + x  # residual handled here
                return fwd
            # Best-effort patch — adjust to InternVideo2's actual block API in M2 preflight.
            blk.forward = make_fwd(blk, blk.attn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W). InternVideo2 typically expects (B, C, T, H, W).
        B, T, Cc, H, W = x.shape
        x_intv = x.permute(0, 2, 1, 3, 4)
        from tc_module import TCState
        self._tc_state = TCState()
        feat = self.intv2(x_intv)  # exact API depends on InternVideo2 release
        # feat: best case (B, T_eff, D); pool to (B, D) via head_attn
        if feat.dim() == 2:
            video_feat = feat
        else:
            # treat sequence as (B, N, D); use cls-like mean
            h = self.head_norm(feat)
            h, _ = self.head_attn(h, h, h, need_weights=False)
            video_feat = h.mean(dim=1)
        logits = self.head_fc(self.norm(video_feat))
        return logits


# ── Dual branch (M3) ───────────────────────────────────────────────────────────

class DualBranchVideo(nn.Module):
    """Both branches in one module. Returns (fine_logits, coarse_logits).

    Stage 1: both branches trainable per their internal rules.
    Stage 2: caller freezes everything except .coarse.head and .fine.head_fc.
    """

    def __init__(self, vmae_path='OpenGVLab/VideoMAEv2-Large',
                 intv2_path='OpenGVLab/InternVideo2-Stage1-L14',
                 num_coarse=7, num_fine=52, num_frames=16):
        super().__init__()
        self.coarse = CoarseBranch(vmae_path=vmae_path, num_coarse=num_coarse,
                                    num_frames=num_frames)
        self.fine = FineBranch(intv2_path=intv2_path, num_fine=num_fine,
                                num_frames=num_frames)

    def forward(self, x: torch.Tensor):
        fine_logits = self.fine(x)
        coarse_logits = self.coarse(x)
        return fine_logits, coarse_logits

    def freeze_for_stage2(self):
        """Freeze everything; caller will re-init heads and unfreeze them."""
        for p in self.parameters():
            p.requires_grad_(False)
