"""Temporal Contextualization for InternVideo2 MHSA.

Three steps per block:
  1. Select top-n attention tokens per frame.
  2. Aggregate selected tokens into K summary tokens via DPC-KNN.
  3. Inject summary tokens into K/V of subsequent blocks.

Reference: Kim et al., "Leveraging Temporal Contextualization..." (2024).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dpc_knn(x: torch.Tensor, k: int = 8, knn: int = 5):
    """Density-Peaks Clustering with KNN density estimation.

    Args:
        x:   (B, N, D) feature tokens
        k:   number of cluster centers (= summary tokens)
        knn: neighbors used to compute local density ρ

    Returns:
        summary:    (B, k, D)  centroids of the k clusters
        center_idx: (B, k)     indices of the chosen seed tokens
    """
    B, N, D = x.shape
    assert k <= N and knn <= N

    # pairwise squared distance in normalized space (cosine-like)
    xn = F.normalize(x, dim=-1)
    dist = torch.cdist(xn, xn)  # (B, N, N)

    # local density ρ_i = exp(-mean(top-k smallest dists))
    knn_dists, _ = dist.topk(knn + 1, dim=-1, largest=False)  # +1 to skip self
    rho = (-knn_dists[..., 1:].mean(-1)).exp()  # (B, N)

    # for each point, δ = min dist to a higher-density point; if highest density, δ = max dist
    rho_diff = rho.unsqueeze(-1) < rho.unsqueeze(-2)  # (B, N, N): True if other has higher ρ
    masked_dist = dist.masked_fill(~rho_diff, float('inf'))
    delta, _ = masked_dist.min(dim=-1)  # (B, N)
    # the max-density point gets δ = global max dist
    rho_max = rho.argmax(dim=-1)  # (B,)
    for b in range(B):
        delta[b, rho_max[b]] = dist[b].max()

    score = rho * delta  # (B, N)
    center_idx = score.topk(k, dim=-1).indices  # (B, k)
    centers = torch.gather(x, 1, center_idx.unsqueeze(-1).expand(-1, -1, D))  # (B, k, D)

    # soft assign each point to nearest center (cosine); average to form summary
    cn = F.normalize(centers, dim=-1)
    sim = xn @ cn.transpose(1, 2)  # (B, N, k)
    assign = sim.argmax(-1)  # (B, N)
    summary = torch.zeros_like(centers)
    counts = torch.zeros(B, k, device=x.device)
    summary.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), x)
    counts.scatter_add_(1, assign, torch.ones_like(assign, dtype=torch.float))
    summary = summary / counts.clamp(min=1.0).unsqueeze(-1)
    return summary, center_idx


class TCState:
    """Carries summary tokens across stacked TCMHSA blocks within one forward."""
    def __init__(self):
        self.summary: torch.Tensor | None = None  # (B, k, D)


class TCMHSA(nn.Module):
    """Wraps an existing MHSA module; on forward:

      1) Run inner attention, capture attention weights.
      2) Pick top-n_select tokens per frame by attention score (averaged over heads & queries).
      3) Aggregate via DPC-KNN into k_summary tokens → write to state.
      4) On NEXT block call, the state.summary is concatenated to keys/values so every
         token can attend to global spatio-temporal context.

    This implementation calls the inner MHA twice when state.summary is non-empty:
    once on the bare sequence (to harvest seeds), and once with concatenated K/V (to inject).
    """

    def __init__(self, inner_attn: nn.Module, dim: int, num_heads: int,
                 n_select: int = 16, k_summary: int = 8,
                 n_frames: int = 8, spatial_tokens: int = 196):
        super().__init__()
        self.inner = inner_attn
        self.dim = dim
        self.num_heads = num_heads
        self.n_select = n_select
        self.k_summary = k_summary
        self.n_frames = n_frames
        self.spatial_tokens = spatial_tokens

        self.summary_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def _harvest_seeds(self, x: torch.Tensor, attn_weights: torch.Tensor):
        """Pick top-n tokens per frame by attention score.

        x:           (B, N, D),  N = n_frames * spatial_tokens
        attn_weights:(B, N, N)   query-summed across heads
        """
        B, N, D = x.shape
        T, S = self.n_frames, self.spatial_tokens
        # token "importance" = how much it is attended TO (column sum)
        importance = attn_weights.mean(dim=1)  # (B, N)
        importance = importance.view(B, T, S)
        topk = importance.topk(self.n_select, dim=-1).indices  # (B, T, n_select)
        # gather corresponding tokens
        x_grid = x.view(B, T, S, D)
        seeds = torch.gather(x_grid, 2,
                             topk.unsqueeze(-1).expand(-1, -1, -1, D))  # (B, T, n, D)
        return seeds.reshape(B, T * self.n_select, D)

    def forward(self, x: torch.Tensor, state: TCState) -> torch.Tensor:
        # ── inject summary into K/V if available ─────────────────────────────────
        if state.summary is not None:
            kv = torch.cat([x, state.summary], dim=1)
            out, attn = self.inner(x, kv, kv, need_weights=True, average_attn_weights=True)
            # attn here is (B, N_q, N_kv) including summary positions; harvest seeds
            # using only the orig-x slice of attn:
            attn_orig = attn[:, :, : x.shape[1]]
        else:
            out, attn = self.inner(x, x, x, need_weights=True, average_attn_weights=True)
            attn_orig = attn

        # ── update summary for next block ────────────────────────────────────────
        seeds = self._harvest_seeds(x, attn_orig)            # (B, T*n, D)
        summary, _ = dpc_knn(seeds, k=self.k_summary, knn=5)  # (B, k, D)
        summary = summary + self.summary_ffn(summary)         # residual FFN
        state.summary = summary
        return out
