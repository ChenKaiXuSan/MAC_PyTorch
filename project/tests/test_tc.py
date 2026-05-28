import torch


def test_tcmhsa_forward_shape():
    """TCMHSA wraps an existing attention module; output shape matches input."""
    from tc_module import TCMHSA, TCState

    B, N, D, num_heads = 2, 196 * 4, 1024, 16  # 4 frames × 14×14 spatial
    inner = torch.nn.MultiheadAttention(D, num_heads, batch_first=True)
    # crude shim so TCMHSA can call .forward(q, k, v); details inside TCMHSA.
    tc = TCMHSA(inner_attn=inner, dim=D, num_heads=num_heads,
                n_select=16, k_summary=8, n_frames=4, spatial_tokens=196)

    x = torch.randn(B, N, D)
    state = TCState()  # accumulates summary tokens across blocks
    out = tc(x, state=state)
    assert out.shape == x.shape
    # after one block, state should hold summary tokens of shape (B, k, D)
    assert state.summary is not None and state.summary.shape == (B, 8, D)


def test_tcmhsa_gradient_flows():
    from tc_module import TCMHSA, TCState
    B, N, D, num_heads = 2, 196 * 2, 256, 8
    inner = torch.nn.MultiheadAttention(D, num_heads, batch_first=True)
    tc = TCMHSA(inner_attn=inner, dim=D, num_heads=num_heads,
                n_select=8, k_summary=4, n_frames=2, spatial_tokens=196)
    x = torch.randn(B, N, D, requires_grad=True)
    out = tc(x, state=TCState())
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
