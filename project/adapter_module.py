import torch
import torch.nn as nn


class Adapter3DResNet(nn.Module):
    """3D-ResNet adapter for VideoMAE v2 PEFT.

    Structure: Down -> 3D-conv -> FC -> ReLU -> 3D-conv -> FC -> Up + residual.

    Input/output: (B, L, C) where L = t * h * w.
    """

    def __init__(self, channels: int, reduction: int = 4,
                 t: int = 8, h: int = 14, w: int = 14,
                 kernel: int = 3, zero_init_up: bool = True):
        super().__init__()
        self.t, self.h, self.w = t, h, w
        c_mid = max(channels // reduction, 16)

        self.down = nn.Linear(channels, c_mid)
        self.conv1 = nn.Conv3d(c_mid, c_mid, kernel, padding=kernel // 2)
        self.fc1 = nn.Linear(c_mid, c_mid)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(c_mid, c_mid, kernel, padding=kernel // 2)
        self.fc2 = nn.Linear(c_mid, c_mid)
        self.up = nn.Linear(c_mid, channels)

        if zero_init_up:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        t, h, w = self.t, self.h, self.w
        assert L == t * h * w, f"L={L} != t*h*w={t*h*w}"

        z = self.down(x)
        # (B, L, c_mid) -> (B, c_mid, t, h, w) for 3D conv
        z3 = z.transpose(1, 2).reshape(B, -1, t, h, w)
        z3 = self.conv1(z3)
        z = z3.reshape(B, -1, L).transpose(1, 2)  # (B, L, c_mid)
        z = self.fc1(z)
        z = self.act(z)

        z3 = z.transpose(1, 2).reshape(B, -1, t, h, w)
        z3 = self.conv2(z3)
        z = z3.reshape(B, -1, L).transpose(1, 2)
        z = self.fc2(z)

        z = self.up(z)
        return x + z
