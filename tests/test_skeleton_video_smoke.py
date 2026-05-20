import pytest
import torch
import torch.nn as nn

pytest.importorskip("mamba_ssm")
pytest.importorskip("transformers")

import project.models.skeleton_video_model as svm


class _FakeProcessor:
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]


class _FakeBackbone(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim

    def forward(self, x, return_dict=True):
        n = x.shape[0]

        class _Out:
            pass

        out = _Out()
        out.pooler_output = torch.randn(n, self.feat_dim, device=x.device)
        out.last_hidden_state = torch.randn(n, 4, self.feat_dim, device=x.device)
        return out


@pytest.fixture
def patch_dino(monkeypatch):
    def _fake_processor_from_pretrained(*args, **kwargs):
        return _FakeProcessor()

    def _fake_model_from_pretrained(model_name, *args, **kwargs):
        feat_dim = svm.DINOV3_DIMS[model_name]
        return _FakeBackbone(feat_dim)

    monkeypatch.setattr(svm.AutoImageProcessor, "from_pretrained", staticmethod(_fake_processor_from_pretrained))
    monkeypatch.setattr(svm.AutoModel, "from_pretrained", staticmethod(_fake_model_from_pretrained))


def test_skeleton_video_two_branch_forward_smoke(patch_dino):
    model = svm.build_skeleton_video_model(
        num_joints=70,
        num_classes=52,
        num_coarse_classes=7,
        d_model=64,
        dino_d_model=64,
        d_state=8,
    ).cuda()

    kpt_3d = torch.randn(2, 16, 70, 3).cuda() # b, t, j, c
    frames = torch.randn(2, 16, 3, 224, 224).cuda() # b, t, c, h, w

    outputs = model(kpt_3d, frames=frames, return_dict=True)

    assert outputs["fine_logits"].shape == (2, 52)
    assert outputs["coarse_logits"].shape == (2, 7)
    assert torch.isfinite(outputs["fine_logits"]).all()
    assert torch.isfinite(outputs["coarse_logits"]).all()


def test_skeleton_video_fine_only_forward_smoke(patch_dino):
    model = svm.build_skeleton_video_model(
        num_joints=70,
        num_classes=52,
        num_coarse_classes=7,
        d_model=64,
        dino_d_model=64,
        d_state=8,
    ).cuda()

    kpt_3d = torch.randn(2, 16, 70, 3).cuda()

    outputs = model(kpt_3d, frames=None, return_dict=True)

    assert outputs["fine_logits"].shape == (2, 52)
    assert outputs["coarse_logits"] is None
