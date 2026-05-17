import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import cv2
from typing import Optional, Tuple

# fallback hardcoded mapping (used when no fine2coarse.txt is provided)
# 0-4→0(body), 5-10→1(head), 11-23→2(upper limb),
# 24-31→3(lower limb), 32-37→4(body-hand), 38-47→5(head-hand), 48-51→6(leg-hand)
_DEFAULT_FINE2COARSE = [0]*5 + [1]*6 + [2]*13 + [3]*8 + [4]*6 + [5]*10 + [6]*4


def uniform_subsample_indices(src_t: int, target_t: int) -> torch.Tensor:
    """Return uniformly sampled indices of length target_t from range [0, src_t)."""
    if src_t <= 0:
        raise ValueError(f"src_t must be > 0, got {src_t}")
    if target_t <= 0:
        raise ValueError(f"target_t must be > 0, got {target_t}")

    idx_float = torch.linspace(0, max(src_t - 1, 0), target_t, dtype=torch.float32)
    return torch.round(idx_float).long()


def load_fine2coarse_file(path: str):
    """Parse fine2coarse.txt.

    Returns:
        fine2coarse: list[int], fine2coarse[fine_label] = coarse_label
        coarse_names: dict[int, str], coarse_names[coarse_label] = name
    """
    fine2coarse = {}
    coarse_names = {}
    section = None

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('coarse label'):
                section = 'coarse'
                continue
            if line.startswith('fine2coarse'):
                section = 'fine2coarse'
                continue

            parts = line.split(None, 1)
            if len(parts) < 2:
                continue

            if section == 'coarse':
                coarse_names[int(parts[0])] = parts[1].strip()

            elif section == 'fine2coarse':
                coarse_idx = int(parts[1].strip())
                range_str = parts[0].strip()
                if '-' in range_str:
                    start, end = range_str.split('-')
                    for i in range(int(start), int(end) + 1):
                        fine2coarse[i] = coarse_idx
                else:
                    fine2coarse[int(range_str)] = coarse_idx

    num_fine = max(fine2coarse.keys()) + 1
    fine2coarse_list = [fine2coarse[i] for i in range(num_fine)]
    return fine2coarse_list, coarse_names

class MA52Dataset(Dataset):
    """MA 52 video action classification dataset.

    Annotation file format (space-separated, filename only):
        test5545.mp4 0
        test5546.mp4 3

    Args:
        ann_file:    path to annotation txt (train/val/test_list_videos.txt)
        video_root:  root directory containing all video files
        sam3d_body_root: root directory containing SAM3D body, include 2d/3d pose and bbox annotations
        num_frames:  number of frames T to uniformly sample per video
        transform:   torchvision transforms applied to each PIL frame
        training:    random segment jitter when True, center sampling when False
        fine2coarse: list mapping fine→coarse label; uses built-in default if None

    Returns per sample:
        video_name:  str, e.g. "test5545.mp4"
        frames:       (T, C, H, W)
        2dkpt:      (T, 70, 2)
        3dkpt:      (T, 70, 3)
        fine_label:   int in [0, 51]
        coarse_label: int in [0, 6]
    """

    def __init__(self, ann_file: Path, video_root: Path, sam3d_body_root: Path, num_frames: int = 8,
                 transform=None, fine2coarse: Optional[list] = None, load_frames: bool = True, load_kpt_2d: bool = False, load_kpt_3d: bool = False):

        self.num_frames = num_frames
        self.transform = transform
        _fine2coarse = fine2coarse if fine2coarse is not None else _DEFAULT_FINE2COARSE
        self.samples = self._prepare_sample(ann_file, video_root, sam3d_body_root, _fine2coarse)

        self._load_frame: bool = load_frames
        self._load_2dkpt: bool = load_kpt_2d
        self._load_3dkpt: bool = load_kpt_3d

    @staticmethod
    def _prepare_sample(ann_file, video_root, sam3d_body_root, fine2coarse):
        samples = []
        with open(ann_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.rsplit(',', 1) if ',' in line else line.rsplit(' ', 1)
                filename, fine_label = parts[0].strip(), int(parts[1].strip())
                coarse_label = fine2coarse[fine_label]
                video_path = os.path.join(video_root, filename)
                sam3d_body_path = sam3d_body_root / filename
                sam3d_body_path_list = list(sam3d_body_path.glob(f"*.npz"))
                # 按照frame idx排序
                sam3d_body_path_list.sort(key=lambda p: int(p.stem.split('_')[0]))
                non_detected_idx = list(sam3d_body_path.glob("none_detected_frames.txt"))
                if non_detected_idx:
                    with open(str(non_detected_idx[0]), 'r') as f_none:
                        non_detected_idx = set(line.strip() for line in f_none if line.strip())

                else:
                    non_detected_idx = None
                samples.append((filename, video_path, sam3d_body_path_list, non_detected_idx, fine_label, coarse_label))
        return samples

    def _read_frames_by_indices(self, video_path: str, frame_indices: torch.Tensor):

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise RuntimeError(f"Empty video: {video_path}")
        
        frames = []
        for idx in frame_indices.tolist():
            idx = int(idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
            frames.append(torch.from_numpy(frame).to(torch.float32))  # (H, W, C) in BGR order

        cap.release()
        return torch.stack(frames, dim=0).permute(0, 3, 1, 2)  # (T, C, H, W)

    def _load_sam3d_body(self, sam3d_body_path: str, non_detected_idx: Optional[set]):
        kpt_2d, kpt_3d = None, None

        _info = np.load(sam3d_body_path, allow_pickle=True)['output'].item()

        if self._load_2dkpt: 
            # load 2d keypoints (T, 17, 2)

            kpt_2d = _info['pred_keypoints_2d']  # (total_frames, 17, 2)
        
        
        if self._load_3dkpt:
            # load 3d keypoints (T, 17, 3)
            kpt_3d = _info['pred_keypoints_3d']  # (total_frames, 17, 3)

        return kpt_2d, kpt_3d

    @staticmethod
    def _ensure_frame_kpt_shape(kpt: torch.Tensor) -> torch.Tensor:
        """Normalize loaded keypoint tensor to shape (J, C)."""
        if kpt.ndim == 2:
            return kpt
        if kpt.ndim == 3:
            # Some npz may store shape (1, J, C); keep first frame.
            return kpt[0]
        raise ValueError(f"Unsupported keypoint shape: {tuple(kpt.shape)}")

    def _load_kpts_by_indices(self, sam3d_body_path_list, sampled_indices: torch.Tensor):
        selected_paths = [sam3d_body_path_list[int(i)] for i in sampled_indices.tolist()]

        kpt_2d_frames = []
        kpt_3d_frames = []
        for sam3d_body_path in selected_paths:
            kpt_2d, kpt_3d = self._load_sam3d_body(sam3d_body_path, None)

            if self._load_2dkpt and kpt_2d is not None:
                kpt_2d_t = torch.from_numpy(kpt_2d)
                kpt_2d_frames.append(self._ensure_frame_kpt_shape(kpt_2d_t))

            if self._load_3dkpt and kpt_3d is not None:
                kpt_3d_t = torch.from_numpy(kpt_3d)
                kpt_3d_frames.append(self._ensure_frame_kpt_shape(kpt_3d_t))

        kpt_2d_out = None
        kpt_3d_out = None
        if self._load_2dkpt and len(kpt_2d_frames) > 0:
            kpt_2d_out = torch.stack(kpt_2d_frames, dim=0)
        if self._load_3dkpt and len(kpt_3d_frames) > 0:
            kpt_3d_out = torch.stack(kpt_3d_frames, dim=0)

        return kpt_2d_out, kpt_3d_out

    def _compute_sample_indices(
        self,
        video_total_frames: Optional[int],
        kpt_total_frames: Optional[int],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute aligned sampling indices for video frames and kpt files.

        When both modalities exist, indices are aligned by timeline ratio so the sampled
        positions correspond as closely as possible.
        """
        if video_total_frames is None and kpt_total_frames is None:
            raise ValueError("At least one of video_total_frames or kpt_total_frames must be provided.")

        ref_total = video_total_frames if video_total_frames is not None else kpt_total_frames
        assert ref_total is not None
        base_idx = uniform_subsample_indices(ref_total, self.num_frames)

        video_idx = None
        if video_total_frames is not None:
            video_idx = torch.clamp(base_idx, min=0, max=video_total_frames - 1)

        kpt_idx = None
        if kpt_total_frames is not None:
            if ref_total <= 1:
                kpt_idx = torch.zeros_like(base_idx)
            else:
                scale = (kpt_total_frames - 1) / (ref_total - 1)
                kpt_idx = torch.round(base_idx.float() * scale).long()
            kpt_idx = torch.clamp(kpt_idx, min=0, max=kpt_total_frames - 1)

        return video_idx, kpt_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, video_path, sam3d_body_path_list, non_detected_idx, fine_label, coarse_label = self.samples[idx]

        frame_indices = None
        kpt_indices = None
        frames = None
        kpt_2d = None
        kpt_3d = None
        
        # 1. 计算采样索引
        video_total = None
        if self._load_frame:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open: {video_path}")
            video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if video_total <= 0:
                raise RuntimeError(f"Empty video: {video_path}")

        kpt_total = len(sam3d_body_path_list) if (self._load_2dkpt or self._load_3dkpt) else None
        if kpt_total is not None and kpt_total <= 0:
            kpt_total = None

        frame_indices, kpt_indices = self._compute_sample_indices(video_total, kpt_total)

        # load frame images
        if self._load_frame:
            assert frame_indices is not None
            frames = self._read_frames_by_indices(video_path, frame_indices)

            if self.transform is not None:
                frames = [self.transform(f) for f in frames]
                frames = torch.stack(frames, dim=0)  # (T, C, H, W)

        if self._load_2dkpt or self._load_3dkpt:
            if len(sam3d_body_path_list) > 0 and kpt_indices is not None:
                kpt_2d, kpt_3d = self._load_kpts_by_indices(sam3d_body_path_list, kpt_indices)
            else:
                kpt_2d, kpt_3d = None, None

        # time uniform sampling
        if not self._load_frame or frames is None:
            frames = torch.zeros((self.num_frames, 3, 224, 224), dtype=torch.float32)  # default to zeros if not loaded
        if not (self._load_2dkpt and kpt_2d is not None):
            kpt_2d = torch.zeros((self.num_frames, 70, 2), dtype=torch.float32)  # default to zeros if not loaded
        if not (self._load_3dkpt and kpt_3d is not None):
            kpt_3d = torch.zeros((self.num_frames, 70, 3), dtype=torch.float32)  # default to zeros if not loaded

        sample = {
            "video_name": filename, 
            "frames": frames,  # (T, C, H, W)
            "kpt_2d": kpt_2d,  # (T, 70, 2)
            "kpt_3d": kpt_3d,  # (T, 70, 3)
            "fine_label": fine_label,
            "coarse_label": coarse_label
        }

        return sample