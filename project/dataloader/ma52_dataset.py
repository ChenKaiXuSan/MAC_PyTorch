import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import cv2

# fallback hardcoded mapping (used when no fine2coarse.txt is provided)
# 0-4→0(body), 5-10→1(head), 11-23→2(upper limb),
# 24-31→3(lower limb), 32-37→4(body-hand), 38-47→5(head-hand), 48-51→6(leg-hand)
_DEFAULT_FINE2COARSE = [0]*5 + [1]*6 + [2]*13 + [3]*8 + [4]*6 + [5]*10 + [6]*4



def uniform_subsample_along_dim(tensor: torch.Tensor, target_t: int, dim: int) -> torch.Tensor:
    """对任意指定维度做均匀采样（不足时重复最近邻帧）。

    Args:
        tensor: 任意形状的输入张量。
        target_t: 目标长度，必须 > 0。
        dim: 需要采样的维度，支持负索引。

    Returns:
        在指定维度上长度为 target_t 的张量。
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor must be torch.Tensor, got {type(tensor)}")
    if target_t <= 0:
        raise ValueError(f"target_t must be > 0, got {target_t}")
    if tensor.ndim == 0:
        raise ValueError("tensor must have at least 1 dimension")

    dim = dim if dim >= 0 else tensor.ndim + dim
    if dim < 0 or dim >= tensor.ndim:
        raise ValueError(
            f"dim out of range: got {dim}, valid range is [0, {tensor.ndim - 1}]"
        )

    src_t = int(tensor.shape[dim])
    if src_t <= 0:
        raise ValueError(f"source length on dim {dim} must be > 0, got {src_t}")

    idx_float = torch.linspace(
        0,
        max(src_t - 1, 0),
        target_t,
        dtype=torch.float32,
        device=tensor.device,
    )
    idx = torch.round(idx_float).long()
    return torch.index_select(tensor, dim, idx)


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


def load_label_names_file(path: str):
    """Parse label_name.txt.

    Returns:
        fine_names: dict[int, str], fine_names[fine_label] = name
    """
    fine_names = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                fine_names[int(parts[0])] = parts[1].strip()
    return fine_names


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
                 transform=None, fine2coarse: list = None):

        self.num_frames = num_frames
        self.transform = transform
        _fine2coarse = fine2coarse if fine2coarse is not None else _DEFAULT_FINE2COARSE
        self.samples = self._prepare_sample(ann_file, video_root, sam3d_body_root, _fine2coarse)

        self._load_frame: bool = True
        self._load_2dkpt: bool = False
        self._load_3dkpt: bool = False

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
                non_detected_idx = list(sam3d_body_path.glob("none_detected_frames.txt"))
                if non_detected_idx:
                    with open(str(non_detected_idx[0]), 'r') as f_none:
                        non_detected_idx = set(line.strip() for line in f_none if line.strip())

                else:
                    non_detected_idx = None
                samples.append((filename, video_path, sam3d_body_path_list, non_detected_idx, fine_label, coarse_label))
        return samples

    def _read_frames(self, video_path: str):

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise RuntimeError(f"Empty video: {video_path}")
        
        frames = []
        for idx in range(total):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
            frames.append(torch.from_numpy(frame).to(torch.float32))  # (H, W, C) in BGR order

        return torch.stack(frames, dim=0).permute(0, 3, 1, 2)  # (T, C, H, W)

    def _load_sam3d_body(self, sam3d_body_path: str, non_detected_idx: set):
        kpt_2d, kpt_3d = None, None

        _info = np.load(sam3d_body_path, allow_pickle=True)['output'].item()

        if self._load_2dkpt: 
            # load 2d keypoints (T, 17, 2)

            kpt_2d = _info['pred_keypoints_2d']  # (total_frames, 17, 2)
        
        
        if self._load_3dkpt:
            # load 3d keypoints (T, 17, 3)
            kpt_3d = _info['pred_keypoints_3d']  # (total_frames, 17, 3)

        return kpt_2d, kpt_3d

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, video_path, sam3d_body_path_list, non_detected_idx, fine_label, coarse_label = self.samples[idx]

        # load frame images
        if self._load_frame:
            frames = self._read_frames(video_path)

            if self.transform is not None:
                frames = [self.transform(f) for f in frames]
                frames = torch.stack(frames, dim=0)  # (T, C, H, W)

        if self._load_2dkpt or self._load_3dkpt:
            # Initialize lists to store keypoints from all body files
            all_kpt_2d = []
            all_kpt_3d = []

            for sam3d_body_path in sam3d_body_path_list:
                kpt_2d, kpt_3d = self._load_sam3d_body(sam3d_body_path, non_detected_idx)
                if kpt_2d is not None:
                    all_kpt_2d.append(torch.from_numpy(kpt_2d))
                    kpt_2d = torch.stack(all_kpt_2d, dim=0)
                if kpt_3d is not None:
                    all_kpt_3d.append(torch.from_numpy(kpt_3d))
                    kpt_3d = torch.stack(all_kpt_3d, dim=0)

        # time uniform sampling
        if self._load_frame:
            frames = uniform_subsample_along_dim(frames, self.num_frames, dim=0)
        else:
            frames = None
        if self._load_2dkpt and kpt_2d is not None:
            kpt_2d = uniform_subsample_along_dim(kpt_2d, self.num_frames, dim=0)
        else:
            kpt_2d = torch.zeros((self.num_frames, 70, 2), dtype=torch.float32)  # default to zeros if not loaded
        if self._load_3dkpt and kpt_3d is not None:
            kpt_3d = uniform_subsample_along_dim(kpt_3d, self.num_frames, dim=0)
        else:            
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