import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset

# fallback hardcoded mapping (used when no fine2coarse.txt is provided)
# 0-4→0(body), 5-10→1(head), 11-23→2(upper limb),
# 24-31→3(lower limb), 32-37→4(body-hand), 38-47→5(head-hand), 48-51→6(leg-hand)
_DEFAULT_FINE2COARSE = [0]*5 + [1]*6 + [2]*13 + [3]*8 + [4]*6 + [5]*10 + [6]*4


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
        root:        directory where video files live
        num_frames:  number of frames T to uniformly sample per video
        transform:   torchvision transforms applied to each PIL frame
        training:    random segment jitter when True, center sampling when False
        fine2coarse: list mapping fine→coarse label; uses built-in default if None

    Returns per sample:
        frames:       (T, C, H, W)
        fine_label:   int in [0, 51]
        coarse_label: int in [0, 6]
    """

    def __init__(self, ann_file: str, root: str, num_frames: int = 8,
                 transform=None, training: bool = True, fine2coarse: list = None):
        self.root = root
        self.num_frames = num_frames
        self.transform = transform
        self.training = training
        self._fine2coarse = fine2coarse if fine2coarse is not None else _DEFAULT_FINE2COARSE
        self.samples = self._load_annotations(ann_file)

    def _load_annotations(self, ann_file):
        samples = []
        with open(ann_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.rsplit(',', 1) if ',' in line else line.rsplit(' ', 1)
                filename, fine_label = parts[0].strip(), int(parts[1].strip())
                coarse_label = self._fine2coarse[fine_label]
                video_path = os.path.join(self.root, filename)
                samples.append((video_path, fine_label, coarse_label))
        return samples

    def _sample_indices(self, total_frames: int):
        """Segment-based uniform sampling; random jitter during training."""
        seg = total_frames / self.num_frames
        indices = []
        for i in range(self.num_frames):
            start = int(i * seg)
            end = max(start, int((i + 1) * seg) - 1)
            idx = random.randint(start, end) if self.training else (start + end) // 2
            indices.append(min(idx, total_frames - 1))
        return indices

    def _read_frames(self, video_path: str):
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise RuntimeError(f"Empty video: {video_path}")

        frames, last = [], None
        for idx in self._sample_indices(total):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                last = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(last)
        cap.release()

        first_valid = next((f for f in frames if f is not None), None)
        if first_valid is None:
            raise RuntimeError(f"No readable frames: {video_path}")
        for i in range(len(frames)):
            if frames[i] is None:
                frames[i] = frames[i - 1] if i > 0 else first_valid
        return frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, fine_label, coarse_label = self.samples[idx]
        frames = self._read_frames(video_path)
        if self.transform is not None:
            frames = [self.transform(f) for f in frames]
        return torch.stack(frames, dim=0), fine_label, coarse_label  # (T, C, H, W)

    @staticmethod
    def collate_fn(batch):
        frames, fine_labels, coarse_labels = zip(*batch)
        return (
            torch.stack(frames, dim=0),        # (B, T, C, H, W)
            torch.as_tensor(fine_labels),       # (B,)
            torch.as_tensor(coarse_labels),     # (B,)
        )
