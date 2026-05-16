"""A practical dataloader sample file.

This module shows a minimal but complete pattern for:
1) Reading samples from an annotation file
2) Building a torch Dataset
3) Exposing train/val/test dataloaders via LightningDataModule

Annotation file format (one sample per line):
	relative/path/to/file.npy,3
	relative/path/to/another.npy 7
"""

import logging

import torch
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
	ToTensor,
)
from data_loader import DataModule

logger = logging.getLogger(__name__)


class config:
	video_root = "/mnt/code-luoxi-pegasus/MAC_ACM_MM/data/video"
	sam3d_body_root = "/mnt/code-luoxi-pegasus/MAC_ACM_MM/data/sam3d_body/inference"
	ann_file_root = "/mnt/code-luoxi-pegasus/MAC_ACM_MM/data/annotations/"
	num_frames = 32
	transform = Compose([
		Resize((224, 224)),
		Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	training = True
	fine2coarse = None
	batch_size = 4
	num_workers = 0



if __name__ == "__main__":
	
	dm = DataModule(config)
	dm.setup("fit")
	batch = next(iter(dm.train_dataloader()))

	print("Train batch:")
	for key, value in batch.items():
		print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else value}")

	dm.setup("validate")
	batch = next(iter(dm.val_dataloader()))
	print("Validation batch:")
	for key, value in batch.items():
		print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else value}")

	dm.setup("test")
	batch = next(iter(dm.test_dataloader()))
	print("Test batch:")
	for key, value in batch.items():
		print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else value}")