#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /Users/chenkaixu/deep-learning-project-template/project/dataloader/data_loader.py
Project: /Users/chenkaixu/deep-learning-project-template/project/dataloader
Created Date: Saturday November 30th 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday November 30th 2024 12:34:29 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
from typing import Any, Callable, Dict, Optional, Type

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from ma52_dataset import MA52Dataset
from pathlib import Path

logger = logging.getLogger(__name__)

class DataModule(LightningDataModule):
    def __init__(self, opt):
        super().__init__()

        self.video_root = Path(opt.video_root)
        self.sam3d_body_root = Path(opt.sam3d_body_root)
        self.ann_file_root = Path(opt.ann_file_root)
        self.num_frames = opt.num_frames
        self.transform = opt.transform
        self.training = opt.training
        self.fine2coarse = opt.fine2coarse
        self._default_batch_size = opt.batch_size
        self._NUM_WORKERS = opt.num_workers
        
    def prepare_data(self) -> None:
        """here prepare the temp val data path,
        because the val dataset not use the gait cycle index,
        so we directly use the pytorchvideo API to load the video.
        AKA, use whole video to validate the model.
        """
        ...

    def setup(self, stage: Optional[str] = None) -> None:
        """
        assign tran, val, predict datasets for use in dataloaders

        Args:
            stage (Optional[str], optional): trainer.stage, in ('fit', 'validate', 'test', 'predict'). Defaults to None.
        """
        self.train_dataset = MA52Dataset(
            ann_file=self.ann_file_root / "train_list_videos.txt",
            video_root=self.video_root / "train",
            sam3d_body_root=self.sam3d_body_root / "train",
            num_frames=self.num_frames,
            transform=self.transform,
            training=self.training,
            fine2coarse=self.fine2coarse,
        )

        self.val_dataset = MA52Dataset(
            ann_file=self.ann_file_root / "val_list_videos.txt",
            video_root=self.video_root / "val",
            sam3d_body_root=self.sam3d_body_root / "val",
            num_frames=self.num_frames,
            transform=self.transform,
            training=False,
            fine2coarse=self.fine2coarse,
        )

    def collate_fn(self, batch):
        """this function process the batch data, and return the batch data.

        Args:
            batch (list): the batch from the dataset.
            The batch include the one patient info from the json file.
            Here we only cat the one patient video tensor, and label tensor.

        Returns:
            dict: {video: torch.tensor, label: torch.tensor, info: list}
        """
        ...

    def train_dataloader(self) -> DataLoader:
        """ create the train data loader

        Returns:
            DataLoader: _description_
        """        
        train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=self._default_batch_size,
            num_workers=self._NUM_WORKERS,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

        return train_data_loader

    def val_dataloader(self) -> DataLoader:
        """ create the val data loader

        Returns:
            DataLoader: _description_
        """        
        val_data_loader = DataLoader(
            self.val_dataset,
            batch_size=self._default_batch_size,
            num_workers=self._NUM_WORKERS,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
        )

        return val_data_loader

    def test_dataloader(self) -> DataLoader:
        """ create the test data loader

        Returns:
            DataLoader: 
        """
        test_data_loader = DataLoader(
            self.val_dataset,  # use val dataset to test, because the test dataset not have label
            batch_size=self._default_batch_size,
            num_workers=self._NUM_WORKERS,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
        )

        return test_data_loader
