#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/deep-learning-project-template/project/main.py
Project: /workspace/deep-learning-project-template/project
Created Date: Friday November 29th 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday November 29th 2024 12:51:51 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

import os
import logging
import hydra
from omegaconf import DictConfig

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import (
    RichProgressBar,
    RichModelSummary,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from dataloader.data_loader import DataModule

#####################################
# select different experiment trainer 
#####################################

def train(hparams: DictConfig):
    """the train process for the one fold.

    Args:
        hparams (hydra): the hyperparameters.
        dataset_idx (int): the dataset index for the one fold.
        fold (int): the fold index.

    Returns:
        list: best trained model, data loader
    """

    seed_everything(42, workers=True)

    if isinstance(hparams.train.devices, int):
        devicie = [int(hparams.train.devices)]  # DDP expects a list of device ids, e.g. [0, 1]
    elif isinstance(hparams.train.devices, list):
        devicie = hparams.train.devices
    else:
        logging.info(
            f"Using {hparams.train.devices} GPUs for training, the batch size will be automatically multiplied by the number of GPUs."
        )

    data_module = DataModule(hparams.data)

    # for the tensorboard
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(hparams.train.log_path),
        name="train",
    )

    csv_logger = CSVLogger(
        save_dir=os.path.join(hparams.train.log_path),
        name="train",
    )

    # some callbacks
    progress_bar = RichProgressBar(leave=True)
    rich_model_summary = RichModelSummary(max_depth=3)

    # define the checkpoint becavier.
    model_check_point = ModelCheckpoint(
        filename="{epoch}-{val/loss:.2f}-{val/video_acc:.4f}",
        auto_insert_metric_name=False,
        monitor="val/video_acc",
        mode="max",
        save_last=False,
        save_top_k=2,
    )

    # define the early stop.
    early_stopping = EarlyStopping(
        monitor="val/video_acc",
        patience=3,
        mode="max",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        accelerator="gpu",
        devices=devicie,
        strategy="ddp",
        max_epochs=hparams.train.max_epochs,
        logger=[tb_logger, csv_logger],
        check_val_every_n_epoch=1,
        callbacks=[
            progress_bar,
            rich_model_summary,
            model_check_point,
            early_stopping,
            lr_monitor,
        ],
    )

    trainer.fit(classification_module, data_module)


@hydra.main(
    version_base=None,
    config_path="../configs", # * the config_path is relative to location of the python script
    config_name="train.yaml",
)
def init_params(config):

    train(config)


if __name__ == "__main__":

    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
