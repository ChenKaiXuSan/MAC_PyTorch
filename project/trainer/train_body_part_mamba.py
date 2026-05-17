#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/trainer/train_two_stream.py
Project: /workspace/skeleton/project/trainer
Created Date: Friday June 7th 2024
Author: Kaixu Chen
-----
Comment:
This file implements the training process for two stream method.
In this two streams are trained separately and then the results of two streams are fused to get the final result.
Here, saving the results and calculating the metrics are done in separate functions.

Have a good code time :)
-----
Last Modified: Friday June 7th 2024 7:50:12 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import torch
import torch.nn.functional as F

from models.skeleton_model import build_skeleton_mamba_model

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
)

class BodyPartMambaClassificationModule(LightningModule):

    def __init__(self, hparams):
        super().__init__()

        # Support both full config and model-only config.
        data_cfg = getattr(hparams, "data", hparams)
        loss_cfg = getattr(hparams, "loss", None)

        self.lr = (
            getattr(loss_cfg, "lr", 0.001)
        )
        self.weight_decay = getattr(loss_cfg, "weight_decay", 0.01)
        self.fine_loss_weight = float(getattr(loss_cfg, "fine_loss_weight", 1.0))
        self.coarse_loss_weight = float(getattr(loss_cfg, "coarse_loss_weight", 1.0))

        self.num_fine_classes = int(
            getattr(data_cfg, "fine_class_num", getattr(data_cfg, "num_classes", 52))
        )
        self.num_coarse_classes = int(getattr(data_cfg, "coarse_class_num", 7))
        self.train_target = str(getattr(getattr(hparams, "model", hparams), "train_target", "fine")).lower()
        if self.train_target not in {"fine", "coarse", "both"}:
            raise ValueError("model.train_target must be one of: fine, coarse, both")

        self.num_joints = int(getattr(data_cfg, "num_joints", 70))
        self.root_idx = int(getattr(data_cfg, "root_idx", 0))
        scale_joints = getattr(data_cfg, "scale_joints", None)
        if isinstance(scale_joints, (list, tuple)) and len(scale_joints) == 2:
            scale_joints = (int(scale_joints[0]), int(scale_joints[1]))
        else:
            scale_joints = None

        self.model = build_skeleton_mamba_model(
            num_joints=self.num_joints,
            num_classes=self.num_fine_classes,
            num_coarse_classes=self.num_coarse_classes,
            d_model=int(getattr(data_cfg, "d_model", 256)),
            d_state=int(getattr(data_cfg, "d_state", 16)),
            d_conv=int(getattr(data_cfg, "d_conv", 4)),
            expand=int(getattr(data_cfg, "expand", 2)),
            root_idx=self.root_idx,
            scale_joints=scale_joints,
            dropout=float(getattr(data_cfg, "dropout", 0.0)),
        )
        
        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters(ignore=["model"])

        self.fine_acc = MulticlassAccuracy(num_classes=self.num_fine_classes)
        self.coarse_acc = MulticlassAccuracy(num_classes=self.num_coarse_classes)
        self.fine_f1_score = MulticlassF1Score(num_classes=self.num_fine_classes, average="macro")
        self.coarse_f1_score = MulticlassF1Score(num_classes=self.num_coarse_classes, average="macro")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        tmax = getattr(self.trainer, "estimated_stepping_batches", None)
        if not isinstance(tmax, int) or tmax <= 0:
            tmax = 1000

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, stage="val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, stage="test")
        return loss

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, dict):
            kpt_3d = batch.get("kpt_3d")
            fine_label = batch.get("fine_label", batch.get("label"))
            coarse_label = batch.get("coarse_label")
        elif isinstance(batch, (tuple, list)):
            if len(batch) >= 3:
                kpt_3d, fine_label, coarse_label = batch[:3]
            else:
                raise ValueError("Unsupported batch format. Expected dict or tuple/list with kpt_3d, fine_label and coarse_label.")
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        if kpt_3d is None or fine_label is None or coarse_label is None:
            raise ValueError("Batch must contain kpt_3d, fine_label and coarse_label.")

        kpt_3d = kpt_3d.float()

        if kpt_3d.ndim != 4:
            raise ValueError(f"Expected kpt_3d shape (B, T, J, 3), got {tuple(kpt_3d.shape)}")

        return kpt_3d, fine_label.long(), coarse_label.long()

    def _log_metrics(self, stage, loss, fine_pred, coarse_pred, fine_label, coarse_label):
        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=(stage != "train"),
            sync_dist=True,
        )

        if self.train_target in {"fine", "both"}:
            self.log(f"{stage}/fine_acc", self.fine_acc(fine_pred, fine_label), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{stage}/fine_f1", self.fine_f1_score(fine_pred, fine_label), on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        if self.train_target in {"coarse", "both"}:
            self.log(f"{stage}/coarse_acc", self.coarse_acc(coarse_pred, coarse_label), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{stage}/coarse_f1", self.coarse_f1_score(coarse_pred, coarse_label), on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

    def _shared_step(self, batch, stage="train"):
        kpt_3d, fine_label, coarse_label = self._unpack_batch(batch)
        preds = self.model(kpt_3d, return_dict=True)
        fine_pred = preds["fine_logits"]
        coarse_pred = preds["coarse_logits"]

        fine_loss = F.cross_entropy(fine_pred, fine_label)
        coarse_loss = F.cross_entropy(coarse_pred, coarse_label)

        if self.train_target == "fine":
            loss = self.fine_loss_weight * fine_loss
        elif self.train_target == "coarse":
            loss = self.coarse_loss_weight * coarse_loss
        else:
            loss = self.fine_loss_weight * fine_loss + self.coarse_loss_weight * coarse_loss

        self._log_metrics(stage, loss, fine_pred, coarse_pred, fine_label, coarse_label)

        return loss
