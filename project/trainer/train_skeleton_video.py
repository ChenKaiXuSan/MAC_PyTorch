#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from models.skeleton_video_model import build_skeleton_video_model


class SkeletonVideoClassificationModule(LightningModule):
	"""Trainer module for skeleton fine branch + DINO coarse branch model."""

	def __init__(self, hparams):
		super().__init__()

		data_cfg = getattr(hparams, "data", hparams)
		loss_cfg = getattr(hparams, "loss", None)
		model_cfg = getattr(hparams, "skeleton_video", None)

		self.lr = float(getattr(loss_cfg, "lr", 1e-3))
		self.weight_decay = float(getattr(loss_cfg, "weight_decay", 1e-2))
		self.fine_loss_weight = float(getattr(loss_cfg, "fine_loss_weight", 1.0))
		self.coarse_loss_weight = float(getattr(loss_cfg, "coarse_loss_weight", 1.0))

		self.num_fine_classes = int(
			getattr(data_cfg, "fine_class_num", getattr(data_cfg, "num_classes", 52))
		)
		self.num_coarse_classes = int(getattr(data_cfg, "coarse_class_num", 7))
		self.num_joints = int(getattr(data_cfg, "num_joints", 70))
		self.root_idx = int(getattr(data_cfg, "root_idx", 0))

		scale_joints = getattr(data_cfg, "scale_joints", None)
		if isinstance(scale_joints, (list, tuple)) and len(scale_joints) == 2:
			scale_joints = (int(scale_joints[0]), int(scale_joints[1]))
		else:
			scale_joints = None

		self.model = build_skeleton_video_model(
			num_joints=self.num_joints,
			num_classes=self.num_fine_classes,
			num_coarse_classes=self.num_coarse_classes,
			dino_model_name=str(
				getattr(
					model_cfg,
					"dino_model_name",
					"facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
				)
			),
			dino_freeze=bool(getattr(model_cfg, "dino_freeze", True)),
			d_model=int(getattr(model_cfg, "d_model", 256)),
			dino_d_model=int(getattr(model_cfg, "dino_d_model", 256)),
			d_state=int(getattr(model_cfg, "d_state", 16)),
			d_conv=int(getattr(model_cfg, "d_conv", 4)),
			expand=int(getattr(model_cfg, "expand", 2)),
			root_idx=self.root_idx,
			scale_joints=scale_joints,
			dropout=float(getattr(model_cfg, "dropout", 0.0)),
		)

		self.save_hyperparameters(ignore=["model"])

		self.fine_acc = MulticlassAccuracy(num_classes=self.num_fine_classes)
		self.fine_f1 = MulticlassF1Score(num_classes=self.num_fine_classes, average="macro")
		self.coarse_acc = MulticlassAccuracy(num_classes=self.num_coarse_classes)
		self.coarse_f1 = MulticlassF1Score(num_classes=self.num_coarse_classes, average="macro")

	def configure_optimizers(self) -> OptimizerLRScheduler:
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

	@staticmethod
	def _unpack_batch(batch):
		if isinstance(batch, dict):
			frames = batch.get("frames", batch.get("video"))
			kpt_3d = batch.get("kpt_3d")
			fine_label = batch.get("fine_label", batch.get("label"))
			coarse_label = batch.get("coarse_label")
		elif isinstance(batch, (tuple, list)):
			if len(batch) >= 4:
				frames, kpt_3d, fine_label, coarse_label = batch[:4]
			else:
				raise ValueError("Unsupported batch format. Expected frames, kpt_3d, fine_label, coarse_label.")
		else:
			raise TypeError(f"Unsupported batch type: {type(batch)}")

		if frames is None or kpt_3d is None or fine_label is None or coarse_label is None:
			raise ValueError("Batch must contain frames, kpt_3d, fine_label and coarse_label.")

		frames = frames.float()
		kpt_3d = kpt_3d.float()
		fine_label = fine_label.long()
		coarse_label = coarse_label.long()

		if frames.ndim != 5:
			raise ValueError(f"Expected frames shape (B, T, C, H, W), got {tuple(frames.shape)}")
		if kpt_3d.ndim != 4:
			raise ValueError(f"Expected kpt_3d shape (B, T, J, 3), got {tuple(kpt_3d.shape)}")

		return frames, kpt_3d, fine_label, coarse_label

	def training_step(self, batch, batch_idx):
		return self._shared_step(batch, stage="train")

	def validation_step(self, batch, batch_idx):
		return self._shared_step(batch, stage="val")

	def test_step(self, batch, batch_idx):
		return self._shared_step(batch, stage="test")

	def _log_metrics(self, stage, loss, fine_logits, coarse_logits, fine_label, coarse_label):
		self.log(
			f"{stage}/loss",
			loss,
			on_step=(stage == "train"),
			on_epoch=True,
			prog_bar=(stage != "train"),
			sync_dist=True,
		)

		self.log(
			f"{stage}/fine_acc",
			self.fine_acc(fine_logits, fine_label),
			on_step=False,
			on_epoch=True,
			prog_bar=True,
			sync_dist=True,
		)
		self.log(
			f"{stage}/fine_f1",
			self.fine_f1(fine_logits, fine_label),
			on_step=False,
			on_epoch=True,
			prog_bar=True,
			sync_dist=True,
		)

		self.log(
			f"{stage}/coarse_acc",
			self.coarse_acc(coarse_logits, coarse_label),
			on_step=False,
			on_epoch=True,
			prog_bar=True,
			sync_dist=True,
		)
		self.log(
			f"{stage}/coarse_f1",
			self.coarse_f1(coarse_logits, coarse_label),
		on_step=False,
		on_epoch=True,
		prog_bar=True,
		sync_dist=True,
	)

	def _shared_step(self, batch, stage: str):
		frames, kpt_3d, fine_label, coarse_label = self._unpack_batch(batch)

		preds = self.model(kpt_3d=kpt_3d, frames=frames, return_dict=True)
		fine_logits = preds["fine_logits"]
		coarse_logits = preds["coarse_logits"]

		fine_loss = F.cross_entropy(fine_logits, fine_label)
		coarse_loss = F.cross_entropy(coarse_logits, coarse_label)

		loss = self.fine_loss_weight * fine_loss + self.coarse_loss_weight * coarse_loss

		self.log(f"{stage}/fine_loss", fine_loss, on_step=(stage == "train"), on_epoch=True, prog_bar=False, sync_dist=True)
		self.log(f"{stage}/coarse_loss", coarse_loss, on_step=(stage == "train"), on_epoch=True, prog_bar=False, sync_dist=True)

		self._log_metrics(stage, loss, fine_logits, coarse_logits, fine_label, coarse_label)
		return loss
