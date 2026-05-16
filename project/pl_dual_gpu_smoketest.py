#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import RichProgressBar


class RandomClassificationDataset(Dataset):
    def __init__(self, size: int = 2048, in_dim: int = 128, num_classes: int = 10):
        self.x = torch.randn(size, in_dim)
        self.y = torch.randint(low=0, high=num_classes, size=(size,))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class RandomDataModule(LightningDataModule):
    def __init__(
        self,
        train_samples: int,
        val_samples: int,
        in_dim: int,
        num_classes: int,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset: Optional[RandomClassificationDataset] = None
        self.val_dataset: Optional[RandomClassificationDataset] = None

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_dataset = RandomClassificationDataset(
                size=self.train_samples,
                in_dim=self.in_dim,
                num_classes=self.num_classes,
            )
            self.val_dataset = RandomClassificationDataset(
                size=self.val_samples,
                in_dim=self.in_dim,
                num_classes=self.num_classes,
            )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized, call setup('fit') first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized, call setup('fit') first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )


class LitToyClassifier(LightningModule):
    def __init__(self, in_dim: int = 128, hidden_dim: int = 256, num_classes: int = 10, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log_dict(
            {"train/loss": loss, "train/acc": acc},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log_dict(
            {"val/loss": loss, "val/acc": acc},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_train_start(self):
        print(
            f"[rank={self.global_rank}] world_size={self.trainer.world_size}, "
            f"device={self.device}, local_rank={self.local_rank}"
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Lightning dual-GPU DDP smoke test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=int, default=0, help="GPU count, default is 2")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=6048)
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--in-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=40960)
    parser.add_argument("--val-samples", type=int, default=10240)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--precision", default="32-true", help="e.g. 32-true, 16-mixed, bf16-mixed")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)

    gpu_count = torch.cuda.device_count()
    if gpu_count < args.devices:
        raise RuntimeError(
            f"Requested devices={args.devices}, but only {gpu_count} CUDA device(s) are visible."
        )


    if isinstance(args.devices, int):
        devicie = [int(args.devices)]  # DDP expects a list of device ids, e.g. [0, 1]
    elif isinstance(args.devices, list):
        devicie = args.devices
    else:
        print(
            f"Using {args.devices} GPUs for training, the batch size will be automatically multiplied by the number of GPUs."
        )

    datamodule = RandomDataModule(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        in_dim=args.in_dim,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = LitToyClassifier(
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        lr=args.lr,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=devicie,
        strategy="ddp",
        max_epochs=args.max_epochs,
        precision=args.precision,
        callbacks=[RichProgressBar(leave=True)],
        log_every_n_steps=1,
        enable_checkpointing=False,
        num_sanity_val_steps=1,
    )

    trainer.fit(model, datamodule=datamodule)
    print("\nDual-GPU DDP smoke test finished successfully.")


if __name__ == "__main__":
    main()
