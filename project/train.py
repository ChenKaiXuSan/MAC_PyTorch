"""Lightning-style training entry for MAR-paper reproduction.

Modes:
  M1: --branch coarse  (single coarse branch)
  M2: --branch fine    (single fine branch)
  M3: --stage 1        (dual branch joint CE training)
  M4: --stage 2        (frozen backbone, re-init heads, WCE+Focal on rare-resampled data)
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchvision import transforms

from losses import ClassBalancedFocalLoss, WeightedCrossEntropyLoss
from my_dataset import (
    MA52Dataset,
    _DEFAULT_FINE2COARSE,
    load_fine2coarse_file,
    make_rare_resampler,
)
from utils import (
    create_lr_scheduler,
    ensemble_fine_with_coarse,
    make_param_groups_stage1,
)


def _get_by_path(dct, path, default=None):
    cur = dct
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _infer_data_root(cfg):
    video_root = _get_by_path(cfg, "data.video_root")
    if video_root:
        p = Path(str(video_root)).expanduser()
        if p.name == "video":
            return str(p.parent)
        return str(p)

    ann_root = _get_by_path(cfg, "data.ann_file_root")
    if ann_root:
        p = Path(str(ann_root)).expanduser()
        if p.name == "annotations":
            return str(p.parent)
        return str(p)

    return None


def _build_args_from_cfg(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        cfg_dict = {}

    data_root = _infer_data_root(cfg_dict) or "/mnt/code-luoxi-pegasus/MAC_ACM_MM/data"

    return argparse.Namespace(
        branch=_get_by_path(cfg_dict, "branch", "coarse"),
        stage=_get_by_path(cfg_dict, "stage", None),
        epochs=_get_by_path(cfg_dict, "train.max_epochs", 50),
        batch_size=_get_by_path(cfg_dict, "data.batch_size", 3),
        num_frames=_get_by_path(cfg_dict, "data.num_frames", 16),
        lr=_get_by_path(cfg_dict, "loss.lr", 1e-3),
        wd=_get_by_path(cfg_dict, "loss.weight_decay", 5e-2),
        data_root=data_root,
        vmae_path=_get_by_path(cfg_dict, "vmae_path", "OpenGVLab/VideoMAEv2-Large"),
        intv2_path=_get_by_path(
            cfg_dict, "intv2_path", "OpenGVLab/InternVideo2-Stage1-L14"
        ),
        output=_get_by_path(cfg_dict, "output", "runs"),
        device=_get_by_path(cfg_dict, "device", "cuda:0"),
        grad_accum=_get_by_path(cfg_dict, "grad_accum", 1),
        lambda_coarse=_get_by_path(cfg_dict, "loss.coarse_loss_weight", 1.0),
        stage1_ckpt=_get_by_path(cfg_dict, "stage1_ckpt", ""),
        num_workers=_get_by_path(cfg_dict, "data.num_workers", 8),
        gpus=_get_by_path(cfg_dict, "train.gpus", 1),
        seed=_get_by_path(cfg_dict, "seed", 42),
        precision=_get_by_path(cfg_dict, "precision", "bf16-mixed"),
    )


def make_transforms(img_size=224):
    return {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(int(img_size * 1.143)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }


class MA52DataModule(pl.LightningDataModule):
    def __init__(self, args, fine2coarse):
        super().__init__()
        self.args = args
        self.fine2coarse = fine2coarse
        self.tfm = make_transforms()
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        def resolve_ann_file(split):
            ann = os.path.join(
                self.args.data_root, f"annotations/{split}_list_videos.txt"
            )
            if os.path.isfile(ann):
                return ann
            raise FileNotFoundError(
                f"Cannot find annotation file: {ann}. "
                f"Expected under <data-root>/annotations/{split}_list_videos.txt"
            )

        def resolve_split_root(split):
            candidates = [
                os.path.join(self.args.data_root, "video", split),
                os.path.join(self.args.data_root, split),
            ]
            for path in candidates:
                if os.path.isdir(path):
                    return path
            raise FileNotFoundError(
                f'Cannot find video directory for split="{split}". Tried: {candidates}. '
                f"Please set --data-root to the dataset root that contains video/{split} or {split}."
            )

        def ds(ann, root, training):
            return MA52Dataset(
                ann_file=ann,
                root=root,
                num_frames=self.args.num_frames,
                transform=self.tfm["train" if training else "val"],
                training=training,
                fine2coarse=self.fine2coarse,
            )

        if self.train_dataset is None:
            self.train_dataset = ds(
                resolve_ann_file("train"), resolve_split_root("train"), True
            )
        if self.val_dataset is None:
            self.val_dataset = ds(
                resolve_ann_file("val"), resolve_split_root("val"), False
            )

    @property
    def num_workers(self):
        if self.args.num_workers is not None:
            return self.args.num_workers
        return min(os.cpu_count() or 4, 8)

    def train_dataloader(self):
        if self.args.stage == 2:
            sampler = make_rare_resampler(
                self.train_dataset, num_fine=len(self.fine2coarse)
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                sampler=sampler,
                pin_memory=True,
                num_workers=self.num_workers,
                collate_fn=MA52Dataset.collate_fn,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=MA52Dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=MA52Dataset.collate_fn,
        )


class MARLightningModule(pl.LightningModule):
    def __init__(self, args, num_coarse, num_fine, fine2coarse):
        super().__init__()
        self.args = args
        self.stage = args.stage
        self.branch = args.branch
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.register_buffer(
            "fine2coarse_tensor",
            torch.as_tensor(fine2coarse, dtype=torch.long),
            persistent=False,
        )

        self.loss_fine = None
        self.loss_coarse = None
        self._stage2_prepared = False

        if self.stage is None:
            self.model = self._build_model_single()
            n_classes = self.num_coarse if self.branch == "coarse" else self.num_fine
            self.train_acc = MulticlassAccuracy(num_classes=n_classes)
            self.val_acc = MulticlassAccuracy(num_classes=n_classes)
            self.val_f1_macro = MulticlassF1Score(
                num_classes=n_classes, average="macro"
            )
            self.val_f1_micro = MulticlassF1Score(
                num_classes=n_classes, average="micro"
            )
        else:
            from video_model import DualBranchVideo

            self.model = DualBranchVideo(
                vmae_path=args.vmae_path,
                intv2_path=args.intv2_path,
                num_coarse=num_coarse,
                num_fine=num_fine,
                num_frames=args.num_frames,
            )
            self.train_fine_acc = MulticlassAccuracy(num_classes=self.num_fine)
            self.train_coarse_acc = MulticlassAccuracy(num_classes=self.num_coarse)
            self.val_body_macro = MulticlassF1Score(
                num_classes=self.num_coarse, average="macro"
            )
            self.val_body_micro = MulticlassF1Score(
                num_classes=self.num_coarse, average="micro"
            )
            self.val_action_macro = MulticlassF1Score(
                num_classes=self.num_fine, average="macro"
            )
            self.val_action_micro = MulticlassF1Score(
                num_classes=self.num_fine, average="micro"
            )

    def _build_model_single(self):
        if self.branch == "coarse":
            from video_model import CoarseBranch

            return CoarseBranch(
                vmae_path=self.args.vmae_path,
                num_coarse=self.num_coarse,
                num_frames=self.args.num_frames,
            )
        if self.branch == "fine":
            from video_model import FineBranch

            return FineBranch(
                intv2_path=self.args.intv2_path,
                num_fine=self.num_fine,
                num_frames=self.args.num_frames,
            )
        raise ValueError(f"unknown branch {self.branch}")

    def setup(self, stage: Optional[str] = None):
        if self.stage == 2 and not self._stage2_prepared:
            self._prepare_stage2()

    def _prepare_stage2(self):
        assert self.args.stage1_ckpt and os.path.exists(self.args.stage1_ckpt), (
            f"--stage1-ckpt is required for stage 2: {self.args.stage1_ckpt}"
        )

        sd = torch.load(self.args.stage1_ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if not isinstance(sd, dict):
            raise ValueError("Invalid checkpoint format for --stage1-ckpt")

        if sd and all(k.startswith("model.") for k in sd.keys()):
            sd = {k[len("model.") :]: v for k, v in sd.items()}

        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        print(
            f"loaded stage1 ckpt; missing={len(missing)} unexpected={len(unexpected)}"
        )

        self.model.freeze_for_stage2()

        d_coarse = self.model.coarse.dim
        d_fine = self.model.fine.dim
        self.model.coarse.norm = torch.nn.LayerNorm(d_coarse)
        self.model.coarse.head = torch.nn.Linear(d_coarse, self.num_coarse)

        self.model.fine.head_norm = torch.nn.LayerNorm(d_fine)
        self.model.fine.head_attn = torch.nn.MultiheadAttention(
            d_fine, num_heads=8, batch_first=True
        )
        self.model.fine.norm = torch.nn.LayerNorm(d_fine)
        self.model.fine.head_fc = torch.nn.Linear(d_fine, self.num_fine)

        head_prefixes = (
            "coarse.norm",
            "coarse.head",
            "fine.head_norm",
            "fine.head_attn",
            "fine.norm",
            "fine.head_fc",
        )
        for name, p in self.model.named_parameters():
            if name.startswith(head_prefixes):
                p.requires_grad_(True)

        train_ds = self.trainer.datamodule.train_dataset
        fine_counts = torch.zeros(self.num_fine)
        coarse_counts = torch.zeros(self.num_coarse)
        for _, fine_label, coarse_label in train_ds.samples:
            fine_counts[fine_label] += 1
            coarse_counts[coarse_label] += 1

        self.loss_coarse = WeightedCrossEntropyLoss(coarse_counts, gamma=0.5)
        wce_fine = WeightedCrossEntropyLoss(fine_counts, gamma=0.5)
        focal_fine = ClassBalancedFocalLoss(fine_counts, gamma_w=0.5, gamma_f=2.0)

        def _combined_fine(logits, targets):
            return wce_fine(logits, targets) + focal_fine(logits, targets)

        self.loss_fine = _combined_fine
        self._stage2_prepared = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"stage 2 trainable params: {trainable}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        frames, fine_labels, coarse_labels = batch

        if self.stage is None:
            labels = coarse_labels if self.branch == "coarse" else fine_labels
            logits = self.model(frames)
            loss = F.cross_entropy(logits, labels)
            self.train_acc.update(logits, labels)
            self.log(
                "train/loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "train/acc",
                self.train_acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            return loss

        fine_logits, coarse_logits = self.model(frames)
        if self.stage == 1:
            loss = F.cross_entropy(
                fine_logits, fine_labels
            ) + self.args.lambda_coarse * F.cross_entropy(coarse_logits, coarse_labels)
        else:
            loss = self.loss_fine(
                fine_logits, fine_labels
            ) + self.args.lambda_coarse * self.loss_coarse(coarse_logits, coarse_labels)

        self.train_fine_acc.update(fine_logits, fine_labels)
        self.train_coarse_acc.update(coarse_logits, coarse_labels)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/fine_acc",
            self.train_fine_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/coarse_acc",
            self.train_coarse_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        frames, fine_labels, coarse_labels = batch

        if self.stage is None:
            labels = coarse_labels if self.branch == "coarse" else fine_labels
            logits = self.model(frames)
            loss = F.cross_entropy(logits, labels)
            self.val_acc.update(logits, labels)
            self.val_f1_macro.update(logits, labels)
            self.val_f1_micro.update(logits, labels)
            self.log(
                "val/loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            return

        fine_logits, coarse_logits = self.model(frames)
        p_fine = fine_logits.softmax(dim=-1)
        p_coarse = coarse_logits.softmax(dim=-1)
        p_fine_ref = ensemble_fine_with_coarse(
            p_fine, p_coarse, self.fine2coarse_tensor
        )
        pred_fine = p_fine_ref.argmax(dim=-1)
        pred_coarse = p_coarse.argmax(dim=-1)

        self.val_body_macro.update(pred_coarse, coarse_labels)
        self.val_body_micro.update(pred_coarse, coarse_labels)
        self.val_action_macro.update(pred_fine, fine_labels)
        self.val_action_micro.update(pred_fine, fine_labels)

    def on_validation_epoch_end(self):
        if self.stage is None:
            self.log("val/acc", self.val_acc.compute(), prog_bar=True, sync_dist=True)
            self.log(
                "val/f1_macro",
                self.val_f1_macro.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "val/f1_micro",
                self.val_f1_micro.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.val_acc.reset()
            self.val_f1_macro.reset()
            self.val_f1_micro.reset()
            return

        body_macro = self.val_body_macro.compute()
        body_micro = self.val_body_micro.compute()
        action_macro = self.val_action_macro.compute()
        action_micro = self.val_action_micro.compute()
        f1_mean = (body_macro + body_micro + action_macro + action_micro) / 4.0

        self.log("val/body_macro", body_macro, prog_bar=True, sync_dist=True)
        self.log("val/body_micro", body_micro, sync_dist=True)
        self.log("val/action_macro", action_macro, prog_bar=True, sync_dist=True)
        self.log("val/action_micro", action_micro, sync_dist=True)
        self.log("val/f1_mean", f1_mean, prog_bar=True, sync_dist=True)

        self.val_body_macro.reset()
        self.val_body_micro.reset()
        self.val_action_macro.reset()
        self.val_action_micro.reset()

    def configure_optimizers(self):
        train_loader = self.trainer.datamodule.train_dataloader()
        steps_per_epoch = max(len(train_loader), 1)

        if self.stage is None:
            params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(params, lr=self.args.lr, weight_decay=self.args.wd)
            scheduler = create_lr_scheduler(
                optimizer,
                num_step=steps_per_epoch,
                epochs=self.args.epochs,
                warmup=True,
                warmup_epochs=1,
            )
        elif self.stage == 1:
            param_groups = make_param_groups_stage1(
                self.model,
                lr_backbone=1e-5,
                lr_new=1e-4,
                lr_head=1e-3,
                wd=self.args.wd,
            )
            optimizer = optim.AdamW(param_groups)
            scheduler = create_lr_scheduler(
                optimizer,
                num_step=max(steps_per_epoch // max(self.args.grad_accum, 1), 1),
                epochs=self.args.epochs,
                warmup=True,
                warmup_epochs=2,
            )
        else:
            head_params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(head_params, lr=self.args.lr, weight_decay=0.0)
            scheduler = create_lr_scheduler(
                optimizer,
                num_step=steps_per_epoch,
                epochs=self.args.epochs,
                warmup=False,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def _resolve_accelerator(args):
    if torch.cuda.is_available() and args.gpus > 0 and args.device.startswith("cuda"):
        if args.gpus > 1:
            return "gpu", args.gpus
        idx = int(args.device.split(":")[1]) if ":" in args.device else 0
        return "gpu", [idx]
    return "cpu", 1


def main(args, cfg: Optional[DictConfig] = None):
    pl.seed_everything(args.seed, workers=True)

    fine2coarse_file = os.path.join(args.data_root, "annotations/fine2coarse.txt")
    if os.path.exists(fine2coarse_file):
        fine2coarse, coarse_names = load_fine2coarse_file(fine2coarse_file)
    else:
        fine2coarse = _DEFAULT_FINE2COARSE
        coarse_names = {i: str(i) for i in range(7)}

    num_coarse, num_fine = len(coarse_names), len(fine2coarse)
    datamodule = MA52DataModule(args, fine2coarse)
    model = MARLightningModule(
        args, num_coarse=num_coarse, num_fine=num_fine, fine2coarse=fine2coarse
    )

    if cfg is not None:
        out_dir = HydraConfig.get().runtime.output_dir
    else:
        run_name = (
            f"{datetime.now():%b%d_%H-%M-%S}_stage{args.stage}"
            if args.stage
            else f"{datetime.now():%b%d_%H-%M-%S}_{args.branch}"
        )
        out_dir = os.path.join(args.output, run_name)
        os.makedirs(out_dir, exist_ok=True)

    tb_logger = TensorBoardLogger(save_dir=out_dir, name="tb", version="")
    cvs_logger = CSVLogger(save_dir=out_dir, name="csv", version="")
    progress_bar = RichProgressBar(
        refresh_rate=0, leave=True
    )  # disable built-in progress bar

    monitor_key = "val/f1_micro" if args.stage is None else "val/f1_mean"
    ckpt = ModelCheckpoint(
        dirpath=out_dir,
        filename="best",
        monitor=monitor_key,
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    precision = args.precision
    accumulate = args.grad_accum if args.stage == 1 else 1
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        max_epochs=args.epochs,
        logger=[tb_logger, cvs_logger],
        callbacks=[ckpt, progress_bar],
        precision=precision,
        accumulate_grad_batches=accumulate,
        log_every_n_steps=10,
        default_root_dir=out_dir,
    )
    trainer.fit(model, datamodule=datamodule)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def hydra_entry(cfg: DictConfig):
    args = _build_args_from_cfg(cfg)
    main(args, cfg)


if __name__ == "__main__":
    hydra_entry()
