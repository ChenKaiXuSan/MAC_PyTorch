"""MAR-paper reproduction training script.

Modes (added across milestones):
  M1: --branch coarse  (single coarse branch)
  M2: --branch fine    (single fine branch)
  M3: --stage 1        (dual branch joint CE training)
  M4: --stage 2        (frozen backbone, re-init heads, WCE+Focal on rare-resampled data)
"""

import argparse
import os
import sys
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MA52Dataset, load_fine2coarse_file, _DEFAULT_FINE2COARSE
from utils import create_lr_scheduler, train_one_epoch_single, evaluate_single


def make_transforms(img_size=224):
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(int(img_size * 1.143)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }


def build_dataloaders(args, fine2coarse):
    tfm = make_transforms()
    def ds(ann, root, training):
        return MA52Dataset(ann_file=ann, root=root, num_frames=args.num_frames,
                           transform=tfm['train' if training else 'val'],
                           training=training, fine2coarse=fine2coarse)
    train_ds = ds(os.path.join(args.data_root, 'annotations/train_list_videos.txt'),
                   os.path.join(args.data_root, 'train'), True)
    val_ds   = ds(os.path.join(args.data_root, 'annotations/val_list_videos.txt'),
                   os.path.join(args.data_root, 'val'), False)
    nw = min(os.cpu_count() or 4, 8)
    def dl(d, shuffle):
        return torch.utils.data.DataLoader(d, batch_size=args.batch_size, shuffle=shuffle,
                                            pin_memory=True, num_workers=nw,
                                            collate_fn=MA52Dataset.collate_fn)
    return dl(train_ds, True), dl(val_ds, False)


def build_model_single(args, num_coarse, num_fine):
    if args.branch == 'coarse':
        from video_model import CoarseBranch
        return CoarseBranch(vmae_path=args.vmae_path, num_coarse=num_coarse,
                            num_frames=args.num_frames)
    if args.branch == 'fine':
        from video_model import FineBranch  # added in M2
        return FineBranch(intv2_path=args.intv2_path, num_fine=num_fine,
                          num_frames=args.num_frames)
    raise ValueError(f"unknown branch {args.branch}")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  stage={args.stage}  branch={args.branch}')

    fine2coarse_file = os.path.join(args.data_root, 'annotations/fine2coarse.txt')
    if os.path.exists(fine2coarse_file):
        fine2coarse, coarse_names = load_fine2coarse_file(fine2coarse_file)
    else:
        fine2coarse = _DEFAULT_FINE2COARSE
        coarse_names = {i: str(i) for i in range(7)}
    num_coarse, num_fine = len(coarse_names), len(fine2coarse)

    train_loader, val_loader = build_dataloaders(args, fine2coarse)

    run_name = f"{datetime.now():%b%d_%H-%M-%S}_stage{args.stage}" if args.stage else f"{datetime.now():%b%d_%H-%M-%S}_{args.branch}"
    out_dir = os.path.join(args.output, run_name)
    os.makedirs(out_dir, exist_ok=True)
    tb = SummaryWriter(out_dir)

    if args.stage is None:
        # M1/M2 single-branch path
        model = build_model_single(args, num_coarse, num_fine).to(device)
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
        scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                         warmup=True, warmup_epochs=1)
        best_f1 = 0.0
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch_single(
                model, optimizer, train_loader, device, epoch, scheduler,
                label_key=args.branch)
            val = evaluate_single(model, val_loader, device, label_key=args.branch)
            print(f"[val ep {epoch}] {val}")
            tb.add_scalar('train/loss', train_loss, epoch)
            for k, v in val.items():
                tb.add_scalar(f'val/{k}', v, epoch)
            if val['f1_micro'] > best_f1:
                best_f1 = val['f1_micro']
                torch.save(model.state_dict(), os.path.join(out_dir, 'best.pth'))
        torch.save(model.state_dict(), os.path.join(out_dir, 'last.pth'))
        return

    # ── Stage 1 / Stage 2 path ────────────────────────────────────────────────
    from video_model import DualBranchVideo
    model = DualBranchVideo(vmae_path=args.vmae_path, intv2_path=args.intv2_path,
                             num_coarse=num_coarse, num_fine=num_fine,
                             num_frames=args.num_frames).to(device)

    if args.stage == 1:
        from utils import make_param_groups_stage1, train_one_epoch_dual, evaluate_dual
        pg = make_param_groups_stage1(model, lr_backbone=1e-5, lr_new=1e-4,
                                       lr_head=1e-3, wd=args.wd)
        optimizer = optim.AdamW(pg)
        scheduler = create_lr_scheduler(optimizer, len(train_loader) // args.grad_accum,
                                         args.epochs, warmup=True, warmup_epochs=2)
        best_f1 = 0.0
        for epoch in range(args.epochs):
            tr = train_one_epoch_dual(model, optimizer, train_loader, device, epoch,
                                       scheduler, grad_accum_steps=args.grad_accum,
                                       lambda_coarse=args.lambda_coarse)
            val = evaluate_dual(model, val_loader, device, fine2coarse=fine2coarse)
            print(f"[s1 val ep{epoch}] {val}")
            for k, v in tr.items():
                tb.add_scalar(f'train/{k}', v, epoch)
            for k, v in val.items():
                tb.add_scalar(f'val/{k}', v, epoch)
            if val['f1_mean'] > best_f1:
                best_f1 = val['f1_mean']
                torch.save(model.state_dict(), os.path.join(out_dir, 'best.pth'))
                print(f"  → new best f1_mean={best_f1:.4f}")
        torch.save(model.state_dict(), os.path.join(out_dir, 'last.pth'))
        return

    if args.stage == 2:
        from utils import train_one_epoch_stage2, evaluate_dual
        from my_dataset import make_rare_resampler
        from losses import WeightedCrossEntropyLoss, ClassBalancedFocalLoss

        # 1. Load Stage 1 checkpoint
        assert args.stage1_ckpt and os.path.exists(args.stage1_ckpt), \
            f'--stage1-ckpt is required for stage 2: {args.stage1_ckpt}'
        sd = torch.load(args.stage1_ckpt, map_location='cpu')
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'loaded stage1 ckpt; missing={len(missing)} unexpected={len(unexpected)}')

        # 2. Freeze everything
        model.freeze_for_stage2()

        # 3. Re-initialize and unfreeze BOTH heads in full.
        #    coarse head = norm + head
        #    fine   head = head_norm + head_attn + norm + head_fc  (per FineBranch in Task 9)
        D_coarse = model.coarse.dim
        D_fine = model.fine.dim
        model.coarse.norm = torch.nn.LayerNorm(D_coarse).to(device)
        model.coarse.head = torch.nn.Linear(D_coarse, num_coarse).to(device)

        model.fine.head_norm = torch.nn.LayerNorm(D_fine).to(device)
        model.fine.head_attn = torch.nn.MultiheadAttention(D_fine, num_heads=8,
                                                           batch_first=True).to(device)
        model.fine.norm = torch.nn.LayerNorm(D_fine).to(device)
        model.fine.head_fc = torch.nn.Linear(D_fine, num_fine).to(device)

        head_prefixes = ('coarse.norm', 'coarse.head',
                         'fine.head_norm', 'fine.head_attn', 'fine.norm', 'fine.head_fc')
        for n_, p in model.named_parameters():
            if n_.startswith(head_prefixes):
                p.requires_grad_(True)
        print('stage 2 trainable params:',
              sum(p.numel() for p in model.parameters() if p.requires_grad))

        # 4. Build rare-class resampled loader
        train_ds = train_loader.dataset
        sampler = make_rare_resampler(train_ds, num_fine=num_fine)
        train_loader_s2 = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=train_loader.num_workers, pin_memory=True,
            collate_fn=MA52Dataset.collate_fn)

        # 5. Compute class counts and build losses
        fine_counts = torch.zeros(num_fine)
        coarse_counts = torch.zeros(num_coarse)
        for _, fl, cl in train_ds.samples:
            fine_counts[fl] += 1
            coarse_counts[cl] += 1
        loss_coarse_obj = WeightedCrossEntropyLoss(coarse_counts.to(device), gamma=0.5)
        wce_fine = WeightedCrossEntropyLoss(fine_counts.to(device), gamma=0.5)
        focal_fine = ClassBalancedFocalLoss(fine_counts.to(device), gamma_w=0.5, gamma_f=2.0)

        def combined_fine_loss(logits, targets):
            return wce_fine(logits, targets) + focal_fine(logits, targets)

        # 6. Optimizer (heads only)
        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(head_params, lr=args.lr, weight_decay=0.0)
        scheduler = create_lr_scheduler(optimizer, len(train_loader_s2), args.epochs,
                                         warmup=False)

        # 7. Train
        best_f1 = 0.0
        for epoch in range(args.epochs):
            tr_loss = train_one_epoch_stage2(
                model, optimizer, train_loader_s2, device, epoch, scheduler,
                loss_fine=combined_fine_loss, loss_coarse=loss_coarse_obj,
                lambda_coarse=args.lambda_coarse)
            val = evaluate_dual(model, val_loader, device, fine2coarse=fine2coarse)
            print(f"[s2 val ep{epoch}] {val}")
            tb.add_scalar('train/loss', tr_loss, epoch)
            for k, v in val.items():
                tb.add_scalar(f'val/{k}', v, epoch)
            if val['f1_mean'] > best_f1:
                best_f1 = val['f1_mean']
                torch.save(model.state_dict(), os.path.join(out_dir, 'best.pth'))
        torch.save(model.state_dict(), os.path.join(out_dir, 'last.pth'))
        return


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--branch', choices=['coarse', 'fine'], default='coarse')
    p.add_argument('--stage', type=int, choices=[1, 2], default=None,
                   help='dual-branch stage (added in M3/M4)')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-frames', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=5e-2)
    p.add_argument('--data-root', type=str, default='/mnt/d/MA52')
    p.add_argument('--vmae-path', type=str, default='OpenGVLab/VideoMAEv2-Large')
    p.add_argument('--intv2-path', type=str, default='OpenGVLab/InternVideo2-Stage1-L14')
    p.add_argument('--output', type=str, default='runs')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--lambda-coarse', type=float, default=1.0)
    p.add_argument('--stage1-ckpt', type=str, default='', help='M4: path to Stage 1 checkpoint')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
