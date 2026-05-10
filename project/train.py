import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import (MA52Dataset, load_fine2coarse_file, load_label_names_file,
                        _DEFAULT_FINE2COARSE)
from video_model import build_dual_video_model
from utils import create_lr_scheduler, get_params_groups, train_one_epoch, evaluate, test_model


def resolve_paths(args):
    """Infer all data paths from --data-root, with per-arg overrides."""

    def pick(explicit, *inferred):
        """Return the first non-empty value."""
        if explicit:
            return explicit
        for v in inferred:
            if v:
                return v
        raise ValueError(f"Cannot resolve path: provide --data-root or explicit args")

    ann = os.path.join(args.data_root, 'annotations') if args.data_root else ''

    train_root = pick(args.train_root,
                      os.path.join(args.data_root, 'train') if args.data_root else '')
    val_root   = pick(args.val_root,
                      os.path.join(args.data_root, 'val')   if args.data_root else '')
    test_root  = pick(args.test_root,
                      os.path.join(args.data_root, 'test')  if args.data_root else '')

    train_ann        = pick(args.train_ann,
                            os.path.join(ann, 'train_list_videos.txt') if ann else '')
    val_ann          = pick(args.val_ann,
                            os.path.join(ann, 'val_list_videos.txt')   if ann else '')
    test_ann         = pick(args.test_ann,
                            os.path.join(ann, 'test_list_videos.txt')  if ann else '')
    fine2coarse_file = pick(args.fine2coarse_file,
                            os.path.join(ann, 'fine2coarse.txt')       if ann else '')
    label_name_file  = pick(args.label_name_file,
                            os.path.join(ann, 'label_name.txt')        if ann else '')

    return (train_root, val_root, test_root,
            train_ann, val_ann, test_ann,
            fine2coarse_file, label_name_file)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    os.makedirs("./weights", exist_ok=True)

    # ── resolve all data paths ────────────────────────────────────────────────
    (train_root, val_root, test_root,
     train_ann, val_ann, test_ann,
     fine2coarse_file, label_name_file) = resolve_paths(args)

    print(f"train videos : {train_root}")
    print(f"val   videos : {val_root}")
    print(f"test  videos : {test_root}")
    print(f"train ann    : {train_ann}")
    print(f"val   ann    : {val_ann}")
    print(f"test  ann    : {test_ann}")

    # ── label mappings ────────────────────────────────────────────────────────
    if os.path.exists(fine2coarse_file):
        fine2coarse, coarse_names = load_fine2coarse_file(fine2coarse_file)
    else:
        fine2coarse  = _DEFAULT_FINE2COARSE
        coarse_names = {0: 'body', 1: 'head', 2: 'upper limb', 3: 'lower limb',
                        4: 'body-hand', 5: 'head-hand', 6: 'leg-hand'}

    if os.path.exists(label_name_file):
        fine_names = load_label_names_file(label_name_file)
    else:
        fine_names = {i: str(i) for i in range(52)}

    # ── transforms ────────────────────────────────────────────────────────────
    img_size = 224
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize(int(img_size * 1.143)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    # ── datasets & loaders ────────────────────────────────────────────────────
    def make_dataset(ann_file, root, training):
        return MA52Dataset(
            ann_file=ann_file,
            root=root,
            num_frames=args.num_frames,
            transform=data_transform["train" if training else "val"],
            training=training,
            fine2coarse=fine2coarse,
        )


    train_dataset = make_dataset(train_ann, train_root, training=True)
    val_dataset   = make_dataset(val_ann,   val_root,   training=False)
    test_dataset  = make_dataset(test_ann,  test_root,  training=False)

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print(f"Using {nw} dataloader workers per process")

    def make_loader(dataset, shuffle):
        return torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=shuffle,
            pin_memory=True, num_workers=nw, collate_fn=MA52Dataset.collate_fn,
        )

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader   = make_loader(val_dataset,   shuffle=False)
    test_loader  = make_loader(test_dataset,  shuffle=False)

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_dual_video_model(
        model_name_1=args.model_name_1,
        weights_1=args.weights_1,
        model_name_2=args.model_name_2,
        weights_2=args.weights_2,
        num_fine=len(fine2coarse),
        num_coarse=len(coarse_names),
        d_state=args.mamba_d_state,
        n_layers=args.mamba_layers,
        dropout=args.mamba_dropout,
    ).to(device)

    # ── optimizer & scheduler ─────────────────────────────────────────────────
    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    # ── training loop ─────────────────────────────────────────────────────────
    tb_writer     = SummaryWriter()
    best_fine_acc = 0.0
    best_ckpt     = "./weights/best_model.pth"
    no_improve    = 0

    for epoch in range(args.epochs):
        train_loss, train_fine, train_coarse = train_one_epoch(
            model, optimizer, train_loader, device, epoch, lr_scheduler,
            coarse_weight=args.coarse_weight,
        )
        val_loss, val_fine, val_coarse = evaluate(
            model, val_loader, device, epoch,
            coarse_weight=args.coarse_weight,
        )

        tags = ["train_loss", "train_fine_acc", "train_coarse_acc",
                "val_loss",   "val_fine_acc",   "val_coarse_acc",  "lr"]
        vals  = [train_loss,   train_fine,        train_coarse,
                 val_loss,     val_fine,           val_coarse,
                 optimizer.param_groups[0]["lr"]]
        for tag, val in zip(tags, vals):
            tb_writer.add_scalar(tag, val, epoch)

        if val_fine > best_fine_acc:
            torch.save(model.state_dict(), best_ckpt)
            best_fine_acc = val_fine
            no_improve = 0
            print(f"  → saved best model (val fine acc {val_fine*100:.2f}%)")
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"  Early stopping: no improvement for {args.patience} epochs.")
                break

    # ── evaluate best checkpoint on val + test ────────────────────────────────
    print(f"\nLoading best checkpoint: {best_ckpt}")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    print("\n[Evaluating on VAL set]")
    test_model(model, val_loader, device,
               fine_names=fine_names, coarse_names=coarse_names,
               save_path="./val_results.txt")

    print("\n[Evaluating on TEST set]")
    test_model(model, test_loader, device,
               fine_names=fine_names, coarse_names=coarse_names,
               save_path="./test_results.txt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ── training ──────────────────────────────────────────────────────────────
    parser.add_argument('--epochs',           type=int,   default=50)
    parser.add_argument('--patience',         type=int,   default=10,
                        help='early stopping: stop if val fine acc does not improve '
                             'for this many epochs (0 = disabled)')
    parser.add_argument('--batch-size',       type=int,   default=32)
    parser.add_argument('--num-frames',       type=int,   default=8)
    parser.add_argument('--lr',               type=float, default=5e-4)
    parser.add_argument('--wd',               type=float, default=5e-2)
    parser.add_argument('--coarse-weight',    type=float, default=0.5)

    # ── data paths ────────────────────────────────────────────────────────────
    # Simplest: just --data-root, everything else is inferred automatically:
    #   videos  → <data-root>/train|val|test/
    #   ann     → <data-root>/annotations/train|val|test_list_videos.txt
    #   labels  → <data-root>/annotations/fine2coarse.txt + label_name.txt
    parser.add_argument('--data-root',        type=str, default='',
                        help='root of MA52 dataset (e.g. /mnt/d/MA52); '
                             'infers all sub-paths automatically')
    # per-path overrides (optional, take priority over --data-root inference)
    parser.add_argument('--train-root',       type=str, default='')
    parser.add_argument('--val-root',         type=str, default='')
    parser.add_argument('--test-root',        type=str, default='')
    parser.add_argument('--train-ann',        type=str, default='')
    parser.add_argument('--val-ann',          type=str, default='')
    parser.add_argument('--test-ann',         type=str, default='')
    parser.add_argument('--fine2coarse-file', type=str, default='')
    parser.add_argument('--label-name-file',  type=str, default='')

    # ── model ─────────────────────────────────────────────────────────────────
    parser.add_argument('--model-name-1',     type=str, default='dinov3_convnext_tiny',
                        choices=['dinov3_convnext_tiny', 'dinov3_convnext_small',
                                 'dinov3_convnext_base', 'dinov3_convnext_large'],
                        help='branch 1 → fine head (52 classes)')
    parser.add_argument('--weights-1',        type=str,
                        default='./dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth')
    parser.add_argument('--model-name-2',     type=str, default='dinov3_convnext_tiny',
                        choices=['dinov3_convnext_tiny', 'dinov3_convnext_small',
                                 'dinov3_convnext_base', 'dinov3_convnext_large'],
                        help='branch 2 → coarse head (7 classes)')
    parser.add_argument('--weights-2',        type=str,
                        default='./dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth')

    # ── mamba ─────────────────────────────────────────────────────────────────
    parser.add_argument('--mamba-d-state',    type=int,   default=64)
    parser.add_argument('--mamba-layers',     type=int,   default=1)
    parser.add_argument('--mamba-dropout',    type=float, default=0.0)

    parser.add_argument('--device',           default='cuda:0')

    opt = parser.parse_args()
    main(opt)
