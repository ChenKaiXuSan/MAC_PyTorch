"""Evaluate a saved checkpoint on val and/or test set."""
import os
import argparse

import torch
from torchvision import transforms

from my_dataset import (MA52Dataset, load_fine2coarse_file, load_label_names_file,
                        _DEFAULT_FINE2COARSE)
from video_model import build_dual_video_model
from utils import test_model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── label mappings ────────────────────────────────────────────────────────
    if args.fine2coarse_file and os.path.exists(args.fine2coarse_file):
        fine2coarse, coarse_names = load_fine2coarse_file(args.fine2coarse_file)
    else:
        fine2coarse  = _DEFAULT_FINE2COARSE
        coarse_names = {0:'body', 1:'head', 2:'upper limb', 3:'lower limb',
                        4:'body-hand', 5:'head-hand', 6:'leg-hand'}

    if args.label_name_file and os.path.exists(args.label_name_file):
        fine_names = load_label_names_file(args.label_name_file)
    else:
        fine_names = {i: str(i) for i in range(52)}

    # ── transform (val/test) ──────────────────────────────────────────────────
    img_size = 224
    transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.143)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_dual_video_model(
        model_name_1=args.model_name_1,
        weights_1=args.weights_1,
        vmae_path=args.vmae_path,
        num_fine=len(fine2coarse),
        num_coarse=len(coarse_names),
    ).to(device)

    assert os.path.exists(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint: {args.checkpoint}")

    def run(ann_file, root, label, save_path):
        if not ann_file or not os.path.exists(ann_file):
            print(f"Skipping {label}: annotation file not found ({ann_file})")
            return
        ds = MA52Dataset(ann_file, root=root, num_frames=args.num_frames,
                         transform=transform, training=False, fine2coarse=fine2coarse)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=MA52Dataset.collate_fn,
        )
        print(f"\n[{label}]")
        test_model(model, loader, device,
                   fine_names=fine_names, coarse_names=coarse_names,
                   save_path=save_path)

    data_root = args.data_root
    ann_dir   = os.path.join(data_root, 'annotations') if data_root else ''

    def resolve(explicit, inferred):
        return explicit if explicit else inferred

    run(
        ann_file  = resolve(args.val_ann,  os.path.join(ann_dir, 'val_list_videos.txt')),
        root      = resolve(args.val_root, os.path.join(data_root, 'val') if data_root else ''),
        label     = 'VAL',
        save_path = './val_results.txt',
    )
    run(
        ann_file  = resolve(args.test_ann,  os.path.join(ann_dir, 'test_list_videos.txt')),
        root      = resolve(args.test_root, os.path.join(data_root, 'test') if data_root else ''),
        label     = 'TEST',
        save_path = './test_results.txt',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',       type=str, default='./weights/best_model.pth')
    parser.add_argument('--num-frames',       type=int, default=16)
    parser.add_argument('--batch-size',       type=int, default=4)
    parser.add_argument('--num-workers',      type=int, default=4)

    parser.add_argument('--data-root',        type=str, default='')
    parser.add_argument('--val-root',         type=str, default='')
    parser.add_argument('--val-ann',          type=str, default='')
    parser.add_argument('--test-root',        type=str, default='')
    parser.add_argument('--test-ann',         type=str, default='')
    parser.add_argument('--fine2coarse-file', type=str, default='')
    parser.add_argument('--label-name-file',  type=str, default='')

    parser.add_argument('--model-name-1',     type=str, default='dinov3_convnext_tiny')
    parser.add_argument('--weights-1',        type=str,
                        default='./dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth')
    parser.add_argument('--vmae-path',        type=str, default='OpenGVLab/VideoMAEv2-Base',
                        help='VideoMAEv2 HuggingFace model ID or local directory')

    parser.add_argument('--device',           default='cuda:0')

    opt = parser.parse_args()
    main(opt)
