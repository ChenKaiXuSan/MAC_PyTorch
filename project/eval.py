"""Test-set inference with ensembling. Produces a JSON of fine predictions."""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from my_dataset import MA52Dataset, load_fine2coarse_file, _DEFAULT_FINE2COARSE
from utils import ensemble_fine_with_coarse
from video_model import DualBranchVideo


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    fine2coarse_file = os.path.join(args.data_root, 'annotations/fine2coarse.txt')
    if os.path.exists(fine2coarse_file):
        fine2coarse, coarse_names = load_fine2coarse_file(fine2coarse_file)
    else:
        fine2coarse = _DEFAULT_FINE2COARSE
        coarse_names = {i: str(i) for i in range(7)}
    num_coarse, num_fine = len(coarse_names), len(fine2coarse)
    fc_tensor = torch.as_tensor(fine2coarse, dtype=torch.long, device=device)

    tfm = transforms.Compose([
        transforms.Resize(int(224 * 1.143)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = MA52Dataset(ann_file=args.test_ann, root=args.test_root,
                     num_frames=args.num_frames, transform=tfm, training=False,
                     fine2coarse=fine2coarse)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True,
                        collate_fn=MA52Dataset.collate_fn)

    model = DualBranchVideo(vmae_path=args.vmae_path, intv2_path=args.intv2_path,
                             num_coarse=num_coarse, num_fine=num_fine,
                             num_frames=args.num_frames).to(device)
    sd = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(sd, strict=False)
    model.eval()

    predictions = []  # list of {'video': ..., 'fine_pred': ..., 'coarse_pred': ...}
    sample_iter = iter(ds.samples)
    with torch.no_grad():
        for frames, _, _ in tqdm(loader):
            frames = frames.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                fine_logits, coarse_logits = model(frames)
            p_fine = fine_logits.softmax(-1)
            p_coarse = coarse_logits.softmax(-1)
            p_fine_ref = ensemble_fine_with_coarse(p_fine, p_coarse, fc_tensor)
            fine_pred = p_fine_ref.argmax(-1).cpu().tolist()
            coarse_pred = p_coarse.argmax(-1).cpu().tolist()
            for fp, cp in zip(fine_pred, coarse_pred):
                video_path, _, _ = next(sample_iter)
                predictions.append({
                    'video': os.path.basename(video_path),
                    'fine_pred': int(fp),
                    'coarse_pred': int(cp),
                })

    with open(args.output, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f'wrote {len(predictions)} predictions to {args.output}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data-root', type=str, default='/mnt/d/MA52')
    p.add_argument('--test-root', type=str, default='/mnt/d/MA52/test')
    p.add_argument('--test-ann', type=str,
                   default='/mnt/d/MA52/annotations/test_list_videos.txt')
    p.add_argument('--vmae-path', type=str, default='OpenGVLab/VideoMAEv2-Large')
    p.add_argument('--intv2-path', type=str, default='OpenGVLab/InternVideo2-Stage1-L14')
    p.add_argument('--num-frames', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--output', type=str, default='test_predictions.json')
    p.add_argument('--device', type=str, default='cuda:0')
    main(p.parse_args())
