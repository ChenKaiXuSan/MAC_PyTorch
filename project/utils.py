import os
import sys
import json
import pickle
import random
import math

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler,
                    coarse_weight: float = 0.5):
    model.train()
    ce = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_fine = torch.zeros(1).to(device)
    accu_coarse = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        frames, fine_labels, coarse_labels = data
        frames = frames.to(device)
        fine_labels = fine_labels.to(device)
        coarse_labels = coarse_labels.to(device)
        sample_num += frames.shape[0]

        fine_pred, coarse_pred = model(frames)

        loss = ce(fine_pred, fine_labels) + coarse_weight * ce(coarse_pred, coarse_labels)
        loss.backward()
        accu_loss += loss.detach()

        accu_fine += torch.eq(fine_pred.argmax(dim=1), fine_labels).sum()
        accu_coarse += torch.eq(coarse_pred.argmax(dim=1), coarse_labels).sum()

        data_loader.desc = (
            "[train epoch {}] loss: {:.3f}, fine_acc: {:.3f}, coarse_acc: {:.3f}, lr: {:.5f}"
            .format(epoch,
                    accu_loss.item() / (step + 1),
                    accu_fine.item() / sample_num,
                    accu_coarse.item() / sample_num,
                    optimizer.param_groups[0]["lr"])
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

    n = step + 1
    return (accu_loss.item() / n,
            accu_fine.item() / sample_num,
            accu_coarse.item() / sample_num)


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, coarse_weight: float = 0.5):
    ce = torch.nn.CrossEntropyLoss()
    model.eval()

    accu_loss = torch.zeros(1).to(device)
    accu_fine = torch.zeros(1).to(device)
    accu_coarse = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        frames, fine_labels, coarse_labels = data
        frames = frames.to(device)
        fine_labels = fine_labels.to(device)
        coarse_labels = coarse_labels.to(device)
        sample_num += frames.shape[0]

        fine_pred, coarse_pred = model(frames)

        loss = ce(fine_pred, fine_labels) + coarse_weight * ce(coarse_pred, coarse_labels)
        accu_loss += loss

        accu_fine += torch.eq(fine_pred.argmax(dim=1), fine_labels).sum()
        accu_coarse += torch.eq(coarse_pred.argmax(dim=1), coarse_labels).sum()

        data_loader.desc = (
            "[valid epoch {}] loss: {:.3f}, fine_acc: {:.3f}, coarse_acc: {:.3f}"
            .format(epoch,
                    accu_loss.item() / (step + 1),
                    accu_fine.item() / sample_num,
                    accu_coarse.item() / sample_num)
        )

    n = step + 1
    return (accu_loss.item() / n,
            accu_fine.item() / sample_num,
            accu_coarse.item() / sample_num)


@torch.no_grad()
def test_model(model, data_loader, device, fine_names: dict, coarse_names: dict,
               save_path: str = './test_results.txt'):
    """Evaluate on test set. Reports accuracy + F1 macro/micro for fine and coarse heads."""
    from sklearn.metrics import f1_score

    model.eval()

    all_fine_pred, all_fine_gt     = [], []
    all_coarse_pred, all_coarse_gt = [], []

    num_fine   = max(fine_names.keys()) + 1
    num_coarse = max(coarse_names.keys()) + 1
    fine_correct   = torch.zeros(num_fine)
    fine_total     = torch.zeros(num_fine)
    coarse_correct = torch.zeros(num_coarse)
    coarse_total   = torch.zeros(num_coarse)

    for frames, fine_labels, coarse_labels in tqdm(data_loader, file=sys.stdout, desc='[test]'):
        frames        = frames.to(device)
        fine_labels   = fine_labels.to(device)
        coarse_labels = coarse_labels.to(device)

        fine_pred, coarse_pred = model(frames)
        fine_cls   = fine_pred.argmax(dim=1)
        coarse_cls = coarse_pred.argmax(dim=1)

        all_fine_pred.extend(fine_cls.cpu().tolist())
        all_fine_gt.extend(fine_labels.cpu().tolist())
        all_coarse_pred.extend(coarse_cls.cpu().tolist())
        all_coarse_gt.extend(coarse_labels.cpu().tolist())

        for c in range(num_fine):
            mask = fine_labels == c
            fine_total[c]   += mask.sum().item()
            fine_correct[c] += (fine_cls[mask] == c).sum().item()

        for c in range(num_coarse):
            mask = coarse_labels == c
            coarse_total[c]   += mask.sum().item()
            coarse_correct[c] += (coarse_cls[mask] == c).sum().item()

    # ── overall metrics ───────────────────────────────────────────────────────
    fine_acc   = fine_correct.sum()   / fine_total.sum()
    coarse_acc = coarse_correct.sum() / coarse_total.sum()

    fine_f1_macro   = f1_score(all_fine_gt,   all_fine_pred,   average='macro',  zero_division=0)
    fine_f1_micro   = f1_score(all_fine_gt,   all_fine_pred,   average='micro',  zero_division=0)
    coarse_f1_macro = f1_score(all_coarse_gt, all_coarse_pred, average='macro',  zero_division=0)
    coarse_f1_micro = f1_score(all_coarse_gt, all_coarse_pred, average='micro',  zero_division=0)

    # ── report ────────────────────────────────────────────────────────────────
    sep = '=' * 62
    lines = [
        sep, 'Test Results', sep,
        f'Fine   (52 classes)  acc: {fine_acc*100:6.2f}%  '
        f'F1-macro: {fine_f1_macro*100:6.2f}%  F1-micro: {fine_f1_micro*100:6.2f}%',
        f'Coarse ( 7 classes)  acc: {coarse_acc*100:6.2f}%  '
        f'F1-macro: {coarse_f1_macro*100:6.2f}%  F1-micro: {coarse_f1_micro*100:6.2f}%',
        '',
        '--- Per Fine Class ---',
    ]
    for c in range(num_fine):
        n, tot = int(fine_correct[c]), int(fine_total[c])
        acc = n / tot * 100 if tot > 0 else 0.0
        lines.append(f'  {c:2d}  {fine_names.get(c, str(c)):<42s} {acc:6.2f}%  ({n}/{tot})')

    lines.append('\n--- Per Coarse Class ---')
    for c in range(num_coarse):
        n, tot = int(coarse_correct[c]), int(coarse_total[c])
        acc = n / tot * 100 if tot > 0 else 0.0
        lines.append(f'  {c}  {coarse_names.get(c, str(c)):<22s} {acc:6.2f}%  ({n}/{tot})')

    report = '\n'.join(lines)
    print(report)
    with open(save_path, 'w') as f:
        f.write(report + '\n')
    print(f'\nSaved → {save_path}')

    return {
        'fine_acc':        fine_acc.item(),
        'fine_f1_macro':   fine_f1_macro,
        'fine_f1_micro':   fine_f1_micro,
        'coarse_acc':      coarse_acc.item(),
        'coarse_f1_macro': coarse_f1_macro,
        'coarse_f1_micro': coarse_f1_micro,
    }


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


import torch.nn.functional as F
from sklearn.metrics import f1_score


def train_one_epoch_single(model, optimizer, loader, device, epoch, scheduler,
                            label_key: str = 'coarse', amp_dtype=torch.bfloat16):
    """Train one epoch on a SINGLE branch (coarse XOR fine).

    label_key: 'coarse' uses coarse_labels; 'fine' uses fine_labels.
    """
    model.train()
    accu_loss = 0.0
    accu_correct = 0
    n_seen = 0
    scaler_dtype = amp_dtype

    optimizer.zero_grad()
    loader = tqdm(loader, file=sys.stdout)
    for step, (frames, fine_labels, coarse_labels) in enumerate(loader):
        frames = frames.to(device)
        labels = (coarse_labels if label_key == 'coarse' else fine_labels).to(device)
        with torch.autocast(device_type='cuda', dtype=scaler_dtype):
            logits = model(frames)
            loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        accu_loss += loss.item()
        accu_correct += (logits.argmax(-1) == labels).sum().item()
        n_seen += labels.size(0)
        loader.desc = (f'[train {label_key} ep {epoch}] '
                       f'loss={accu_loss/(step+1):.3f} acc={accu_correct/n_seen:.3f} '
                       f'lr={optimizer.param_groups[0]["lr"]:.2e}')
    return accu_loss / (step + 1), accu_correct / n_seen


@torch.no_grad()
def evaluate_single(model, loader, device, label_key: str = 'coarse', amp_dtype=torch.bfloat16):
    """Evaluate single branch. Returns dict with f1_macro, f1_micro, acc."""
    model.eval()
    all_pred, all_gt = [], []
    loader = tqdm(loader, file=sys.stdout)
    for step, (frames, fine_labels, coarse_labels) in enumerate(loader):
        frames = frames.to(device)
        labels = (coarse_labels if label_key == 'coarse' else fine_labels).to(device)
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            logits = model(frames)
        all_pred.extend(logits.argmax(-1).cpu().tolist())
        all_gt.extend(labels.cpu().tolist())

    f1_macro = f1_score(all_gt, all_pred, average='macro', zero_division=0)
    f1_micro = f1_score(all_gt, all_pred, average='micro', zero_division=0)
    acc = sum(p == g for p, g in zip(all_pred, all_gt)) / len(all_gt)
    return {'f1_macro': f1_macro, 'f1_micro': f1_micro, 'acc': acc}


def ensemble_fine_with_coarse(p_fine: torch.Tensor, p_coarse: torch.Tensor,
                               fine2coarse: torch.Tensor) -> torch.Tensor:
    """Soft ensembling: p_fine *= p_coarse[fine2coarse]; renormalize.

    p_fine:    (B, 52) softmaxed
    p_coarse:  (B, 7)  softmaxed
    fine2coarse: (52,) long
    Returns:   (B, 52)
    """
    coarse_w = p_coarse[:, fine2coarse]  # (B, 52)
    refined = p_fine * coarse_w
    return refined / refined.sum(dim=-1, keepdim=True).clamp(min=1e-8)


@torch.no_grad()
def evaluate_dual(model, loader, device, fine2coarse: list,
                   amp_dtype=torch.bfloat16):
    """Evaluate dual-branch model. Returns dict with 4 F1s + F1_mean.

    F1_body_* on coarse predictions, F1_action_* on ensembling-refined fine predictions.
    """
    model.eval()
    fc_tensor = torch.as_tensor(fine2coarse, dtype=torch.long, device=device)

    all_fine, all_coarse, all_fine_gt, all_coarse_gt = [], [], [], []
    loader = tqdm(loader, file=sys.stdout)
    for frames, fine_labels, coarse_labels in loader:
        frames = frames.to(device)
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            fine_logits, coarse_logits = model(frames)
        p_fine = fine_logits.softmax(-1)
        p_coarse = coarse_logits.softmax(-1)
        p_fine_ref = ensemble_fine_with_coarse(p_fine, p_coarse, fc_tensor)
        all_fine.extend(p_fine_ref.argmax(-1).cpu().tolist())
        all_coarse.extend(p_coarse.argmax(-1).cpu().tolist())
        all_fine_gt.extend(fine_labels.tolist())
        all_coarse_gt.extend(coarse_labels.tolist())

    f = {
        'body_macro':   f1_score(all_coarse_gt, all_coarse, average='macro', zero_division=0),
        'body_micro':   f1_score(all_coarse_gt, all_coarse, average='micro', zero_division=0),
        'action_macro': f1_score(all_fine_gt,   all_fine,   average='macro', zero_division=0),
        'action_micro': f1_score(all_fine_gt,   all_fine,   average='micro', zero_division=0),
    }
    f['f1_mean'] = sum(f.values()) / 4.0
    return f


def train_one_epoch_dual(model, optimizer, loader, device, epoch, scheduler,
                          grad_accum_steps: int = 1, amp_dtype=torch.bfloat16,
                          lambda_coarse: float = 1.0):
    """Stage 1 joint training: CE on both heads."""
    model.train()
    accu_loss = 0.0
    accu_fine_correct = 0
    accu_coarse_correct = 0
    n_seen = 0
    optimizer.zero_grad()
    loader_t = tqdm(loader, file=sys.stdout)
    for step, (frames, fine_labels, coarse_labels) in enumerate(loader_t):
        frames = frames.to(device)
        fine_labels = fine_labels.to(device)
        coarse_labels = coarse_labels.to(device)
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            fine_logits, coarse_logits = model(frames)
            loss = (F.cross_entropy(fine_logits, fine_labels)
                    + lambda_coarse * F.cross_entropy(coarse_logits, coarse_labels))
            loss = loss / grad_accum_steps
        loss.backward()
        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        accu_loss += loss.item() * grad_accum_steps
        accu_fine_correct += (fine_logits.argmax(-1) == fine_labels).sum().item()
        accu_coarse_correct += (coarse_logits.argmax(-1) == coarse_labels).sum().item()
        n_seen += fine_labels.size(0)
        loader_t.desc = (f'[s1 ep{epoch}] loss={accu_loss/(step+1):.3f} '
                          f'fine={accu_fine_correct/n_seen:.3f} '
                          f'coarse={accu_coarse_correct/n_seen:.3f}')
    return {
        'loss': accu_loss / (step + 1),
        'fine_acc': accu_fine_correct / n_seen,
        'coarse_acc': accu_coarse_correct / n_seen,
    }


def make_param_groups_stage1(model, lr_backbone=1e-5, lr_new=1e-4, lr_head=1e-3,
                              wd=0.05):
    """Stage 1 optimizer parameter groups.

    Buckets:
      'backbone' — InternVideo2 ViT params that existed before TC patching
      'tc'       — new params introduced by TCMHSA (summary_ffn, etc.)
      'adapter'  — 3D-ResNet adapter params (coarse branch)
      'head'     — classification heads + their norms
    """
    backbone, tc, adapter, head, frozen = [], [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            frozen.append(n)
            continue
        if 'adapters' in n:
            adapter.append(p)
        elif 'summary_ffn' in n or n.endswith('_tc_state'):
            tc.append(p)
        elif n.startswith('coarse.head') or n.startswith('coarse.norm') \
             or n.startswith('fine.head') or n.startswith('fine.norm'):
            head.append(p)
        elif n.startswith('fine.intv2') or n.startswith('fine.head_attn') or n.startswith('fine.head_norm'):
            # head_attn and head_norm are part of fine_head; route to head group
            if 'head_attn' in n or 'head_norm' in n or 'head_fc' in n:
                head.append(p)
            else:
                backbone.append(p)
        else:
            backbone.append(p)
    print(f'param groups: backbone={len(backbone)} tc={len(tc)} adapter={len(adapter)} head={len(head)} frozen={len(frozen)}')
    return [
        {'params': backbone, 'lr': lr_backbone, 'weight_decay': wd},
        {'params': tc,       'lr': lr_new,      'weight_decay': wd},
        {'params': adapter,  'lr': lr_new,      'weight_decay': wd},
        {'params': head,     'lr': lr_head,     'weight_decay': 0.0},
    ]


def train_one_epoch_stage2(model, optimizer, loader, device, epoch, scheduler,
                            loss_fine, loss_coarse, amp_dtype=torch.bfloat16,
                            lambda_coarse: float = 1.0):
    """Stage 2: heads only, custom losses (WCE+Focal), backbone+TC+adapter frozen."""
    model.train()
    # Backbone is frozen but BatchNorm/dropout layers should stay in eval mode.
    # We rely on _freeze() having set .eval() previously. Heads will still update.

    accu_loss = 0.0
    n = 0
    loader_t = tqdm(loader, file=sys.stdout)
    optimizer.zero_grad()
    for step, (frames, fine_labels, coarse_labels) in enumerate(loader_t):
        frames = frames.to(device)
        fine_labels = fine_labels.to(device)
        coarse_labels = coarse_labels.to(device)
        with torch.autocast(device_type='cuda', dtype=amp_dtype):
            fine_logits, coarse_logits = model(frames)
            loss = loss_fine(fine_logits, fine_labels) + lambda_coarse * loss_coarse(coarse_logits, coarse_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        accu_loss += loss.item()
        n += 1
        loader_t.desc = f'[s2 ep{epoch}] loss={accu_loss/n:.3f}'
    return accu_loss / n
