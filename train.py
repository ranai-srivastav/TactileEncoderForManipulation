"""
PoseIt training script.

Supports all three split modes (examples):
# by object
python train.py --split object --test_objects mug bowl --sigma 1.0

# by pose
python train.py --split pose --test_poses 1 2 3 4 5 --sigma 1.0

# random
python train.py --split random --anneal_iter 300 --n_iters 600
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import PoseItDataset, split_by_object, split_by_pose, uniform_random_split
from sampler import DRSSampler
from model import BaselineTactileEncoder

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir',     default='./data')
    p.add_argument('--split',        default='object', choices=['object', 'pose', 'random'])
    p.add_argument('--test_objects', nargs='+', default=['mug', 'bowl'])
    p.add_argument('--test_poses',   nargs='+', type=int, default=[1, 2, 3, 4, 5])
    p.add_argument('--sigma',        type=float, default=1.0)   # 0.5 for T; 1.0 for V or V+T
    p.add_argument('--batch_size',   type=int,   default=200)
    p.add_argument('--lr',           type=float, default=0.01)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--dropout',      type=float, default=0.1)
    p.add_argument('--hidden_dim',   type=int,   default=500)
    p.add_argument('--model_emb',    type=int,   default=512)
    p.add_argument('--n_iters',      type=int,   default=600)
    p.add_argument('--anneal_iter',  type=int,   default=300)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--modalities',   nargs='+', default=['V', 'T', 'FT', 'GF', 'G'],
                   help='Space-separated: V (vision), T (tactile), FT (force-torque), GF (gripper force), G (gripper state). e.g. --modalities V T FT GF G')
    return p.parse_args()


def make_split(dataset, args):
    if args.split == 'object':
        return split_by_object(dataset, test_objects=args.test_objects)
    elif args.split == 'pose':
        return split_by_pose(dataset, test_pose_indices=args.test_poses)
    else:
        return uniform_random_split(dataset)


def make_loader(dataset, subset, sampler=None, batch_size=32, num_workers=4, shuffle=False):
    if sampler is not None:
        # batch_sampler controls both batching and shuffling — don't pass batch_size/shuffle
        return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def batch_to_device(batch, device):
    tac, rgb, ft, grip, gf, label, pose_label = batch
    return (
        tac.to(device),
        rgb.to(device),
        ft.to(device),
        grip.to(device),
        gf.to(device),
        label.to(device),
        pose_label.to(device),
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for batch in loader:
        tac, rgb, ft, grip, gf, label, _ = batch_to_device(batch, device)
        logits = model(tac, rgb, ft, grip, gf)
        total_loss += criterion(logits, label).item() * len(label)
        correct    += (logits.argmax(1) == label).sum().item()
        n          += len(label)
    return total_loss / n, correct / n


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Modalities: {args.modalities}")

    # dataset
    ds = PoseItDataset(root_dir=args.root_dir)
    train_set, val_set, test_set = make_split(ds, args)
    print(f"Split ({args.split}): train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    # deferred sampling
    sampler = DRSSampler(
        dataset=ds,
        sigma=args.sigma,
        batch_size=args.batch_size,
        indices=train_set.indices,
    )

    train_loader = make_loader(ds, train_set, sampler=sampler, num_workers=args.num_workers)
    val_loader   = make_loader(ds, val_set,   batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader  = make_loader(ds, test_set,  batch_size=args.batch_size, num_workers=args.num_workers)

    # Model
    model = BaselineTactileEncoder(
        vision_resnet_emb_size=2048,
        tactile_resnet_emb_size=2048,
        model_emb_size=args.model_emb,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        modalities=args.modalities,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=0.9,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # training loop
    best_val_acc = 0.0
    iteration    = 0

    while iteration < args.n_iters:
        model.train()

        for batch in train_loader:
            if iteration >= args.n_iters:
                break

            # LR anneal + activate DRS at the right iteration
            if iteration == args.anneal_iter:
                scheduler.step()
                sampler.activate()
                print(f"[iter {iteration}] LR annealed to {scheduler.get_last_lr()}")

            tac, rgb, ft, grip, gf, label, _ = batch_to_device(batch, device)

            optimizer.zero_grad()
            logits = model(tac, rgb, ft, grip, gf)
            loss   = criterion(logits, label)
            loss.backward()
            optimizer.step()

            if iteration % 50 == 0:
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                print(f"[iter {iteration:4d}] train_loss={loss.item():.4f}  "
                      f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%  "
                      f"DRS={'on' if sampler.is_active else 'off'}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), 'best_model.pt')

            iteration += 1

    # test
    print("\nLoading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load('best_model.pt'))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss={test_loss:.4f}  Test acc={test_acc*100:.2f}%")


if __name__ == '__main__':
    main()