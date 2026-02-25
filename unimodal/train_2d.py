import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import dataloader as _dl
from dataloader import PoseItDataset, collate_variable_length, split_by_object
from unimodal.unimodal_2d import GraspClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir',    default='./data',
                   help='Path to the dataset root directory')
    p.add_argument('--modality',    default='rgb', choices=['rgb', 'tactile'],
                   help='Which image stream to use (both have shape B,T,F1,3,H,W)')
    p.add_argument('--batch_size',  type=int,   default=8)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--epochs',      type=int,   default=20)
    p.add_argument('--num_workers', type=int,   default=4)
    p.add_argument('--L',           type=int,   default=20,
                   help='Max seconds per episode')
    p.add_argument('--test_objects', nargs='+', default=['flashlight'])
    return p.parse_args()


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}  modality: {args.modality}")

    _dl.L = args.L

    dataset = PoseItDataset(root_dir=args.root_dir)
    train_set, val_set, test_set = split_by_object(dataset, test_objects=args.test_objects)
    print(f"train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=collate_variable_length)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              collate_fn=collate_variable_length)

    model     = GraspClassifier(freeze_backbone=True).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        for tactile, rgb, _, _, _, labels, _, lengths in train_loader:
            imgs   = (tactile if args.modality == 'tactile' else rgb).to(device)
            labels  = labels.to(device)
            lengths = lengths.to(device)

            logits = model(imgs, lengths)          # (B, num_classes)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for tactile, rgb, _, _, _, labels, _, lengths in val_loader:
                imgs    = (tactile if args.modality == 'tactile' else rgb).to(device)
                labels  = labels.to(device)
                lengths = lengths.to(device)
                preds   = model(imgs, lengths).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        print(f"Epoch {epoch+1:2d}: val acc = {correct/total:.3f}")


if __name__ == '__main__':
    main()
