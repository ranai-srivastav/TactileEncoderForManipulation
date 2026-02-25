import argparse
import os
import sys

# Add project root to path so dataloader.py can be found when running from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import dataloader as _dl
from dataloader import PoseItDataset, collate_variable_length, split_by_object
from unimodal_2d import GraspClassifier

'''
Usage: 
    python train_2d.py --modality rgb
    python train_2d.py --modality tactile
    python train_2d.py --test --checkpoint trained_models/best_2d.pt --modality rgb

    python unimodal/train_2d.py --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight --modality rgb
'''


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
    p.add_argument('--L',           type=int,   default=3,
                   help='Last L seconds')
    p.add_argument('--test_objects', nargs='+', default=['flashlight'])
    p.add_argument('--checkpoint',  type=str,   default='trained_models/best_2d.pt',
                   help='Path to save/load model checkpoint')
    p.add_argument('--test',        action='store_true',
                   help='Load checkpoint and evaluate on test set only')
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, criterion, device, modality):
    """Returns (loss, accuracy, precision, recall, f1)."""
    model.eval()
    total_loss = 0.0
    tp, fp, fn, tn = 0, 0, 0, 0

    for tactile, rgb, _, _, _, labels, _, lengths in loader:
        imgs    = (tactile if modality == 'tactile' else rgb).to(device)
        labels  = labels.to(device)
        lengths = lengths.to(device)

        logits = model(imgs, lengths)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)

        preds  = logits.argmax(dim=1)
        actual = labels
        tp += ((preds == 1) & (actual == 1)).sum().item()
        fp += ((preds == 1) & (actual == 0)).sum().item()
        fn += ((preds == 0) & (actual == 1)).sum().item()
        tn += ((preds == 0) & (actual == 0)).sum().item()

    n = tp + fp + fn + tn
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    acc       = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return total_loss / n, acc, precision, recall, f1


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}  modality: {args.modality}")

    _dl.L = args.L

    dataset = PoseItDataset(root_dir=args.root_dir)
    train_set, val_set, test_set = split_by_object(dataset, test_objects=args.test_objects)
    print(f"train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

    criterion = nn.CrossEntropyLoss()

    # ---- Test-only mode ----
    if args.test:
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_variable_length)

        model = GraspClassifier(freeze_backbone=True).to(device)
        assert os.path.exists(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")

        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
            model, test_loader, criterion, device, args.modality)
        print(f"\nTest Results:")
        print(f"  loss={test_loss:.4f}  acc={test_acc*100:.2f}%  "
              f"prec={test_prec:.3f}  rec={test_rec:.3f}  f1={test_f1:.3f}")
        return

    # ---- Training mode ----
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=collate_variable_length)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              collate_fn=collate_variable_length)

    model     = GraspClassifier(freeze_backbone=True).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(args.checkpoint) or '.', exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        n_samples    = 0

        for tactile, rgb, _, _, _, labels, _, lengths in train_loader:
            imgs    = (tactile if args.modality == 'tactile' else rgb).to(device)
            labels  = labels.to(device)
            lengths = lengths.to(device)

            logits = model(imgs, lengths)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            n_samples    += labels.size(0)

        # Validation
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
            model, val_loader, criterion, device, args.modality)

        train_loss = running_loss / n_samples
        saved = ''
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.checkpoint)
            saved = '  [saved]'

        print(f"Epoch {epoch+1:2d}/{args.epochs}: "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%  "
              f"prec={val_prec:.3f}  rec={val_rec:.3f}  f1={val_f1:.3f}{saved}")

    print(f"\nBest val acc: {best_val_acc*100:.2f}%  (checkpoint: {args.checkpoint})")
    print(f"Run with --test to evaluate on the test set.")


if __name__ == '__main__':
    main()