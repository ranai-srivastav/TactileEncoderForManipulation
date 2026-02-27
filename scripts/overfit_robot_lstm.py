import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataloader as _dl
from dataloader import PoseItDataset
from model import RobotStateOnlyLSTM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", default="./data")
    p.add_argument("--L", type=int, default=9)
    p.add_argument("--F2", type=int, default=1)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_samples", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _dl.L = args.L
    _dl.F2 = args.F2

    ds = PoseItDataset(
        root_dir=args.root_dir,
        load_tactile=False,
        load_rgb=False,
        show_progress=False,
    )
    if len(ds) == 0:
        raise RuntimeError(f"No samples loaded from {args.root_dir}")

    n = min(args.num_samples, len(ds))
    subset = Subset(ds, list(range(n)))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)

    robot_dim = ds.samples[0]["robot"].shape[-1]
    model = RobotStateOnlyLSTM(
        robot_dim=robot_dim,
        hidden_dim=args.hidden_dim,
        dropout=0.0,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses = []
    model.train()
    for step in range(args.steps):
        batch = next(iter(loader))
        tactile, rgb, ft, gripper, robot, gf, label, _ = batch

        tactile = tactile.to(device)
        rgb = rgb.to(device)
        ft = ft.to(device)
        gripper = gripper.to(device)
        robot = robot.to(device)
        gf = gf.to(device)
        label = label.float().to(device)

        optimizer.zero_grad()
        logits = model(tactile, rgb, ft, gripper, gf, robot=robot).squeeze(1)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"step={step:03d} loss={loss.item():.6f}")

    print("loss_trajectory=", " ".join(f"{loss:.6f}" for loss in losses))


if __name__ == "__main__":
    main()
