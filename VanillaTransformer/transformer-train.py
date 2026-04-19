import argparse
import os
import random
import re
import sys
from os import path
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# repo root for dataloader.py
ROOT = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dataloader as _dl
from dataloader import PoseItDataset, split_by_object, split_by_pose, uniform_random_split
from transformer_model import VanillaTransformer


def print_split_stats(dataset, train_set, val_set, test_set):
    print("Label mapping:")
    print("  0 = pass / no slip")
    print("  1 = slip or drop / fail")

    def _stats(name, subset):
        indices = subset.indices
        labels = [dataset.samples[i]["label"].item() for i in indices]
        n_total = len(labels)
        n_zero = sum(1 for x in labels if x == 0)
        n_one = sum(1 for x in labels if x == 1)
        zero_pct = 100.0 * n_zero / n_total if n_total else 0.0
        one_pct = 100.0 * n_one / n_total if n_total else 0.0
        print(
            f"{name}: total={n_total}  "
            f"0(pass/no slip)={n_zero} ({zero_pct:.1f}%)  "
            f"1(slip/drop)={n_one} ({one_pct:.1f}%)"
        )

    print("Split statistics:")
    _stats("Train", train_set)
    _stats("Val", val_set)
    _stats("Test", test_set)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", default="./data")
    p.add_argument("--split", default="object", choices=["object", "pose", "random"])
    p.add_argument("--test_object_ids", nargs="+", type=int, default=None)
    p.add_argument("--n_test_objects", type=int, default=None)
    p.add_argument("--test_pose_ids", nargs="+", type=int, default=None)
    p.add_argument("--n_test_poses", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--FRGB", type=int, default=1)
    p.add_argument("--FTactile", type=int, default=1)
    p.add_argument("--FFT", type=int, default=1)
    p.add_argument("--FGripper", type=int, default=1)
    p.add_argument("--L", type=int, default=20)
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--debug_max_episodes", type=int, default=None)
    p.add_argument("--debug_max_episodes_per_object", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden_dim", type=int, default=768)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--modalities", nargs="+", default=["V", "T", "FT", "G"])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--model_save_path", default="trained_models/best_vanilla_transformer.pt")
    p.add_argument("--resume_path", default=None)
    p.add_argument("--resume_wandb_artifact", default=None)
    p.add_argument("--wandb_checkpoint_interval", type=int, default=1)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    return p.parse_args()


def object_from_episode_dir(episode_dir):
    match = re.match(r"^(.+)_(\d+)_F(\d+)_pose(\d+)$", episode_dir.name)
    return match.group(1) if match else None


def select_debug_sample_dirs(root_dir, max_episodes=None, max_episodes_per_object=None):
    if max_episodes is None and max_episodes_per_object is None:
        return None

    selected = []
    per_object = {}
    for episode_dir in sorted(Path(root_dir).iterdir()):
        if not episode_dir.is_dir():
            continue
        obj = object_from_episode_dir(episode_dir)
        if obj is None:
            continue
        if max_episodes_per_object is not None:
            count = per_object.get(obj, 0)
            if count >= max_episodes_per_object:
                continue
            per_object[obj] = count + 1
        selected.append(str(episode_dir))
        if max_episodes is not None and len(selected) >= max_episodes:
            break

    print(
        f"[DEBUG] Loading only {len(selected)} episode dirs "
        f"(max_episodes={max_episodes}, max_episodes_per_object={max_episodes_per_object})"
    )
    return selected


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_rng_state(train_generator):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "train_generator": train_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state, train_generator):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    train_generator.set_state(state["train_generator"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def make_split(dataset, args):
    if args.split == "object":
        objects = sorted(set(s["object"] for s in dataset.samples))
        print("Object index (sorted alphabetically):")
        for i, obj in enumerate(objects):
            print(f"  {i:>3}: {obj}")
        if args.test_object_ids is not None:
            test_objects = [objects[i] for i in args.test_object_ids]
        elif args.n_test_objects is not None:
            test_objects = sorted(random.sample(objects, args.n_test_objects))
        else:
            raise ValueError("--split object requires --test_object_ids or --n_test_objects")
        return split_by_object(dataset, test_objects)

    if args.split == "pose":
        poses = sorted(set(s["pose_idx"] for s in dataset.samples))
        print(f"Pose IDs in dataset: {poses}")
        if args.test_pose_ids is not None:
            test_poses = args.test_pose_ids
        elif args.n_test_poses is not None:
            test_poses = sorted(random.sample(poses, args.n_test_poses))
        else:
            raise ValueError("--split pose requires --test_pose_ids or --n_test_poses")
        return split_by_pose(dataset, test_poses)

    return uniform_random_split(dataset)


def make_loader(subset, batch_size, num_workers, shuffle=False, generator=None):
    return DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        generator=generator,
        worker_init_fn=seed_worker if generator is not None else None,
        collate_fn=pad_collate,
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _pad_time(tensor, max_timesteps):
    if tensor.shape[0] == max_timesteps:
        return tensor
    padded = tensor.new_zeros((max_timesteps, *tensor.shape[1:]))
    padded[: tensor.shape[0]] = tensor
    return padded


def pad_collate(batch):
    timesteps = [sample[0].shape[0] for sample in batch]
    max_timesteps = max(timesteps)
    timestep_mask = torch.zeros(len(batch), max_timesteps, dtype=torch.bool)

    tactile, rgb, ft, grip, gf, label, pose_label = [], [], [], [], [], [], []
    for i, sample in enumerate(batch):
        tac_i, rgb_i, ft_i, grip_i, gf_i, label_i, pose_label_i = sample
        timestep_mask[i, : timesteps[i]] = True
        tactile.append(_pad_time(tac_i, max_timesteps))
        rgb.append(_pad_time(rgb_i, max_timesteps))
        ft.append(_pad_time(ft_i, max_timesteps))
        grip.append(_pad_time(grip_i, max_timesteps))
        gf.append(gf_i)
        label.append(label_i)
        pose_label.append(pose_label_i)

    return (
        torch.stack(tactile, dim=0),
        torch.stack(rgb, dim=0),
        torch.stack(ft, dim=0),
        torch.stack(grip, dim=0),
        torch.stack(gf, dim=0),
        torch.stack(label, dim=0),
        torch.stack(pose_label, dim=0),
        timestep_mask,
    )


def batch_to_device(batch, device):
    tac, rgb, ft, grip, gf, label, pose_label, timestep_mask = batch
    return (
        tac.to(device),
        rgb.to(device),
        ft.to(device),
        grip.to(device),
        gf.to(device),
        label.to(device),
        pose_label.to(device),
        timestep_mask.to(device),
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    tp = fp = tn = fn = n = 0
    y_true = []
    y_pred = []

    for batch in loader:
        tac, rgb, ft, grip, gf, label, _, timestep_mask = batch_to_device(batch, device)
        logits = model(tac, rgb, ft, grip, gf, timestep_mask=timestep_mask).squeeze(-1)
        total_loss += criterion(logits, label.float()).item() * len(label)

        preds = logits > 0
        actual = label.bool()
        tp += (preds & actual).sum().item()
        fp += (preds & ~actual).sum().item()
        tn += (~preds & ~actual).sum().item()
        fn += (~preds & actual).sum().item()
        n += len(label)
        y_true.extend(label.cpu().tolist())
        y_pred.extend(preds.long().cpu().tolist())

    if n == 0:
        return {
            "loss": 0.0,
            "acc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "y_true": [],
            "y_pred": [],
        }

    acc = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "loss": total_loss / n,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def wandb_confusion_matrix(metrics):
    return wandb.plot.confusion_matrix(
        y_true=metrics["y_true"],
        preds=metrics["y_pred"],
        class_names=["pass/no slip", "slip/drop"],
    )


def save_checkpoint(pathname, model, optimizer, epoch, best_val_f1, args, train_generator):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_f1": best_val_f1,
        "args": vars(args),
        "rng_state": get_rng_state(train_generator),
    }, pathname)


def wandb_artifact_ref(args):
    ref = args.resume_wandb_artifact
    if ref is None:
        return None
    if "/" in ref:
        return ref
    if args.wandb_project is None:
        raise ValueError("--resume_wandb_artifact without a full entity/project/name:alias ref requires --wandb_project")
    if args.wandb_entity is not None:
        return f"{args.wandb_entity}/{args.wandb_project}/{ref}"
    return f"{args.wandb_project}/{ref}"


def download_wandb_checkpoint(args):
    ref = wandb_artifact_ref(args)
    if ref is None:
        return None
    if not _WANDB_AVAILABLE:
        raise ImportError("wandb is required for --resume_wandb_artifact")

    artifact = wandb.Api().artifact(ref, type="model")
    download_root = path.join(path.dirname(args.model_save_path) or ".", "wandb_artifacts")
    artifact_dir = artifact.download(root=download_root)
    checkpoint_path = path.join(artifact_dir, "checkpoint.pt")
    if path.exists(checkpoint_path):
        return checkpoint_path
    for filename in os.listdir(artifact_dir):
        if filename.endswith(".pt"):
            return path.join(artifact_dir, filename)
    raise FileNotFoundError(f"No .pt checkpoint found in W&B artifact {ref}")


def apply_checkpoint_config(args, ckpt_args):
    if not ckpt_args:
        return args
    current = vars(args)
    keep_current = {
        "root_dir",
        "model_save_path",
        "resume_path",
        "resume_wandb_artifact",
        "wandb_project",
        "wandb_run",
        "wandb_entity",
    }
    for key, value in ckpt_args.items():
        if key in current and key not in keep_current:
            setattr(args, key, value)
    return args


def log_wandb_checkpoint(pathname, kind, epoch, best_val_f1, args):
    artifact = wandb.Artifact(
        name=f"{wandb.run.id}-{kind}",
        type="model",
        metadata={"epoch": epoch, "best_val_f1": best_val_f1, "config": vars(args)},
    )
    artifact.add_file(pathname, name="checkpoint.pt")
    aliases = [kind, f"epoch-{epoch + 1}", "latest"]
    wandb.log_artifact(artifact, aliases=aliases)


def load_checkpoint(pathname, map_location):
    try:
        return torch.load(pathname, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(pathname, map_location=map_location)


def main():
    args = parse_args()

    resume_checkpoint = None
    resume_path = args.resume_path
    if args.resume_wandb_artifact is not None:
        resume_path = download_wandb_checkpoint(args)
        print(f"Downloaded W&B checkpoint to {resume_path}")
    if resume_path is not None:
        resume_checkpoint = load_checkpoint(resume_path, map_location="cpu")
        args = apply_checkpoint_config(args, resume_checkpoint.get("args"))
        print("Loaded training config from checkpoint")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    use_wandb = _WANDB_AVAILABLE and args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            entity=args.wandb_entity,
            config=vars(args),
        )
        wandb.config.update(vars(args), allow_val_change=True)
        print(f"W&B latest checkpoint artifact: {'/'.join(wandb.run.path[:2])}/{wandb.run.id}-latest:latest")
        print(f"W&B best checkpoint artifact:   {'/'.join(wandb.run.path[:2])}/{wandb.run.id}-best:best")
    elif args.wandb_project is not None:
        print("[WARN] wandb not installed — W&B logging disabled.")

    _dl.L = args.L
    _dl.FRGB = args.FRGB
    _dl.FTactile = args.FTactile
    _dl.FFT = args.FFT
    _dl.FGripper = args.FGripper
    _dl.refresh_sampling_dims()

    sample_dirs = select_debug_sample_dirs(
        args.root_dir,
        max_episodes=args.debug_max_episodes,
        max_episodes_per_object=args.debug_max_episodes_per_object,
    )
    dataset = PoseItDataset(root_dir=args.root_dir) if sample_dirs is None else PoseItDataset(sample_dirs=sample_dirs)
    if len(dataset) == 0:
        raise ValueError(f"No samples found in {args.root_dir}")

    tac, rgb, ft, grip, gf, label, pose_label = dataset[0]
    print("Example sample shapes:")
    print(f"  tactile:       {tuple(tac.shape)}")
    print(f"  rgb:           {tuple(rgb.shape)}")
    print(f"  ft:            {tuple(ft.shape)}")
    print(f"  gripper:       {tuple(grip.shape)}")
    print(f"  gripper_force: {tuple(gf.shape)}")
    print(f"  label:         {label.item()}")
    print(f"  pose_label:    {pose_label.item()}")

    model = VanillaTransformer(
        frames_per_sec=args.FRGB,
        ft_dim=_dl.FT_DIM,
        gripper_dim=_dl.GR_DIM,
        max_timesteps=args.L if args.L > 0 else max(1, tac.shape[0]),
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        modalities=args.modalities,
    ).to(device)
    model.eval()

    with torch.no_grad():
        logits, debug = model(
            tac.unsqueeze(0).to(device),
            rgb.unsqueeze(0).to(device),
            ft.unsqueeze(0).to(device),
            grip.unsqueeze(0).to(device),
            gf.unsqueeze(0).to(device),
            return_debug=True,
        )

    print("Encoded output shapes:")
    for name, value in debug.items():
        print(f"  {name}: {tuple(value.shape)}")
    print(f"  logits:        {tuple(logits.shape)}")

    if args.overfit:
        dataset.samples = dataset.samples[:1]
        subset = Subset(dataset, [0])
        train_set = val_set = test_set = subset
    else:
        train_set, val_set, test_set = make_split(dataset, args)

    print_split_stats(dataset, train_set, val_set, test_set)

    train_generator = torch.Generator()
    if args.seed is not None:
        train_generator.manual_seed(args.seed)

    train_loader = make_loader(
        train_set,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        generator=train_generator,
    )
    val_loader = make_loader(val_set, args.batch_size, args.num_workers)
    test_loader = make_loader(test_set, args.batch_size, args.num_workers)

    print(f"Loaded dataset with {len(dataset.samples)} samples")
    print(f"Train / Val / Test: {len(train_set)} / {len(val_set)} / {len(test_set)}")
    print(
        f"FRGB={args.FRGB}, FTactile={args.FTactile}, "
        f"FFT={args.FFT}, FGripper={args.FGripper}, L={args.L}"
    )
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches:   {len(val_loader)}")
    print(f"Test loader batches:  {len(test_loader)}")

    train_labels = [dataset.samples[i]["label"].item() for i in train_set.indices]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    print(f"pos_weight={pos_weight.item():.3f} (n_pos={n_pos}, n_neg={n_neg})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_dir = path.dirname(args.model_save_path) or "."
    path.isdir(save_dir) or os.makedirs(save_dir, exist_ok=True)
    latest_path = path.join(save_dir, "vanilla_transformer_latest.pt")
    best_val_f1 = -1.0
    start_epoch = 0

    if resume_checkpoint is not None:
        ckpt = resume_checkpoint
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        optimizer_to_device(optimizer, device)
        set_rng_state(ckpt.get("rng_state"), train_generator)
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", -1.0)
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch in train_loader:
            tac, rgb, ft, grip, gf, label, _, timestep_mask = batch_to_device(batch, device)
            logits = model(tac, rgb, ft, grip, gf, timestep_mask=timestep_mask).squeeze(-1)
            loss = criterion(logits, label.float())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(label)
            train_count += len(label)

        avg_train_loss = train_loss_sum / max(train_count, 1)
        train_metrics = evaluate(model, train_loader, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        print(
            f"[epoch {epoch + 1:3d}/{args.epochs:3d}] "
            f"train_loss={avg_train_loss:.4f} "
            f"train_acc={train_metrics['acc']:.3f} "
            f"train_prec={train_metrics['precision']:.3f} "
            f"train_rec={train_metrics['recall']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.3f} "
            f"val_prec={val_metrics['precision']:.3f} "
            f"val_rec={val_metrics['recall']:.3f} "
            f"val_f1={val_metrics['f1']:.3f} "
            f"test_acc={test_metrics['acc']:.3f} "
            f"test_prec={test_metrics['precision']:.3f} "
            f"test_rec={test_metrics['recall']:.3f}"
        )
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "train/eval_loss": train_metrics["loss"],
                "train/acc": train_metrics["acc"],
                "train/precision": train_metrics["precision"],
                "train/recall": train_metrics["recall"],
                "train/f1": train_metrics["f1"],
                "train/confusion_matrix": wandb_confusion_matrix(train_metrics),
                "val/loss": val_metrics["loss"],
                "val/acc": val_metrics["acc"],
                "val/precision": val_metrics["precision"],
                "val/recall": val_metrics["recall"],
                "val/f1": val_metrics["f1"],
                "val/confusion_matrix": wandb_confusion_matrix(val_metrics),
                "test/loss": test_metrics["loss"],
                "test/acc": test_metrics["acc"],
                "test/precision": test_metrics["precision"],
                "test/recall": test_metrics["recall"],
                "test/f1": test_metrics["f1"],
                "test/confusion_matrix": wandb_confusion_matrix(test_metrics),
            })
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            save_checkpoint(args.model_save_path, model, optimizer, epoch, best_val_f1, args, train_generator)
            if use_wandb:
                log_wandb_checkpoint(args.model_save_path, "best", epoch, best_val_f1, args)
        save_checkpoint(latest_path, model, optimizer, epoch, best_val_f1, args, train_generator)
        should_log_latest = (
            args.wandb_checkpoint_interval > 0
            and (epoch + 1) % args.wandb_checkpoint_interval == 0
        )
        if use_wandb and should_log_latest:
            log_wandb_checkpoint(latest_path, "latest", epoch, best_val_f1, args)

    print("\nLoading best checkpoint for test evaluation...")
    best_ckpt = load_checkpoint(args.model_save_path, map_location=device)
    model.load_state_dict(best_ckpt["model"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(
        f"Test loss={test_metrics['loss']:.4f}  "
        f"acc={test_metrics['acc']:.3f}  "
        f"precision={test_metrics['precision']:.3f}  "
        f"recall={test_metrics['recall']:.3f}  "
        f"f1={test_metrics['f1']:.3f}"
    )
    if use_wandb:
        wandb.log({
            "best/test_loss": test_metrics["loss"],
            "best/test_acc": test_metrics["acc"],
            "best/test_precision": test_metrics["precision"],
            "best/test_recall": test_metrics["recall"],
            "best/test_f1": test_metrics["f1"],
            "best/test_confusion_matrix": wandb_confusion_matrix(test_metrics),
            "best_val_f1": best_val_f1,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
