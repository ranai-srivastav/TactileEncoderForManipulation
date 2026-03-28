#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import dataloader_resnet_varlen as dl
from model_resnet_varlen import ALL_MODALITIES, GraspStabilityLSTMVarLen


DEFAULT_ENTITY = "mrsd-smores"
DEFAULT_PROJECT = "TEMU"
DEFAULT_RUN = "vfiu3oh4"
DEFAULT_CHECKPOINT = "best_model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the vfiu3oh4 variable-length ResNet model over the full loaded dataset."
    )
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--wandb-run", default=DEFAULT_RUN)
    parser.add_argument("--wandb-entity", default=DEFAULT_ENTITY)
    parser.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--download-dir", type=Path, default=Path("wandb_downloads"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("analysis_outputs/vfiu3oh4_full_dataset_ablation.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("analysis_outputs/vfiu3oh4_full_dataset_ablation.md"),
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolve_run_path(run_spec: str, entity: str, project: str) -> str:
    if run_spec.startswith("http://") or run_spec.startswith("https://"):
        match = re.search(r"wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#]+)", run_spec)
        if not match:
            raise ValueError(f"Could not parse W&B run URL: {run_spec}")
        return f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
    if run_spec.count("/") == 2:
        return run_spec
    return f"{entity}/{project}/{run_spec}"


def download_run_assets(
    run_path: str,
    download_dir: Path,
    checkpoint_name: str,
) -> tuple[dict, dict, Path]:
    import wandb

    download_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    run = api.run(run_path)
    checkpoint_path = Path(
        run.file(checkpoint_name).download(root=str(download_dir / run.id), replace=True).name
    )
    return dict(run.config), dict(run.summary), checkpoint_path


def configure_dataloader(config: dict) -> tuple[int, int, int]:
    dl.F1 = int(config.get("F1", 1))
    dl.F2 = int(config.get("F2", 1))
    dl.FT_DIM = dl.F2 * 6
    dl.GR_DIM = dl.F2 * 2
    dl.L = None if bool(config.get("variable_length", False)) else int(config.get("L", 20))
    return dl.F1, dl.FT_DIM, dl.GR_DIM


def standardize_batch(
    ft: torch.Tensor,
    gripper: torch.Tensor,
    gripper_force: torch.Tensor,
    lengths: torch.Tensor,
    stats: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = ft.device
    ft = (ft - stats["ft_mean"].to(device)) / stats["ft_std"].to(device)
    gripper = (gripper - stats["gr_mean"].to(device)) / stats["gr_std"].to(device)
    gripper_force = (gripper_force - stats["gf_mean"].to(device)) / stats["gf_std"].to(device)

    max_steps = ft.shape[1]
    mask = (torch.arange(max_steps, device=device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
    ft = ft * mask
    gripper = gripper * mask
    return ft, gripper, gripper_force


@torch.no_grad()
def evaluate(
    model: GraspStabilityLSTMVarLen,
    loader: DataLoader,
    device: torch.device,
    stats: dict[str, torch.Tensor] | None,
    desc: str,
) -> dict[str, float]:
    model.eval()
    tp = fp = fn = tn = 0

    for batch in tqdm(loader, desc=desc, unit="batch"):
        tactile, rgb, ft, gripper, gripper_force, label, _pose_label, lengths = batch
        tactile = tactile.to(device)
        rgb = rgb.to(device)
        ft = ft.to(device)
        gripper = gripper.to(device)
        gripper_force = gripper_force.to(device)
        label = label.to(device)
        lengths = lengths.to(device)

        if stats is not None:
            ft, gripper, gripper_force = standardize_batch(ft, gripper, gripper_force, lengths, stats)

        logits = model(tactile, rgb, ft, gripper, gripper_force, lengths).squeeze(1)
        prediction = logits > 0
        actual = label.bool()

        tp += int((prediction & actual).sum().item())
        fp += int((prediction & ~actual).sum().item())
        fn += int((~prediction & actual).sum().item())
        tn += int((~prediction & ~actual).sum().item())

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "n": float(total),
    }


def markdown_table(rows: list[dict[str, object]]) -> str:
    header = "| Setting | Active Modalities | Accuracy | Precision | Recall |"
    divider = "| --- | --- | ---: | ---: | ---: |"
    body = [
        f"| {row['setting']} | {' '.join(row['active_modalities'])} | "
        f"{row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def print_metrics_row(setting: str, metrics: dict[str, float], active_modalities: list[str]) -> None:
    active = " ".join(active_modalities)
    print(
        f"[{setting}] active={active} "
        f"acc={metrics['accuracy']:.4f} "
        f"prec={metrics['precision']:.4f} "
        f"rec={metrics['recall']:.4f}"
    )


def json_safe(value):
    return json.loads(json.dumps(value, default=str))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    run_path = resolve_run_path(args.wandb_run, args.wandb_entity, args.wandb_project)
    print(f"Resolving W&B run: {run_path}")
    config, summary, checkpoint_path = download_run_assets(
        run_path,
        args.download_dir,
        args.checkpoint_name,
    )
    print(f"Downloaded checkpoint: {checkpoint_path}")

    frames_per_sec, ft_dim, gr_dim = configure_dataloader(config)
    print(
        f"Configured loader: L={dl.L}, F1={dl.F1}, F2={dl.F2}, "
        f"ft_dim={ft_dim}, gripper_dim={gr_dim}"
    )

    print("Building full dataset...")
    dataset = dl.PoseItDatasetResNetVarLen(root_dir=args.root_dir)
    print(f"Loaded dataset for full evaluation: {len(dataset)} samples")

    stats = None
    if bool(config.get("standardize_sensors", False)):
        stats = dl.compute_sensor_stats(
            dataset,
            range(len(dataset)),
            desc="Sensor stats over full dataset",
        )
        print("Computed full-dataset sensor standardization stats.")

    batch_size = args.batch_size if args.batch_size is not None else int(config.get("batch_size", 32))
    num_workers = args.num_workers if args.num_workers is not None else int(config.get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dl.collate_variable_length,
    )
    print(f"DataLoader ready: batch_size={batch_size}, num_workers={num_workers}")

    requested_modalities = list(config.get("modalities", list(ALL_MODALITIES)))
    print("Constructing model...")
    model = GraspStabilityLSTMVarLen(
        frames_per_sec=frames_per_sec,
        ft_dim=ft_dim,
        gripper_dim=gr_dim,
        hidden_dim=int(config.get("hidden_dim", 256)),
        lstm_layers=int(config.get("lstm_layers", 2)),
        bidirectional=not bool(config.get("unidirectional", False)),
        dropout=float(config.get("dropout", 0.1)),
        modalities=requested_modalities,
        resnet_weights=None,
    ).to(device)
    print("Loading checkpoint into model...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    print("Checkpoint loaded.")

    active_sets = [("all-modalities", requested_modalities)] + [
        (f"{modality}-only", [modality]) for modality in requested_modalities
    ]

    rows = []
    print("Evaluating over the full loaded dataset, not the original random split.")
    for setting, active_modalities in tqdm(active_sets, desc="Ablations", unit="setting"):
        model.set_modalities(active_modalities)
        metrics = evaluate(model, loader, device, stats, desc=f"Evaluating {setting}")
        row = {
            "setting": setting,
            "active_modalities": active_modalities,
            **metrics,
        }
        rows.append(row)
        print_metrics_row(setting, metrics, active_modalities)

    table = markdown_table(rows)
    print()
    print(table)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "wandb_run": run_path,
                "checkpoint_path": str(checkpoint_path),
                "dataset_root": args.root_dir,
                "dataset_size": len(dataset),
                "config": json_safe(config),
                "wandb_summary": json_safe(summary),
                "full_dataset_results": rows,
                "table_markdown": table,
                "note": "Evaluated on the full loaded dataset, not the original random split.",
            },
            indent=2,
        )
    )
    args.output_md.write_text(table + "\n")


if __name__ == "__main__":
    main()
