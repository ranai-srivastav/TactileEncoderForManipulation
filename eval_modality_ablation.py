from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import dataloader_clipt3 as dl
from clipt3_model import ALL_MODALITIES, CLIPT3


DEFAULT_ENTITY = "mrsd-smores"
DEFAULT_PROJECT = "TEMU"
DEFAULT_RUN = "fl6nvyho"


@dataclass
class SensorStats:
    ft_mean: torch.Tensor
    ft_std: torch.Tensor
    gr_mean: torch.Tensor
    gr_std: torch.Tensor
    gf_mean: torch.Tensor
    gf_std: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--wandb-run", default=DEFAULT_RUN)
    parser.add_argument("--wandb-entity", default=DEFAULT_ENTITY)
    parser.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    parser.add_argument("--checkpoint-name", default="best_model.pt")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--download-dir", type=Path, default=Path("wandb_downloads"))
    parser.add_argument("--output-json", type=Path, default=Path("analysis_outputs/modality_ablation.json"))
    parser.add_argument("--output-md", type=Path, default=Path("analysis_outputs/modality_ablation.md"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--evaluation-mode",
        choices=["single_active", "single_drop"],
        default="single_active",
    )
    parser.add_argument("--include-all-modalities-row", action="store_true")
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
) -> tuple[dict, Path, Path | None]:
    import wandb

    download_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    run = api.run(run_path)
    checkpoint_path = Path(
        run.file(checkpoint_name).download(root=str(download_dir / run.id), replace=True).name
    )

    output_log = None
    try:
        output_log = Path(
            run.file("output.log").download(root=str(download_dir / run.id), replace=True).name
        )
    except Exception:
        output_log = None
    return dict(run.config), checkpoint_path, output_log


def configure_dataloader(config: dict) -> None:
    dl.F1 = int(config.get("F1", 1))
    dl.F2 = int(config.get("F2", 1))
    dl.FT_DIM = dl.F2 * 6
    dl.GR_DIM = dl.F2 * 2
    dl.L = None if config.get("variable_length", False) else config.get("L")


def make_split(dataset: dl.PoseItDatasetCLIPT3, config: dict, seed: int):
    split = config.get("split", "random")
    if split == "object":
        return dl.split_by_object(dataset, config.get("test_objects", ["mug", "bowl"]), seed=seed)
    if split == "pose":
        return dl.split_by_pose(dataset, config.get("test_poses", [1, 2, 3, 4, 5]), seed=seed)
    return dl.uniform_random_split(dataset, seed=seed)


def compute_sensor_stats(
    dataset: dl.PoseItDatasetCLIPT3,
    indices: Iterable[int],
    desc: str = "Sensor stats",
) -> SensorStats:
    index_list = list(indices)
    ft_rows = []
    gr_rows = []
    gf_rows = []
    for index in tqdm(index_list, desc=desc, unit="sample"):
        ft, gripper, gripper_force = dataset.sensor_sample(index)
        ft_rows.append(ft)
        gr_rows.append(gripper)
        gf_rows.append(gripper_force)

    ft = torch.cat(ft_rows, dim=0)
    gr = torch.cat(gr_rows, dim=0)
    gf = torch.cat(gf_rows, dim=0)

    def _stats(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = values.mean(dim=0)
        std = values.std(dim=0, unbiased=False).clamp_min(1e-6)
        return mean, std

    ft_mean, ft_std = _stats(ft)
    gr_mean, gr_std = _stats(gr)
    gf_mean, gf_std = _stats(gf)
    return SensorStats(ft_mean, ft_std, gr_mean, gr_std, gf_mean, gf_std)


def standardize_batch(
    ft: torch.Tensor,
    gripper: torch.Tensor,
    gripper_force: torch.Tensor,
    lengths: torch.Tensor,
    stats: SensorStats,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = ft.device
    ft = (ft - stats.ft_mean.to(device)) / stats.ft_std.to(device)
    gripper = (gripper - stats.gr_mean.to(device)) / stats.gr_std.to(device)
    gripper_force = (gripper_force - stats.gf_mean.to(device)) / stats.gf_std.to(device)

    max_steps = ft.shape[1]
    mask = (torch.arange(max_steps, device=device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
    ft = ft * mask
    gripper = gripper * mask
    return ft, gripper, gripper_force


@torch.no_grad()
def evaluate(
    model: CLIPT3,
    loader: DataLoader,
    device: torch.device,
    stats: SensorStats | None,
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


def modality_sets(mode: str) -> list[tuple[str, list[str]]]:
    if mode == "single_drop":
        return [
            (f"drop-{modality}", [m for m in ALL_MODALITIES if m != modality])
            for modality in ALL_MODALITIES
        ]
    return [(f"{modality}-only", [modality]) for modality in ALL_MODALITIES]


def load_model(config: dict, checkpoint_path: Path, device: torch.device) -> CLIPT3:
    model = CLIPT3(
        hidden_dim=int(config.get("hidden_dim", 256)),
        lstm_layers=int(config.get("lstm_layers", 2)),
        dropout=float(config.get("dropout", 0.1)),
        bidirectional=not bool(config.get("unidirectional", False)),
        modalities=config.get("modalities", list(ALL_MODALITIES)),
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    return model


def parse_logged_test_metrics(output_log: Path | None) -> dict[str, float] | None:
    if output_log is None or not output_log.exists():
        return None
    pattern = re.compile(
        r"Test loss=(?P<loss>[0-9.]+)\s+acc=(?P<acc>[0-9.]+)%\s+prec=(?P<prec>[0-9.]+)\s+rec=(?P<rec>[0-9.]+)\s+f1=(?P<f1>[0-9.]+)"
    )
    for line in reversed(output_log.read_text().splitlines()):
        match = pattern.search(line)
        if match:
            return {
                "loss": float(match.group("loss")),
                "accuracy": float(match.group("acc")) / 100.0,
                "precision": float(match.group("prec")),
                "recall": float(match.group("rec")),
                "f1": float(match.group("f1")),
            }
    return None


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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    run_path = resolve_run_path(args.wandb_run, args.wandb_entity, args.wandb_project)
    if args.checkpoint_path is None:
        config, checkpoint_path, output_log = download_run_assets(
            run_path=run_path,
            download_dir=args.download_dir,
            checkpoint_name=args.checkpoint_name,
        )
    else:
        config = {
            "F1": 1,
            "F2": 1,
            "L": None,
            "variable_length": True,
            "split": "random",
            "hidden_dim": 256,
            "lstm_layers": 2,
            "dropout": 0.1,
            "modalities": list(ALL_MODALITIES),
            "unidirectional": False,
            "standardize_sensors": True,
        }
        checkpoint_path = args.checkpoint_path
        output_log = None

    configure_dataloader(config)
    dataset = dl.PoseItDatasetCLIPT3(root_dir=args.root_dir)
    train_set, val_set, test_set = make_split(dataset, config, seed=args.random_seed)

    stats = None
    warnings = []
    if config.get("split") == "random":
        warnings.append(
            "Random split reconstruction is approximate because the W&B run does not expose split indices or a split seed."
        )

    if config.get("standardize_sensors", False):
        stats = compute_sensor_stats(dataset, train_set.indices)

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dl.collate_variable_length,
    )

    model = load_model(config, checkpoint_path, device)
    rows = []

    if args.include_all_modalities_row:
        model.set_modalities(config.get("modalities", list(ALL_MODALITIES)))
        baseline = evaluate(model, test_loader, device, stats, desc="Evaluating all-modalities")
        rows.append(
            {
                "setting": "all-modalities",
                "active_modalities": list(config.get("modalities", list(ALL_MODALITIES))),
                **baseline,
            }
        )
        print_metrics_row("all-modalities", baseline, list(config.get("modalities", list(ALL_MODALITIES))))

    for setting, active_modalities in tqdm(modality_sets(args.evaluation_mode), desc="Ablations", unit="setting"):
        model.set_modalities(active_modalities)
        metrics = evaluate(model, test_loader, device, stats, desc=f"Evaluating {setting}")
        rows.append(
            {
                "setting": setting,
                "active_modalities": active_modalities,
                **metrics,
            }
        )
        print_metrics_row(setting, metrics, active_modalities)

    logged_metrics = parse_logged_test_metrics(output_log)
    report = {
        "run_path": run_path,
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "config": config,
        "random_seed": args.random_seed,
        "warnings": warnings,
        "split_sizes": {
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set),
        },
        "logged_reference_metrics": logged_metrics,
        "results": rows,
        "table_markdown": markdown_table(rows),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    args.output_md.write_text(report["table_markdown"] + "\n")

    print(report["table_markdown"])
    if warnings:
        print()
        for warning in warnings:
            print(f"[WARN] {warning}")
    if logged_metrics is not None:
        print()
        print(
            "Logged full-model test metrics from W&B run: "
            f"acc={logged_metrics['accuracy']:.4f} "
            f"prec={logged_metrics['precision']:.4f} "
            f"rec={logged_metrics['recall']:.4f}"
        )
        print("Current evaluation only reports the requested modality-zeroed settings.")


if __name__ == "__main__":
    main()
