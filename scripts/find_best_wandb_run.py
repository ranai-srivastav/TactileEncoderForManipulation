#!/usr/bin/env python3
"""
Find the wandb run with the best metrics.

Usage:
    python scripts/find_best_wandb_run.py [--project TEMU] [--entity mrsd-smores] [--metric test/f1]
"""

import argparse

try:
    import wandb
except ImportError:
    print("Install wandb: pip install wandb")
    exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="TEMU", help="W&B project name")
    p.add_argument("--entity", default="mrsd-smores", help="W&B entity")
    p.add_argument("--metric", default="test/f1",
                   help="Metric to rank by (e.g. test/f1, val/f1, test/acc)")
    p.add_argument("--limit", type=int, default=100, help="Max runs to fetch")
    args = p.parse_args()

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}", per_page=args.limit)

    results = []
    for run in runs:
        summary = run.summary._json_dict if run.summary else {}
        metric_val = summary.get(args.metric)
        if metric_val is not None:
            results.append({
                "id": run.id,
                "name": run.name,
                "metric": metric_val,
                "config": dict(run.config) if run.config else {},
            })

    if not results:
        print(f"No runs with {args.metric} found in {args.entity}/{args.project}")
        return

    results.sort(key=lambda x: x["metric"], reverse=True)
    print(f"Top 10 runs by {args.metric} (higher is better):\n")
    for i, r in enumerate(results[:10], 1):
        print(f"  {i}. {r['name']} (id={r['id']})")
        print(f"     {args.metric}={r['metric']:.4f}")
        if r["config"].get("model"):
            print(f"     model={r['config']['model']}")
        print()

    best = results[0]
    print(f"Best run: {best['name']}")
    print(f"  URL: https://wandb.ai/{args.entity}/{args.project}/runs/{best['id']}")
    print(f"  {args.metric}={best['metric']:.4f}")


if __name__ == "__main__":
    main()
