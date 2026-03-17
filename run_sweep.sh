#!/bin/bash
# ============================================================
# Bridges-2 W&B Sweep — job array (one agent per array slot)
#
# Each array element runs one W&B agent independently.
# The W&B controller assigns a distinct config to each agent.
#
# Usage:
#   1. Create the sweep first (run once on login node):
#         wandb sweep sweep_data.yaml    # or sweep_optimizer.yaml
#         → prints: mrsd-smores/TEMU/<sweep-id>
#
#   2. Set SWEEP_ID below, then submit:
#         sbatch run_sweep.sh
#
#   Change --array to control how many agents run in parallel.
#   Example: --array=0-4  launches 5 agents (adjust to your
#   concurrent GPU limit — check with: sacctmgr show qos).
#
#   To run more agents after the first batch finishes:
#         sbatch run_sweep.sh   (again — agents auto-poll W&B)
# ============================================================

#SBATCH --job-name=temu-sweep
#SBATCH -p GPU-shared             # shared GPU partition (not GPU — that's exclusive/expensive)
#SBATCH --qos=gpushared           # required for GPU-shared; wall time cap = 2 days
#SBATCH --gres=gpu:v100-32:1      # 1 V100-32GB per agent
#SBATCH --ntasks-per-node=1       # 1 process per job (not MPI)
#SBATCH --cpus-per-task=5         # 4 dataloader workers + 1 main
#SBATCH --time=40:00:00           # ~40 h = 19 runs × 2 h each (96 configs ÷ 5 agents)
                                  # max for gpushared QOS is 2 days (48 h)
#SBATCH --array=0-4               # ← 5 parallel agents; increase if cluster has headroom
#SBATCH -A cis260031p             # your allocation
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err

# ── Config ────────────────────────────────────────────────────────────────────
SWEEP_ID="uwa7au73"   # e.g. "abc123de"
ENTITY="mrsd-smores"
PROJECT="TEMU"

CONDA_ENV="/ocean/projects/cis260031p/shared/temu_conda"
REPO_DIR="/home/ranai/MRSD/mmml/TactileEncoderForManipulation"
# ──────────────────────────────────────────────────────────────────────────────

echo "=========================================="
echo " Job array ID : $SLURM_ARRAY_JOB_ID"
echo " Agent index  : $SLURM_ARRAY_TASK_ID"
echo " Node         : $(hostname)"
echo " GPU(s)       : $CUDA_VISIBLE_DEVICES"
echo " Start time   : $(date)"
echo "=========================================="

module load anaconda3
conda activate "$CONDA_ENV"

# Make sure log dir exists
mkdir -p "$REPO_DIR/logs"
cd "$REPO_DIR"

# Each array slot is an independent W&B agent.
# The agent polls the sweep controller for the next unfinished config,
# runs train.py with those args, reports results, then loops.
wandb agent "${ENTITY}/${PROJECT}/${SWEEP_ID}"

echo "Agent $SLURM_ARRAY_TASK_ID finished at $(date)"
