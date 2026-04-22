#!/bin/bash
# Usage: bash MBT/launch_sweep.sh [sweep-id] [n_jobs] [max_parallel]
#
# Creates a wandb sweep from sweep.yaml (if no sweep-id given),
# then submits a SLURM job array capped at max_parallel concurrent agents.
#
# Examples:
#   bash MBT/launch_sweep.sh                           # creates sweep, 36 jobs, 5 at a time
#   bash MBT/launch_sweep.sh abc123xyz                 # reuse sweep, 36 jobs, 5 at a time
#   bash MBT/launch_sweep.sh abc123xyz 36 10           # reuse sweep, 36 jobs, 10 at a time

set -e
cd "$(dirname "$0")/.."   # repo root

SWEEP_ID=${1:-""}
N_JOBS=${2:-36}           # 3 bottlenecks × 3 fusion_layers × 4 modality_sets
MAX_PARALLEL=${3:-5}      # max simultaneous SLURM jobs

if [[ -z "$SWEEP_ID" ]]; then
    echo "Creating sweep from MBT/sweep.yaml ..."
    SWEEP_OUT=$(wandb sweep MBT/sweep.yaml 2>&1)
    echo "$SWEEP_OUT"
    SWEEP_ID=$(echo "$SWEEP_OUT" | grep -oP '(?<=with ID: )\S+')
    if [[ -z "$SWEEP_ID" ]]; then
        echo "ERROR: could not parse sweep ID from wandb output."
        exit 1
    fi
    echo "Sweep ID: $SWEEP_ID"
fi

echo "Submitting array of $N_JOBS jobs (max $MAX_PARALLEL concurrent) for sweep $SWEEP_ID ..."
sbatch --array=1-${N_JOBS}%${MAX_PARALLEL} \
       --export=ALL,SWEEP_ID=$SWEEP_ID \
       MBT/sweep_agent.sh

echo "Done. Monitor at: https://wandb.ai/mrsd-smores/TEMU/sweeps/$SWEEP_ID"