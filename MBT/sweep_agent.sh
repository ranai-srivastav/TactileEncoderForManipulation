#!/bin/bash
#SBATCH --job-name=mbt-sweep
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=8:00:00
#SBATCH --account=cis260031p
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err

mkdir -p logs
module load anaconda3
conda activate /ocean/projects/cis260031p/shared/temu_conda
cd /ocean/projects/cis260031p/srivastr/TactileEncoderForManipulation

# SWEEP_ID is passed via --export from launch_sweep.sh
wandb agent mrsd-smores/TEMU/${SWEEP_ID}