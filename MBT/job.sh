#!/bin/bash
#SBATCH --job-name=mbt
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=4:00:00
#SBATCH --account=cis260031p
#SBATCH --output=logs/%x_%j.out

mkdir -p logs
module load anaconda3/2022.10
conda activate /ocean/projects/cis260031p/shared/temu_conda
cd /ocean/projects/cis260031p/mlee12/TactileEncoderForManipulation

# MBT — 5-modality bottleneck fusion
python MBT/train.py --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
  --split random --modalities V T FT G GF --sigma 0.5 --n_iters 600 --anneal_iter 300 \
  --batch_size 32 --lr 1e-4 --wandb_run mbt --L 9