#!/bin/bash
# Example launch script for the vanilla transformer.
# Edit paths / resources / arguments as needed.

set -euo pipefail

#SBATCH --job-name=vanilla-transformer
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=8:00:00
#SBATCH --account=cis260031p
#SBATCH --output=logs/%x_%j.out

mkdir -p logs

module load anaconda3/2022.10
conda activate /ocean/projects/cis260031p/shared/temu_conda

cd /ocean/projects/cis260031p/mlee12/TactileEncoderForManipulation

# To resume from W&B, add:
#   --resume_wandb_artifact entity/project/artifact-name:latest
python VanillaTransformer/transformer-train.py \
  --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
  --split object \
  --test_object_ids 1 \
  --batch_size 1 \
  --num_workers 4 \
  --FRGB 2 \
  --FTactile 8 \
  --FFT 8 \
  --FGripper 1 \
  --L 0 \
  --seed 42 \
  --hidden_dim 768 \
  --depth 4 \
  --num_heads 8 \
  --mlp_ratio 4.0 \
  --dropout 0.1 \
  --modalities V T FT G \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --epochs 20 \
  --model_save_path trained_models/vanilla_transformer_best.pt \
  --wandb_project TEMU \
  --wandb_run vanilla-transformer \
  --wandb_entity mrsd-smores
