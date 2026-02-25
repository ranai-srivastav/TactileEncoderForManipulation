#!/bin/bash
#SBATCH --job-name=FirstFullRun_ranais
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks-per-node=5
#SBATCH --time=12:00:00
#SBATCH --account=cis250252p
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── environment ───────────────────────────────────────────────────────────────
module load anaconda3
conda activate /ocean/projects/cis260031p/shared/temu_conda

cd /ocean/projects/cis260031p/srivastr/TactileEncoderForManipulation

# ── training ──────────────────────────────────────────────────────────────────
python train.py \
    --root_dir   /ocean/projects/cis260031p/shared/dataset/Gelsight \
    --split      object \
    --test_objects mug bowl flashlight \
    --sigma      1.0 \
    --batch_size 164 \
    --lr         0.001 \
    --weight_decay 0.01 \
    --hidden_dim 256 \
    --n_iters    6000 \
    --anneal_iter 1000 \
    --modalities V T FT G GF \
    --subsample 1.0 \
    --num_workers 4 \
    --L 9 \
    --wandb_run  FirstFullRun_ranais
