#!/bin/bash
#SBATCH --job-name=mbt
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=8:00:00
#SBATCH --account=cis260031p
#SBATCH --output=logs/%x_%j.out

mkdir -p logs
module load anaconda3/2022.10
conda activate /ocean/projects/cis260031p/shared/temu_conda
cd /ocean/projects/cis260031p/mlee12/TactileEncoderForManipulation

python MBT/mbt_train.py \
  --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
  --split random \
  --modalities V T FT G GF \
  --t3_encoder_domain gs_black \
  --pretrained_dir /ocean/projects/cis260031p/shared/pretrained \
  --adapter_dim 128 \
  --num_bottlenecks 4 \
  --fusion_layer 8 \
  --n_iters 1500 \
  --anneal_iter 1000 \
  --drs_iter 99999 \
  --batch_size 4 \
  --grad_accum 8 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --wandb_run mbt_t3 \
  --L 9
