#!/bin/bash
#SBATCH --job-name=mbt-test1
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
  --split object \
  --test_object_ids 0 5 10 15 \
  --modalities V T FT G GF \
  --t3_encoder_domain gs_black \
  --pretrained_dir /ocean/projects/cis260031p/shared/pretrained \
  --adapter_dim 128 \
  --num_bottlenecks 4 \
  --fusion_layer 8 \
  --n_iters 1500 \
  --anneal_iter 1000 \
  --batch_size 4 \
  --grad_accum 8 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --dropout 0.1 \
  --wandb_run mbt_t3_obj \
  --L 9 \
  --sigma 0.5 \
  --seed 42

# FOR SIGMA original ratio of stable/unstable is 75/25. 
# 0.5 drops half the stable ones and makes it more 66/33. 
# Change to 1.0 to make it 50/50