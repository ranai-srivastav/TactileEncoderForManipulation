#!/bin/bash
#SBATCH --job-name=mbt
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=8:00:00
#SBATCH --account=cis260031p
#SBATCH --output=logs/%x_%j.out

mkdir -p logs
module load anaconda3
conda activate /ocean/projects/cis260031p/shared/temu_conda
cd /ocean/projects/cis260031p/srivastr/TactileEncoderForManipulation  ###### CHANGE THIS ###########

python MBT/mbt_train.py \
  --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
  --split object \
  --n_test_objects 5 \
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
  --wandb_run "train_1sigPretrain" \
  --model_save_path "trained_models/train_1sigPretrain" \
  --L 9 \
  --seed 42

# FOR SIGMA original ratio of stable/unstable is 75/25. 
# 0.5 drops half the stable ones and makes it more 66/33. 
# Change to 1.0 to make it 50/50
