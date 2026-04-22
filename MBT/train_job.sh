#!/bin/bash
#SBATCH --job-name=mbt
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=8:00:00
#SBATCH --account=cis260031p
#SBATCH --output=logs/mbt_8bt_8fl_768Re_T3d_TFtGf.out

mkdir -p logs
module load anaconda3
conda activate /ocean/projects/cis260031p/shared/temu_conda
cd /ocean/projects/cis260031p/srivastr/TactileEncoderForManipulation  ###### CHANGE THIS ###########

python MBT/mbt_train.py \
  --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
  --split random \
  --modalities T FT GF\
  --pretrained_dir /ocean/projects/cis260031p/shared/pretrained \
  --adapter_dim 256 \
  --num_bottlenecks 8 \
  --fusion_layer 8 \
  --n_iters 1500 \
  --anneal_iter 800 \
  --batch_size 4 \
  --grad_accum 8 \
  --lr 5e-4 \
  --weight_decay 0.01 \
  --dropout 0.25 \
  --wandb_run "mbt_8bt_8fl_768Re_T3d_TFtGf" \
  --log_interval 50 \
  --model_save_path "trained_models/mbt_8bt_8fl_768Re_T3d_TFtGf/best_mbt_model.pt" \
  --seed 42

# FOR SIGMA original ratio of stable/unstable is 75/25. 
# 0.5 drops half the stable ones and makes it more 66/33. 
# Change to 1.0 to make it 50/50
