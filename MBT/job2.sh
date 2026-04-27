#!/bin/bash
#SBATCH --job-name=mbt-vt-ft
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=8:00:00
#SBATCH --account=cis260031p
#SBATCH --output=logs/%x_%j.out

mkdir -p logs
mkdir -p trained_models/mbt_vt_ft

export PATH=/ocean/projects/cis260031p/shared/temu_conda/bin:$PATH
export TORCH_HOME=/ocean/projects/cis260031p/ayapilla/.cache/torch
export HF_HOME=/ocean/projects/cis260031p/ayapilla/.cache/huggingface

cd /ocean/projects/cis260031p/ayapilla/TactileEncoderForManipulation

python MBT/mbt_train.py --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
  --split object --n_test_objects 5 --modalities V T FT --t3_encoder_domain gs_black --pretrained_dir /ocean/projects/cis260031p/shared/pretrained --seed 42 \
  --adapter_dim 128 --num_bottlenecks 4 --fusion_layer 8 --n_iters 1500 --anneal_iter 1000 --batch_size 4 --grad_accum 8 --lr 1e-4 --weight_decay 0.01 \
  --dropout 0.1 --ogm 1 --ogm_alpha 0.5 --aux_loss_weight 0.0 --wandb_run mbt-ogm-no-aux-v-t-ft-run2 \
  --model_save_path trained_models/mbt_vt_ft/best_mbt_model.pt --L 9