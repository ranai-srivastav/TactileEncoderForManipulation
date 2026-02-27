#!/bin/bash
#SBATCH --job-name=multi-run
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=4:00:00
#SBATCH --account=cis260031p
#SBATCH --output=logs/%x_%j.out

mkdir -p logs
module load anaconda3/2022.10
conda activate /ocean/projects/cis260031p/shared/temu_conda
cd /ocean/projects/cis260031p/psingh13/TactileEncoderForManipulation

# Run 1: V T R LSTM bidirectional 2 layer
python train.py --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
  --split random --modalities V T R --sigma 0.5 --n_iters 600 --anneal_iter 300 \
  --hidden_dim 500 --batch_size 200 --wandb_run v-t-r --L 9

# Run 2: V T LSTM 6 layer bidirectional
python train.py --root_dir /ocean/projects/cis260031p/shared/dataset/Gelsight \
  --split random --modalities V T --sigma 0.5 --n_iters 600 --anneal_iter 300 \
  --hidden_dim 500 --batch_size 200 --lstm_layers 6 --wandb_run v-t-6layer --L 9
