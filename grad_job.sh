#!/bin/bash
#SBATCH --job-name=gradcam
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=4:00:00
#SBATCH --account=cis260031p
#SBATCH --output=logs/%x_%j.out

mkdir -p logs
module load anaconda3/2022.10
conda activate /ocean/projects/cis260031p/shared/temu_conda
cd /ocean/projects/cis260031p/mlee12/TactileEncoderForManipulation

MODALITIES="$@"
MOD_FLAG=$(echo $MODALITIES | tr ' ' '-')
CKPT="trained_models/best_model_${MOD_FLAG}.pt"
VIS_DIR="pfs_vis_${MOD_FLAG}"
ROOT=/ocean/projects/cis260031p/shared/dataset/Gelsight
L=5

python train.py --root_dir $ROOT \
  --split random --modalities $MODALITIES --n_iters 1000 --drs_iter 9999 \
  --batch_size 32 --lr 0.005 --L $L \
  --model_save_path $CKPT \
  --wandb_run "pfs-${MOD_FLAG}"

# Determine gradcam_mode from modalities
case "$MODALITIES" in
  *V*T*|*T*V*) GCAM_MODE=both ;;
  *V*)         GCAM_MODE=rgb ;;
  *T*)         GCAM_MODE=tactile ;;
  *)           GCAM_MODE=both ;;
esac

python gradcam_metric.py --root_dir $ROOT \
  --checkpoint $CKPT \
  --modalities $MODALITIES \
  --gradcam_mode $GCAM_MODE \
  --steps 10 --n_samples 50 --L $L \
  --vis_dir $VIS_DIR
