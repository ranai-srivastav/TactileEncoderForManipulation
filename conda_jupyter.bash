module load anaconda3
conda deactivate
conda activate /ocean/projects/cis260031p/shared/temu_conda
echo "LOADED CONDA ENV - temu_conda"
python3 -m ipykernel install --user --name temu_conda --display-name "Python (temu_conda)"
jupyter notebook --no-browser --ip=0.0.0.0