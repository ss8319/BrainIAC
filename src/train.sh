#!/bin/bash
#SBATCH --job-name=train_brainiac_debug
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=13:00:00
#SBATCH --mem=39G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

export WANDB_API_KEY=2bd9da9f8c9031d1a7bdddb45f3bdf84f3139346

# Initialize conda
eval "$(/home/ssim0068/miniconda3/bin/conda shell.bash hook)"

# Activate conda environment
conda activate brainiac

# Check if conda environment was activated successfully
if [[ "$CONDA_DEFAULT_ENV" != "brainiac" ]]; then
    echo "ERROR: Failed to activate conda environment 'brainiac'"
    echo "Current environment: $CONDA_DEFAULT_ENV"
    exit 1
fi

echo "Successfully activated conda environment: $CONDA_DEFAULT_ENV"

# Use the full path to python in the brainiac environment
BRAINIAC_PYTHON="/home/ssim0068/miniconda3/envs/brainiac/bin/python"


# Navigate to script directory
cd /home/ssim0068/code/multimodal-AD/BrainIAC/src/

# Test GPU availability before running preprocessing
echo "=== GPU Test Before Preprocessing ==="
$BRAINIAC_PYTHON -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('CUDA not available - will use CPU')
"

# Run the script directly using the brainiac python
echo "=== Starting Preprocessing ==="
$BRAINIAC_PYTHON train_lightning_mci.py \
--config config_adni_cn_ad.yml
