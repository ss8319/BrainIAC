#!/bin/bash
#SBATCH --job-name=train_brainiac_debug
#SBATCH --output=logs/28_2/train_%j.out
#SBATCH --error=logs/28_2/train_%j.err
#SBATCH --time=13:00:00
#SBATCH --mem=39G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

export WANDB_API_KEY=2bd9da9f8c9031d1a7bdddb45f3bdf84f3139346

# Navigate to project directory
cd /home/ssim0068/multimodal-AD

# Use uv to run Python in the project environment
UV_PYTHON="uv run python"


# Navigate to script directory
cd /home/ssim0068/multimodal-AD/src/mri/BrainIAC/src/

# Test GPU availability before running preprocessing
echo "=== GPU Test Before Preprocessing ==="
$UV_PYTHON -c "
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

# Run the script using uv
echo "=== Starting Preprocessing ==="
$UV_PYTHON train_lightning_mci.py \
--config config_adni_ft.yml
