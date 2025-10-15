#!/bin/bash
#SBATCH --job-name=test_inference
#SBATCH --output=logs/test_inference_%j.out
#SBATCH --error=logs/test_inference_%j.err
#SBATCH --time=13:00:00
#SBATCH --mem=39G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4


# Initialize uv
eval "$(uv run --help > /dev/null 2>&1 && echo 'uv available' || echo 'uv not found')"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. Please install uv first."
    exit 1
fi

echo "Successfully found uv environment manager"

# Use uv to run python commands
UV_PYTHON="uv run python"


# Navigate to script directory
cd /home/ssim0068/code/multimodal-AD/BrainIAC/src/

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

# Run the script directly using uv
echo "=== Starting Preprocessing ==="
$UV_PYTHON test_inference_finetune.py \

