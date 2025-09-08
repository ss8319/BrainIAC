#!/bin/bash
#SBATCH --job-name=preprocess_adni
#SBATCH --output=preprocess_%j.out
#SBATCH --error=preprocess_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=39G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

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

# Verify SimpleITK is available
$BRAINIAC_PYTHON -c "import SimpleITK; print('SimpleITK version:', SimpleITK.Version())" || {
    echo "ERROR: SimpleITK not available in brainiac environment"
    echo "Trying to check what's available..."
    $BRAINIAC_PYTHON -c "import sys; print('Python path:', sys.executable)"
    $BRAINIAC_PYTHON -c "import pkg_resources; print('Installed packages:'); [print(p.project_name, p.version) for p in pkg_resources.working_set if 'simple' in p.project_name.lower()]"
    exit 1
}

# Navigate to script directory
cd /home/ssim0068/code/multimodal-AD/BrainIAC/src/preprocessing

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
$BRAINIAC_PYTHON mri_preprocess_adni.py \
--temp_img /home/ssim0068/code/multimodal-AD/BrainIAC/src/preprocessing/atlases/nihpd_asym_13.0-18.5_t1w.nii \
--input_dir /home/ssim0068/data/ADNI/nifti \
--output_dir /home/ssim0068/data/ADNI/preprocessed \
--resume \
--limit 204