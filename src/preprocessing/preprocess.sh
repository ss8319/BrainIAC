#!/bin/bash
#SBATCH --job-name=preprocess_adni
#SBATCH --output=preprocess_%j.out
#SBATCH --error=preprocess_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=39G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Activate conda environment if needed
source /home/ssim0068/miniconda3/bin/activate brainiac

# Navigate to script directory
cd /home/ssim0068/code/multimodal-AD/BrainIAC/src/preprocessing

# Run the script with CPU configuration
conda run -n brainiac python mri_preprocess_adni.py \
--temp_img /home/ssim0068/code/multimodal-AD/BrainIAC/src/preprocessing/atlases/nihpd_asym_13.0-18.5_t1w.nii \
--input_dir /home/ssim0068/data/ADNI/nifti \
--output_dir /home/ssim0068/data/ADNI/preprocessed
