#!/usr/bin/env python3
"""
Single image inference script for BrainIAC feature extraction.

This script loads a single NIfTI image, runs it through the pretrained BrainIAC model,
and extracts latent features for analysis.

Usage:
    python single_image_inference.py --image_path /path/to/image.nii.gz --output_path /path/to/features.csv
"""

import torch
import numpy as np
import pandas as pd
import argparse
import os
import shutil
from pathlib import Path
from dataset import BrainAgeDataset, get_validation_transform
from load_brainiac import load_brainiac

def create_temp_csv(image_path):
    """
    Create a temporary CSV file for the single image to work with BrainAgeDataset.
    
    Args:
        image_path (str): Path to the NIfTI image
    
    Returns:
        str: Path to temporary CSV file
    """
    # Extract patient ID from path (e.g., 002_S_0413 from the full path)
    image_path = Path(image_path)
    pat_id = image_path.parent.parent.parent.name  # Gets 002_S_0413 from the nested structure
    
    # Create temporary CSV data (label is not needed for feature extraction)
    temp_data = pd.DataFrame({
        'pat_id': [pat_id],
        'label': [0]  # Dummy label, not used for feature extraction
    })
    
    # Save to temporary CSV
    temp_csv_path = f"/tmp/temp_single_image_{pat_id}.csv"
    temp_data.to_csv(temp_csv_path, index=False)
    
    return temp_csv_path

def extract_features_single_image(model, image_path, device='cuda'):
    """
    Extract BrainIAC features from a single NIfTI image.
    
    Args:
        model: Loaded BrainIAC model
        image_path (str): Path to the NIfTI image
        device (str): Device to run inference on
    
    Returns:
        np.ndarray: Extracted features
    """
    # Create temporary CSV for the single image
    temp_csv_path = create_temp_csv(image_path)
    
    try:
        # Create dataset with validation transforms (no augmentation)
        # Use the actual image directory as root_dir and create a symlink
        image_path_obj = Path(image_path)
        pat_id = image_path_obj.parent.parent.parent.name
        
        # Create a temporary directory with the expected structure
        temp_dir = f"/tmp/temp_brainiac_{pat_id}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create symlink to the actual image file
        temp_image_path = os.path.join(temp_dir, f"{pat_id}.nii.gz")
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        os.symlink(image_path, temp_image_path)
        
        dataset = BrainAgeDataset(
            csv_path=temp_csv_path,
            root_dir=temp_dir,
            transform=get_validation_transform()
        )
        
        # Get the single sample
        sample = dataset[0]
        image = sample['image'].unsqueeze(0).to(device)  # Add batch dimension
        
        # Run inference
        model.eval()
        with torch.no_grad():
            features = model(image)
            features_numpy = features.cpu().numpy().flatten()
        
        return features_numpy
        
    finally:
        # Clean up temporary files
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(description='Extract BrainIAC features from a single NIfTI image')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Path to the NIfTI image file')
    parser.add_argument('--checkpoint', type=str, 
                      default='/home/ssim0068/code/multimodal-AD/BrainIAC/src/checkpoints/BrainIAC.ckpt',
                      help='Path to BrainIAC checkpoint (default: checkpoints/BrainIAC.ckpt)')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to save the extracted features CSV')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device to run inference on (cuda or cpu)')
    parser.add_argument('--label', type=int, default=None,
                      help='Optional label for the image (0=CN, 1=AD) - only used for output metadata')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    print(f"🖼️  Loading image: {args.image_path}")
    print(f"🧠 Loading BrainIAC model from: {args.checkpoint}")
    
    # Load BrainIAC model
    model = load_brainiac(args.checkpoint, args.device)
    print("✅ Model loaded successfully!")
    
    # Extract features
    print("🔄 Extracting features...")
    features = extract_features_single_image(model, args.image_path, args.device)
    
    # Create output DataFrame
    feature_columns = [f'Feature_{i}' for i in range(len(features))]
    features_df = pd.DataFrame([features], columns=feature_columns)
    features_df['pat_id'] = Path(args.image_path).parent.parent.parent.name
    features_df['image_path'] = args.image_path
    
    # Add label only if provided
    if args.label is not None:
        features_df['label'] = args.label
    
    # Save features
    features_df.to_csv(args.output_path, index=False)
    
    print(f"✅ Features extracted successfully!")
    print(f"📊 Feature shape: {features.shape}")
    print(f"💾 Saved to: {args.output_path}")
    print(f"📋 Patient ID: {features_df['pat_id'].iloc[0]}")
    if args.label is not None:
        print(f"🏷️  Label: {args.label} ({'CN' if args.label == 0 else 'AD'})")
    else:
        print("🏷️  No label provided (feature extraction only)")

if __name__ == "__main__":
    main()
