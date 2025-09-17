#!/usr/bin/env python3
"""
Convert ADNI data to BrainIAC format
- Match metadata.csv entries to NIfTI files
- Create train/val/test splits with balanced sex and AD/CN ratios
- Optionally copy files to new directory structure
- Generate CSV files for each split

Usage:
    python convert_adni_to_brainiac_v2.py [--copy-files]
    
Arguments:
    --copy-files: Copy NIfTI files to new directory structure (default: False)
"""

import pandas as pd
import os
import shutil
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

def analyze_metadata(metadata_path):
    """Analyze the metadata to understand data distribution"""
    print("=== Analyzing Metadata ===")
    df = pd.read_csv(metadata_path)
    
    print(f"Total entries: {len(df)}")
    print(f"Unique subjects: {df['Subject'].nunique()}")
    
    # Group distribution
    group_counts = df['Group'].value_counts()
    print(f"\nGroup distribution:")
    for group, count in group_counts.items():
        print(f"  {group}: {count}")
    
    # Sex distribution
    sex_counts = df['Sex'].value_counts()
    print(f"\nSex distribution:")
    for sex, count in sex_counts.items():
        print(f"  {sex}: {count}")
    
    # Sex distribution by group
    print(f"\nSex distribution by group:")
    sex_group = df.groupby(['Group', 'Sex']).size().unstack(fill_value=0)
    print(sex_group)
    
    return df

def match_files_to_metadata(df, nifti_base_dir):
    """Match metadata entries to actual NIfTI files"""
    print("\n=== Matching Files to Metadata ===")
    
    matched_data = []
    missing_files = []
    
    for idx, row in df.iterrows():
        # Extract path components from ScanPath
        scan_path = row['ScanPath']
        # Convert Windows path separators to Unix
        scan_path = scan_path.replace('\\', '/')
        
        # Remove the Image ID part (e.g., \I303066)
        if '/' in scan_path:
            path_parts = scan_path.split('/')
            if len(path_parts) >= 3:
                subject = path_parts[0]
                scan_type = path_parts[1]
                timestamp = path_parts[2]
                
                # Construct the NIfTI file path
                nifti_path = os.path.join(nifti_base_dir, subject, scan_type, timestamp, f"{timestamp}.nii.gz")
                
                if os.path.exists(nifti_path):
                    matched_data.append({
                        'pat_id': subject,
                        'label': 1 if row['Group'] == 'AD' else 0,  # AD=1, CN=0
                        'Sex': row['Sex'],
                        'Age': row['Age'],
                        'nifti_path': nifti_path,
                        'original_scan_path': scan_path
                    })
                else:
                    missing_files.append({
                        'subject': subject,
                        'expected_path': nifti_path,
                        'scan_path': scan_path
                    })
    
    print(f"Successfully matched: {len(matched_data)} files")
    print(f"Missing files: {len(missing_files)}")
    
    if missing_files:
        print("\nFirst 5 missing files:")
        for missing in missing_files[:5]:
            print(f"  {missing['expected_path']}")
    
    return pd.DataFrame(matched_data), missing_files

def create_balanced_splits(df, test_size=0.15, val_size=0.15, random_state=42):
    """Create train/val/test splits with balanced sex ratios AND label ratios"""
    print("\n=== Creating Balanced Splits ===")
    
    # Calculate target ratios from the full dataset
    sex_counts = df['Sex'].value_counts()
    label_counts = df['label'].value_counts()
    total_samples = len(df)
    
    target_male_ratio = sex_counts['M'] / total_samples
    target_female_ratio = sex_counts['F'] / total_samples
    target_cn_ratio = label_counts[0] / total_samples
    target_ad_ratio = label_counts[1] / total_samples
    
    print(f"Target sex ratios - Male: {target_male_ratio:.3f}, Female: {target_female_ratio:.3f}")
    print(f"Target label ratios - CN (0): {target_cn_ratio:.3f}, AD (1): {target_ad_ratio:.3f}")
    
    # Create stratification based on both sex AND label
    df['stratify_col'] = df['Sex'].astype(str) + '_' + df['label'].astype(str)
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['stratify_col']  # Stratify by both sex and label
    )
    
    # Second split: separate train and val from remaining data
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=train_val_df['stratify_col']  # Stratify by both sex and label
    )
    
    # Verify sex balance in each split
    splits = {'train': train_df, 'val': val_df, 'test': test_df}
    
    print(f"\nSplit sizes:")
    for split_name, split_df in splits.items():
        print(f"  {split_name}: {len(split_df)} samples")
        
        # Check sex balance
        sex_dist = split_df['Sex'].value_counts()
        male_ratio = sex_dist.get('M', 0) / len(split_df)
        female_ratio = sex_dist.get('F', 0) / len(split_df)
        
        print(f"    Male: {sex_dist.get('M', 0)} ({male_ratio:.3f})")
        print(f"    Female: {sex_dist.get('F', 0)} ({female_ratio:.3f})")
        
        # Check label balance
        label_dist = split_df['label'].value_counts()
        print(f"    CN (0): {label_dist.get(0, 0)}")
        print(f"    AD (1): {label_dist.get(1, 0)}")
    
    return train_df, val_df, test_df

def copy_files_and_create_csvs(train_df, val_df, test_df, output_base_dir, copy_files=False):
    """Copy NIfTI files (if requested) and create CSV files for each split"""
    print("\n=== Creating CSVs and Copying Files ===")
    
    # Create output directories
    csv_dir = os.path.join(output_base_dir, 'csvs')
    os.makedirs(csv_dir, exist_ok=True)
    
    images_dir = None
    if copy_files:
        images_dir = os.path.join(output_base_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        print(f"NIfTI files will be copied to: {images_dir}")
    else:
        print("NIfTI files will NOT be copied (use --copy-files to enable)")
    
    # Process each split
    splits = {
        'train': train_df,
        'val': val_df, 
        'test': test_df
    }
    
    for split_name, split_df in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        # Create CSV for this split
        csv_data = split_df[['pat_id', 'label', 'Sex', 'Age']].copy()
        csv_path = os.path.join(csv_dir, f'mci_{split_name}.csv')
        csv_data.to_csv(csv_path, index=False)
        print(f"  Created CSV: {csv_path}")
        
        # Copy NIfTI files if requested
        if copy_files and images_dir:
            copied_count = 0
            for idx, row in split_df.iterrows():
                source_path = row['nifti_path']
                # Use pat_id as filename instead of timestamp
                filename = f"{row['pat_id']}.nii.gz"
                dest_path = os.path.join(images_dir, filename)
                
                if not os.path.exists(dest_path):
                    shutil.copy2(source_path, dest_path)
                    copied_count += 1
            
            print(f"  Copied {copied_count} new NIfTI files")
            print(f"  Total files in directory: {len(os.listdir(images_dir))}")

def main():
    parser = argparse.ArgumentParser(description="Convert ADNI data to BrainIAC format")
    parser.add_argument("--copy-files", action="store_true", 
                       help="Copy NIfTI files to new directory structure")
    parser.add_argument("--metadata-path", type=str, 
                       default="/home/ssim0068/data/AD_CN_train_v2/metadata.csv",
                       help="Path to metadata CSV file")
    parser.add_argument("--nifti-base-dir", type=str,
                       default="/home/ssim0068/data/v2_preprocessed_icbm152",
                       help="Base directory containing NIfTI files")
    parser.add_argument("--output-base-dir", type=str,
                       default="/home/ssim0068/data/ADNI_v2_icbm152",
                       help="Base directory for output files")
    parser.add_argument("--test-size", type=float, default=0.15,
                       help="Test split size (default: 0.15)")
    parser.add_argument("--val-size", type=float, default=0.15,
                       help="Validation split size (default: 0.15)")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random state for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    print("ADNI to BrainIAC Data Conversion")
    print("=" * 50)
    print(f"Metadata: {args.metadata_path}")
    print(f"NIfTI base: {args.nifti_base_dir}")
    print(f"Output base: {args.output_base_dir}")
    print(f"Copy files: {args.copy_files}")
    print(f"Test size: {args.test_size}, Val size: {args.val_size}")
    print(f"Random state: {args.random_state}")
    print()
    
    # Step 1: Analyze metadata
    df = analyze_metadata(args.metadata_path)
    
    # Step 2: Match files to metadata
    matched_df, missing_files = match_files_to_metadata(df, args.nifti_base_dir)
    
    if len(matched_df) == 0:
        print("No files matched! Exiting.")
        return
    
    # Step 3: Create balanced splits
    train_df, val_df, test_df = create_balanced_splits(
        matched_df, 
        test_size=args.test_size, 
        val_size=args.val_size, 
        random_state=args.random_state
    )
    
    # Step 4: Copy files and create CSVs
    copy_files_and_create_csvs(train_df, val_df, test_df, args.output_base_dir, args.copy_files)
    
    print("\n=== Summary ===")
    print(f"Output directory: {args.output_base_dir}")
    print(f"Total processed: {len(matched_df)} files")
    print(f"Train: {len(train_df)} files")
    print(f"Validation: {len(val_df)} files") 
    print(f"Test: {len(test_df)} files")
    print(f"Missing files: {len(missing_files)}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
