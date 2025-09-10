#!/usr/bin/env python3
"""
Convert ADNI dataset to BrainIAC format for CN vs AD classification.

This script:
1. Reads ADNI train_split.csv and test_split.csv
2. Handles multiple scans per subject (randomly selects one)
3. Copies NIfTI files to BrainIAC structure
4. Creates train/val/test CSV files in BrainIAC format
5. Generates config file for training
"""

import pandas as pd
import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml

def find_available_scans(subject_id, adni_base_path):
    """
    Find all available scans for a subject.
    
    Returns:
        List of (scan_type, scan_date, nifti_path) tuples
    """
    subject_path = Path(adni_base_path) / subject_id
    available_scans = []
    
    if not subject_path.exists():
        return available_scans
    
    # Iterate through scan types (MPRAGE, MPRAGE_REPE, etc.)
    for scan_type_dir in subject_path.iterdir():
        if scan_type_dir.is_dir():
            # Iterate through scan dates
            for scan_date_dir in scan_type_dir.iterdir():
                if scan_date_dir.is_dir():
                    scan_date = scan_date_dir.name
                    nifti_file = scan_date_dir / f"{scan_date}.nii.gz"
                    
                    if nifti_file.exists():
                        available_scans.append((
                            scan_type_dir.name,
                            scan_date,
                            str(nifti_file)
                        ))
    
    return available_scans

def convert_adni_to_brainiac(
    train_csv_path,
    test_csv_path,
    adni_base_path,
    output_base_path,
    val_split=0.2,
    random_seed=42
):
    """
    Convert ADNI dataset to BrainIAC format.
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Create output directories
    output_path = Path(output_base_path)
    images_dir = output_path / "images"
    csvs_dir = output_path / "csvs"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    csvs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Created directories:")
    print(f"  • Images: {images_dir}")
    print(f"  • CSVs: {csvs_dir}")
    
    # Read ADNI CSVs
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    print(f"\n📊 Loaded data:")
    print(f"  • Train samples: {len(train_df)}")
    print(f"  • Test samples: {len(test_df)}")
    
    def process_subjects(df, split_name):
        """Process subjects and create BrainIAC format data."""
        brainiac_data = []
        copied_files = 0
        errors = []
        
        # Group by subject to handle multiple scans per subject
        subject_groups = df.groupby('Subject')
        
        for subject_id, group in subject_groups:
            # Find all available scans for this subject
            available_scans = find_available_scans(subject_id, adni_base_path)
            
            if not available_scans:
                error_msg = f"No scans found for subject {subject_id}"
                errors.append(error_msg)
                print(f"  ❌ {error_msg}")
                continue
            
            # Randomly select one scan if multiple available
            selected_scan = random.choice(available_scans)
            scan_type, scan_date, source_nifti_path = selected_scan
            
            # Get group label from any row (all rows for same subject have same group)
            group_label = group.iloc[0]['Group']
            label = 1 if group_label == 'AD' else 0
            
            # Create patient ID
            pat_id = subject_id
            
            # Target file path
            target_nifti_path = images_dir / f"{pat_id}.nii.gz"
            
            try:
                # Copy NIfTI file
                shutil.copy2(source_nifti_path, target_nifti_path)
                copied_files += 1
                
                # Add to BrainIAC data
                brainiac_data.append({
                    'pat_id': pat_id,
                    'label': label,
                    'group': group_label,
                    'scan_type': scan_type,
                    'scan_date': scan_date,
                    'source_path': source_nifti_path
                })
                
                print(f"  ✓ {pat_id}: {group_label} ({scan_type}, {scan_date})")
                
            except Exception as e:
                error_msg = f"Failed to copy {source_nifti_path}: {e}"
                errors.append(error_msg)
                print(f"  ❌ {error_msg}")
        
        print(f"\n📊 {split_name} processing summary:")
        print(f"  • Subjects processed: {len(subject_groups)}")
        print(f"  • Files copied: {copied_files}")
        print(f"  • Errors: {len(errors)}")
        
        if errors:
            print("❌ Errors encountered:")
            for error in errors:
                print(f"    {error}")
            raise RuntimeError(f"Encountered {len(errors)} errors during processing")
        
        return pd.DataFrame(brainiac_data)
    
    # Process train and test data
    print(f"\n🔄 Processing training data...")
    train_brainiac_df = process_subjects(train_df, "Training")
    
    print(f"\n🔄 Processing test data...")
    test_brainiac_df = process_subjects(test_df, "Test")
    
    # Create balanced train/val split manually
    print(f"\n📊 Creating balanced train/val split...")
    
    # Separate CN and AD subjects
    cn_subjects = train_brainiac_df[train_brainiac_df['label'] == 0]
    ad_subjects = train_brainiac_df[train_brainiac_df['label'] == 1]
    
    total_cn = len(cn_subjects)
    total_ad = len(ad_subjects)
    total_subjects = len(train_brainiac_df)
    
    print(f"📊 Available subjects: CN={total_cn}, AD={total_ad}, Total={total_subjects}")
    
    # Calculate validation set size and balance
    val_size = int(total_subjects * val_split)
    
    # For balanced validation: try to get equal CN and AD if possible
    if val_size % 2 == 0:
        # Even number: split equally
        val_cn_target = val_size // 2
        val_ad_target = val_size // 2
    else:
        # Odd number: one more CN (since we typically have more CN)
        val_cn_target = (val_size + 1) // 2
        val_ad_target = val_size // 2
    
    # Adjust if we don't have enough samples of one class
    val_cn_actual = min(val_cn_target, total_cn)
    val_ad_actual = min(val_ad_target, total_ad)
    
    # If we can't get the target for one class, adjust the other
    remaining_val_slots = val_size - val_cn_actual - val_ad_actual
    if remaining_val_slots > 0:
        if val_cn_actual < total_cn:
            val_cn_actual = min(val_cn_actual + remaining_val_slots, total_cn)
        elif val_ad_actual < total_ad:
            val_ad_actual = min(val_ad_actual + remaining_val_slots, total_ad)
    
    print(f"🎯 Validation targets: CN={val_cn_actual}, AD={val_ad_actual}, Total={val_cn_actual + val_ad_actual}")
    
    # Randomly select validation subjects
    random.seed(random_seed)
    val_cn_subjects = cn_subjects.sample(n=val_cn_actual, random_state=random_seed)
    val_ad_subjects = ad_subjects.sample(n=val_ad_actual, random_state=random_seed)
    
    # Combine validation subjects
    val_subjects_df = pd.concat([val_cn_subjects, val_ad_subjects])
    val_ids = val_subjects_df['pat_id'].values
    
    # Remaining subjects go to training
    train_ids = train_brainiac_df[~train_brainiac_df['pat_id'].isin(val_ids)]['pat_id'].values
    
    # Verify the split
    final_val_labels = val_subjects_df['label'].value_counts().sort_index()
    final_train_labels = train_brainiac_df[train_brainiac_df['pat_id'].isin(train_ids)]['label'].value_counts().sort_index()
    
    print(f"✅ Final split:")
    print(f"   Training: {dict(final_train_labels)} (total: {len(train_ids)})")
    print(f"   Validation: {dict(final_val_labels)} (total: {len(val_ids)})")
    
    # Calculate balance percentages
    val_cn_pct = final_val_labels.get(0, 0) / len(val_ids) * 100
    val_ad_pct = final_val_labels.get(1, 0) / len(val_ids) * 100
    print(f"   Validation balance: CN={val_cn_pct:.1f}%, AD={val_ad_pct:.1f}%")
    
    # Create train/val splits
    final_train_df = train_brainiac_df[train_brainiac_df['pat_id'].isin(train_ids)]
    val_df = train_brainiac_df[train_brainiac_df['pat_id'].isin(val_ids)]
    
    # Create BrainIAC format CSVs (only pat_id and label columns)
    def save_brainiac_csv(df, filename):
        brainiac_csv = df[['pat_id', 'label']].copy()
        csv_path = csvs_dir / filename
        brainiac_csv.to_csv(csv_path, index=False)
        
        # Print summary
        label_counts = brainiac_csv['label'].value_counts()
        group_counts = df['group'].value_counts()
        print(f"  ✓ {filename}: {len(brainiac_csv)} samples")
        print(f"    Labels (0=CN, 1=AD): {dict(label_counts)}")
        print(f"    Groups: {dict(group_counts)}")
        
        return csv_path
    
    print(f"\n💾 Saving CSV files...")
    train_csv_path = save_brainiac_csv(final_train_df, "mci_train.csv")
    val_csv_path = save_brainiac_csv(val_df, "mci_val.csv")
    test_csv_path = save_brainiac_csv(test_brainiac_df, "mci_test.csv")
    
    # Create config file
    config = {
        'model': {
            'max_epochs': 50
        },
        'data': {
            'size': [128, 128, 128],
            'batch_size': 16,
            'num_workers': 4,
            'csv_file': str(train_csv_path),
            'val_csv': str(val_csv_path),
            'root_dir': str(images_dir)
        },
        'simclrvit': {
            'ckpt_path': '/home/ssim0068/code/multimodal-AD/BrainIAC/src/checkpoints/BrainIAC.ckpt'
        },
        'optim': {
            'lr': 0.001,
            'momentum': 0.9,
            'weight_decay': 0.0001,
            'clr': 'no'
        },
        'logger': {
            'save_dir': '/home/ssim0068/code/multimodal-AD/BrainIAC/checkpoints',
            'save_name': 'adni_cn_ad_linear_probe-{epoch:02d}-{val_auc:.2f}',
            'run_name_mci': 'adni_cn_ad_linear_probe',
            'project_name_mci': 'adni_cn_ad_classification'
        },
        'gpu': {
            'visible_device': '0',
            'devices': 1
        },
        'trainer': {
            'strategy': 'ddp_find_unused_parameters_true',
            'precision': '16-mixed'
        },
        'train': {
            'freeze': 'yes',  # Linear probe
            'finetune': 'no'
        }
    }
    
    config_path = output_path / "config_adni_cn_ad.yml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"\n⚙️ Config saved to: {config_path}")
    
    # Final summary
    total_images = len(list(images_dir.glob("*.nii.gz")))
    print(f"\n✅ Conversion complete!")
    print(f"📁 Output directory: {output_path}")
    print(f"🖼️ Total images copied: {total_images}")
    print(f"📄 CSV files created: 3 (train, val, test)")
    print(f"⚙️ Config file created: config_adni_cn_ad.yml")
    
    print(f"\n🚀 To run training:")
    print(f"cd /home/ssim0068/code/multimodal-AD/BrainIAC/src")
    print(f"python train_lightning_mci.py --config {config_path}")
    
    return {
        'train_csv': train_csv_path,
        'val_csv': val_csv_path, 
        'test_csv': test_csv_path,
        'config_path': config_path,
        'images_dir': images_dir
    }

if __name__ == "__main__":
    # Paths
    train_csv = "/home/ssim0068/code/multimodal-AD/AD_CN/proteomics/Biomarkers Consortium Plasma Proteomics MRM/MRI/splits/train_split.csv"
    test_csv = "/home/ssim0068/code/multimodal-AD/AD_CN/proteomics/Biomarkers Consortium Plasma Proteomics MRM/MRI/splits/test_split.csv"
    adni_base = "/home/ssim0068/data/ADNI/preprocessed_adult"
    output_base = "/home/ssim0068/data/ADNI/preprocessed_brainiac"
    
    # Run conversion
    result = convert_adni_to_brainiac(
        train_csv_path=train_csv,
        test_csv_path=test_csv,
        adni_base_path=adni_base,
        output_base_path=output_base,
        val_split=0.2,
        random_seed=42
    )