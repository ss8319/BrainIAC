# Overall Survival Prediction

<p align="left">
  <img src="../../pictures/os.jpeg" width="200" alt="Overall Survival Prediction Example"/>
</p>

## Overview

This module predicts overall survival for GBM patients using multiple MRI sequences (T1CE, T1w, T2w, and FLAIR). The model has been trained on [dataset details] and achieves [performance metrics].

## Data Requirements

- **Input**: T1CE, T1w, T2w, and FLAIR MRI scans
- **Format**: NIFTI (.nii.gz)
- **Preprocessing**: Bias field corrected, registered to standard space, skull stripped
- **CSV Structure**:
  ```
  pat_id,scandate,label
  subject001,20240101,365    # survival time in days
  ```

## Usage

### Using Docker

```bash
docker pull brainiac/survival:latest
docker run -v /path/to/data:/data brainiac/survival:latest [args]
```

### Manual Setup

1. **Configuration**:
   ```yaml
   # config.yml
   data:
     train_csv: "path/to/train.csv"
     val_csv: "path/to/val.csv"
     test_csv: "path/to/test.csv"
     root_dir: "path/to/preprocessed/scans"
     collate: 4  # multi-sequence framework
   ```

2. **Training**:
   ```bash
   python train_os.py
   ```

3. **Inference**:
   ```bash
   python infer_os.py
   ```

## Model Architecture

[Details about the model architecture, training strategy, etc.]

## Performance

[Performance metrics, validation results, etc.] 