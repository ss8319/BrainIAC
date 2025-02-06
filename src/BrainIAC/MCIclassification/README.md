# MCI Classification

<p align="left">
  <img src="../../pictures/mci.jpeg" width="200" alt="MCI Classification Example"/>
</p>

## Overview

This module performs binary classification between Mild Cognitive Impairment (MCI) and healthy controls using T1-weighted MRI scans. The model has been trained on [dataset details] and achieves [performance metrics].

## Data Requirements

- **Input**: T1-weighted MRI scans
- **Format**: NIFTI (.nii.gz)
- **Preprocessing**: Bias field corrected, registered to standard space, skull stripped
- **CSV Structure**:
  ```
  pat_id,scandate,label
  subject001,20240101,1    # 1 for MCI, 0 for healthy control
  ```

## Usage

### Using Docker

```bash
docker pull brainiac/mci:latest
docker run -v /path/to/data:/data brainiac/mci:latest [args]
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
     collate: 1  # single scan framework
   ```

2. **Training**:
   ```bash
   python train_mci.py
   ```

3. **Inference**:
   ```bash
   python infer_mci.py
   ```

## Model Architecture

[Details about the model architecture, training strategy, etc.]

## Performance

[Performance metrics, validation results, etc.] 