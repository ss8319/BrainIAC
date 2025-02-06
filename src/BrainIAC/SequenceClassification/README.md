# MR Sequence Classification

<p align="left">
  <img src="../../pictures/sequence.jpeg" width="200" alt="Sequence Classification Example"/>
</p>

## Overview

This module performs multi-class classification of MR sequences (T1w, T2w, FLAIR, T1CE). The model has been trained on [dataset details] and achieves [performance metrics].

## Data Requirements

- **Input**: Any brain MRI sequence
- **Format**: NIFTI (.nii.gz)
- **Preprocessing**: Bias field corrected, registered to standard space, skull stripped
- **CSV Structure**:
  ```
  pat_id,scandate,label
  subject001,20240101,0    # 0:T1w, 1:T2w, 2:FLAIR, 3:T1CE
  ```

## Usage

### Using Docker

```bash
docker pull brainiac/sequence:latest
docker run -v /path/to/data:/data brainiac/sequence:latest [args]
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
   python train_sequence.py
   ```

3. **Inference**:
   ```bash
   python infer_sequence.py
   ```

## Model Architecture

[Details about the model architecture, training strategy, etc.]

## Performance

[Performance metrics, validation results, etc.] 