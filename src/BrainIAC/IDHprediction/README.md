# IDH Mutation Classification

<p align="left">
  <img src="../../pictures/idh.jpeg" width="200" alt="IDH Mutation Classification Example"/>
</p>

## Overview

This module performs binary classification of IDH mutation status using T1CE and FLAIR MRI scans. The model has been trained on [dataset details] and achieves [performance metrics].

## Data Requirements

- **Input**: T1CE and FLAIR MRI scans
- **Format**: NIFTI (.nii.gz)
- **Preprocessing**: Bias field corrected, registered to standard space, skull stripped
- **CSV Structure**:
  ```
  pat_id,scandate,label
  subject001,20240101,1    # 1 for IDH mutant, 0 for wildtype
  ```

## Usage

### Using Docker

```bash
docker pull brainiac/idh:latest
docker run -v /path/to/data:/data brainiac/idh:latest [args]
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
     collate: 2  # dual scan framework
   ```

2. **Training**:
   ```bash
   python train_idh.py
   ```

3. **Inference**:
   ```bash
   python infer_idh.py
   ```

## Model Architecture

[Details about the model architecture, training strategy, etc.]

## Performance

[Performance metrics, validation results, etc.] 