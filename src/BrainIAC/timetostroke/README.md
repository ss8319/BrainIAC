# Time to Stroke Prediction

<p align="left">
  <img src="../../pictures/stroke.jpeg" width="200" alt="Time to Stroke Prediction Example"/>
</p>

## Overview

This module predicts the time since stroke onset using T1-weighted MRI scans. The model has been trained on [dataset details] and achieves [performance metrics].

## Data Requirements

- **Input**: T1-weighted MRI scans
- **Format**: NIFTI (.nii.gz)
- **Preprocessing**: Bias field corrected, registered to standard space, skull stripped
- **CSV Structure**:
  ```
  pat_id,scandate,label
  subject001,20240101,4.5    # time since stroke onset in hours
  ```

## Usage

### Using Docker

```bash
docker pull brainiac/stroke:latest
docker run -v /path/to/data:/data brainiac/stroke:latest [args]
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
   python train_stroke.py
   ```

3. **Inference**:
   ```bash
   python infer_stroke.py
   ```

## Model Architecture

[Details about the model architecture, training strategy, etc.]

## Performance

[Performance metrics, validation results, etc.] 