# Overall Survival Prediction

<p align="left">
  <img src="os.jpeg" width="200" alt="Overall Survival Prediction Example"/>
</p>

## Overview

We present the overall survival prediction training and inference code for BrainIAC as a downstream task. The pipeline is trained and infered on multiple MRI sequences (T1CE, T1w, T2w, and FLAIR), with MAE as evaluation metric.

## Data Requirements

- **Input**: T1CE, T1w, T2w, and FLAIR MR sequences from a single scan
- **Format**: NIFTI (.nii.gz)
- **Preprocessing**: Bias field corrected, registered to standard space, skull stripped
- **CSV Structure**:
  ```
  pat_id,scandate,label
  subject001,imaging_sequence,1    # 1 year overall survival 
  ```
refer to [ quickstart.ipynb](../quickstart.ipynb) to find how to preprocess data and generate csv file.

## Setup

1. **Configuration**:
change the [config.yml](../config.yml) file accordingly.
   ```yaml
   # config.yml
   data:
     train_csv: "path/to/train.csv"
     val_csv: "path/to/val.csv"
     test_csv: "path/to/test.csv"
     root_dir: "../data/sample/processed"
     collate: 4  # 4 sequence pipeline
    
   checkpoints: "./checkpoints/os_model.00"     # for inference/testing 
   
   train:
    finetune: 'yes'      # yes to finetune the entire model 
    freeze: 'no'         # yes to freeze the resnet backbone 
    weights: ./checkpoints/brainiac.ckpt  # path to brainiac weights
   ```

2. **Training**:
   ```bash
   python -m OverallSurvival.train_os
   ```

3. **Inference**:
   ```bash
   python -m OverallSurvival.infer_os
   ```

