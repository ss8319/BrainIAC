# Brain Age Prediction

<p align="left">
  <img src="brainage.jpeg" width="200" alt="Brain Age Prediction Example"/>
</p>

## Overview

We present the brainage prediction training and inference code for BrainIAC as a downstream task. The pipeline is trained and infered on T1 scans, with MAE as evaluation metric.

## Data Requirements

- **Input**: T1-weighted MRI scans
- **Format**: NIFTI (.nii.gz)
- **Preprocessing**: Bias field corrected, registered to standard space, skull stripped
- **CSV Structure**:
  ```
  pat_id,scandate,label
  subject001,20240101,65    # brain age in years
  ```
refer to [ quickstart.ipynb](../quickstart.ipynb)  to find how to preprocess data and generate csv file.


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
     collate: 1  # single scan framework
    
   checkpoints: "./checkpoints/brainage_model.00"     # for inference/testing 
   
   train:
    finetune: 'yes'      # yes to finetune the entire model 
    freeze: 'no'         # yes to freeze the resnet backbone 
    weights: ./checkpoints/brainiac.ckpt  # path to brainiac weights

   ```

2. **Training**:
   ```bash
   python -m Brainage.train_brainage
   ```

3. **Inference**:
   ```bash
   python -m Brainage.infer_brainage
   ```

