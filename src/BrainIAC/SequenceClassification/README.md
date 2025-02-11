# MR Sequence Classification

<p align="left">
  <img src="sequence.jpeg" width="200" alt="Sequence Classification Example"/>
</p>

## Overview

We present the MR sequence classification training and inference code for BrainIAC as a downstream task. The pipeline is trained and infered on T1/T2/FLAIR/T1CE brain MR, with balanced accuracy and AUC as evaluation metric.

## Data Requirements

- **Input**: single Brain MR sequence
- **Format**: NIFTI (.nii.gz)
- **Preprocessing**: Bias field corrected, registered to standard space, skull stripped
- **CSV Structure**:
  ```
  pat_id,scandate,label
  subject001,20240101,0    # 0:T1w, 1:T2w, 2:FLAIR, 3:T1CE
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
     collate: 1  # single scan framework
    
   checkpoints: "./checkpoints/sequence_model.00"     # for inference/testing 
   
   train:
    finetune: 'yes'      # yes to finetune the entire model 
    freeze: 'no'         # yes to freeze the resnet backbone 
    weights: ./checkpoints/brainiac.ckpt  # path to brainiac weights
   ```

2. **Training**:
   ```bash
   python -m SequenceClassification.train_sequence
   ```

3. **Inference**:
   ```bash
   python -m SequenceClassification.infer_sequence
   ```

