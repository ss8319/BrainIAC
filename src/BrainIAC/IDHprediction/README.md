# IDH Mutation Classification

<p align="left">
  <img src="idh.jpeg" width="200" alt="IDH Mutation Classification Example"/>
</p>

## Overview

We present the IDH mutation classification training and inference code for BrainIAC as a downstream task. The pipeline is trained and infered on T1CE and FLAIR scans, with AUC and F1 as evaluation metric.

## Data Requirements

- **Input**: T1CE and FLAIR MR sequences from a single scan
- **Format**: NIFTI (.nii.gz)
- **Preprocessing**: Bias field corrected, registered to standard space, skull stripped
- **CSV Structure**:
  ```
  pat_id,scandate,label
  subject001,scan_sequence,1    # 1 for IDH mutant, 0 for wildtype
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
     collate: 2  # two sequence pipeline  
    
   checkpoints: "./checkpoints/idh_model.00"     # for inference/testing 
   
   train:
    finetune: 'yes'      # yes to finetune the entire model 
    freeze: 'no'         # yes to freeze the resnet backbone 
    weights: ./checkpoints/brainiac.ckpt  # path to brainiac weights
   ```

2. **Training**:
   ```bash
   python -m IDHprediction.train_idh
   ```

3. **Inference**:
   ```bash
   python -m IDHprediction.infer_idh
   ```

