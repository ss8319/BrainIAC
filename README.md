# BrainIAC: A foundation model for generalized Brain MRI analysis

<p align="center">
  <img src="pictures/brainiac.jpeg" width="1100" alt="BrainIAC Logo"/>
</p>

## Overview

This repository provides the model and implementation details for BrainIAC, a vision foundation model for generalized Brain MRI analysis. BrainIAC has been trained using SimCLR on 32,000 brain MR scans. The foundation model has been downstream validated across 6 different tasks with wide ranging endpoint difficulties - 
- MR Sequence Classification
- Brain age prediction
- IDH Mutation classification (IDH Mutant Vs Wildtype)
- Overall survival for GBM subjects
- Mild Cognitive impairment (MCI) classification (MCI Vs Healthy Control)
- Time to stroke onset prediction

The core of BrainIAC is a 3D ResNet50 model, which takes in complete 3D Brain MR volume to generate robust adaptable representations that are downstream transferable. For more insights checkout the preprint [Here](https://www.medrxiv.org/content/10.1101/2024.12.02.24317992v1)

## Key Features

- **Setup and installation**: Setup the environment and install BrainIAC and the downstream models
- **Preprocessing**: Preprocess the brain MRI scans for training and inference
- **BrainIAC Feature Extraction**: Extract 2048-dimensional feature vectors from brain MRI scans
- **Saliency Map Visualization**: Generate saliency maps to visualize model attention
- **Downstream Task Train/Infer**: Infer or train the downstream models
- **Use as a library**: Run and build on top of BrainIAC and the downstream models via python library 
- **Plug and play docker**: Plug and play with BrainIAC and downstream models via docker

## Model checkpoints
Download the BrainIAC and downstream model checkpoints here : [Model Checkpoints](https://drive.google.com/drive/folders/13xMyLS8vy07dNgWHyXDR4A-O_m7hgZZQ?usp=sharing)


## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/DivyanshuTak/BrainIAC.git
cd brainiac

# Create conda environment
conda create -n brainiac python=3.9
conda activate brainiac

# Install dependencies
pip install -r requirements.txt

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

```

## Preprocessing
First convert the dicom to nifti . Skip this step if you have the MR in nifti format
```bash
python dicomtonifti.py -i /path/to/input/directory -o /path/to/output/directory
```
Preprocess the nifti files for training and inference. The preprocessing includes bias field correction, regitration and skull stripping (HD-BET).

```bash
## pass the path to the atlas template (atlases directory) and the nifti directory and the output path
python mri_preprocess_3d.py \
  --temp_img /atlases/template.nii.gz \
  --T2W_dir /path/to/T2W/directory \
  --output_path /path/to/output/directory
```

The preprocessed images will stored in "/output_path/nnunet/imagesTs/" directory. 


## BrainIAC Feature Extraction

To extract the radiomics features:
1. Ensure your image filenames follow the format: `id_scandate.nii.gz`
2. Create a metadata CSV file following the structure shown in `BrainIAC/sample.csv`
3. Run the feature extraction:

```bash
python get_brainiac_features.py \
    --checkpoint /path/to/model/checkpoint \
    --input_csv input_scans.csv \
    --output_csv features.csv \
    --root_dir /path/to/data
```
The resultant features will be stored in the features.csv file

## Saliency map visualization

BrainIAC inference can be used to generate saliency maps that highlight regions of the brain MR that most influenced the model's decisions. To generate saliency maps:

1. Prepare your input data following the same format as feature extraction
2. Run the saliency map generation:

```bash
python get_brainiac_saliencymap.py \
    --checkpoint /path/to/model/checkpoint \
    --input_csv input_scans.csv \
    --output_dir /path/to/saliency_maps \
    --root_dir /path/to/data
```

The script will generate two files in the output directory for each input scan:
- `{scan_id}_image.nii.gz`: The preprocessed input scan
- `{scan_id}_saliencymap.nii.gz`: The corresponding 3D saliency map

To visualize the saliency maps overlaid on the original scans, use the provided Jupyter notebook:
1. Open `utils/visualization.ipynb`
2. Update the input paths in the configuration cell
3. Run the notebook to generate:
   - saliency heatmap 
   - overlay countour

## Downstream tasks

BrainIAC was validated on multiple downstream tasks through a common pipeline. 

- **Common Components**:
  - `config.yml`: Central configuration file for all tasks
  - `model.py`: Core model architectures and components
  - `dataset2.py`: Data loading and augmentation pipelines
  - `utils.py`: Shared utilities and helper functions

### Pipeline 

1. **Data Preprocessing**:
   Follow the preprocessing steps outlined in the [Preprocessing](#preprocessing) section above:
   Ensuring the data is:
   - Bias field corrected
   - Registered to standard space
   - Skull stripped


2. **Data Preparation**:
   ```bash
   # Create CSV files with columns: pat_id, label

   # Example format:
   
   # pat_id,scandate,label
   # subject001,20240101,65.5    # For brain age
   # subject002,20240102,1       # For binary classification
   ```

1. **Configuration**:
   ```yaml
   # Update config.yml with task-specific settings
   data:
     train_csv: "path/to/train.csv"
     val_csv: "path/to/val.csv"
     test_csv: "path/to/test.csv"
     root_dir: "path/to/preprocessed/scans"
   
   infer:
    checkpoints: "path/to/downstream/models/checkpoints" # for running inference of downstream models

   train:
     finetune: "yes"  # Use BrainIAC weights
     weights: "path/to/brainiac/checkpoint"
   ```

### Available Tasks

#### 1. Brain Age Prediction

<p align="left">
  <img src="pictures/brainage.jpeg" width="100" alt="Brain Age Prediction Example"/>
  <br>
</p>

Predict brain age from T1-weighted MRI:
```bash
## set config["data"]["collate"] = 1 for single scan framework
cd DownstreamTasks
# Training
## change the config["train"]["weights"] to the path to the brainiac weights
python -m Brainage.train_brainage

# Inference
python -m Brainage.infer_brainage
```

#### 2. IDH Mutation Classification

<p align="left">
  <img src="pictures/idh.jpeg" width="100"/>
  <br>
</p>

Binary classification of IDH mutation status using T1CE and FLAIR:
```bash
## set config["data"]["collate"] = 2 for single scan framework
cd DownstreamTasks
# Training
## change the config["train"]["weights"] to the path to the brainiac weights
python -m IDHprediction.train_idh

# Inference
python -m IDHprediction.infer_idh
```

#### 3. MCI Classification

<p align="left">
  <img src="pictures/mci.jpeg" width="100"/>
  <br>
</p>

Binary classification of Mild Cognitive Impairment vs healthy control from T1w images:
```bash
## set config["data"]["collate"] = 1 for single scan framework
cd DownstreamTasks
# Training
## change the config["train"]["weights"] to the path to the brainiac weights
python -m MCIclassification.train_mci

# Inference
python -m MCIclassification.infer_mci
```

#### 4. Overall Survival Prediction

<p align="left">
  <img src="pictures/os.jpeg" width="90"/>
  <br>
</p>

Predicts overall survival for GBM patients using T1CE, T1w, T2w and FLAIR:
```bash
## set config["data"]["collate"] = 4 for single scan framework
cd DownstreamTasks
# Training
python -m OverallSurvival.train_os

# Inference
python -m OverallSurvival.infer_os
```

#### 5. Sequence Classification

<p align="left">
  <img src="pictures/sequence.jpeg" width="100"/>
  <br>
</p>

Multi-class classification of MR sequences:
```bash
## set config["data"]["collate"] = 1 for single scan framework
cd DownstreamTasks
# Training
## change the config["train"]["weights"] to the path to the brainiac weights
python -m SequenceClassification.train_sequence

# Inference
python -m SequenceClassification.infer_sequence
```

#### 6. Time to Stroke Prediction

<p align="left">
  <img src="pictures/stroke.jpeg" width="100"/>
  <br>
</p>

Regression model for stroke onset prediction from T1w images:
```bash
## set config["data"]["collate"] = 1 for single scan framework
cd DownstreamTasks
# Training
## change the config["train"]["weights"] to the path to the brainiac weights
python -m timetostroke.train_stroke

# Inference
python -m timetostroke.infer_stroke
```

## Use as a library

TODO..

## Plug and play docker

TODO..

## 📚 Citation

If you use BrainIAC in your research, please cite:

```bibtex
@article{tak2024brainiac,
    title={BrainIAC: A Foundation Model for Generalized Brain MRI Analysis},
    author={Tak, Divyanshu and others},
    journal={medRxiv},
    year={2024},
    doi={10.1101/2024.12.02.24317992}
}
```

## 📄 License

BrainIAC is released under the MIT License. See [LICENSE](LICENSE) for details.


---


