# BrainIAC: A foundation model for generalized Brain MRI analysis

<p align="center">
  <img src="pictures/brainiac.jpeg" width="1100" alt="BrainIAC Logo"/>
</p>

## Overview

BrainIAC is a vision foundation model for generalized Brain MRI analysis, trained using SimCLR on 32,000 brain MR scans. The model has been validated across 6 different downstream tasks:

- MR Sequence Classification
- Brain age prediction
- IDH Mutation classification
- Overall survival for GBM subjects
- Mild Cognitive impairment (MCI) classification
- Time to stroke onset prediction

For detailed information, check out our [preprint](https://www.medrxiv.org/content/10.1101/2024.12.02.24317992v1).

## Quick Start

```bash
# Clone and setup
git clone https://github.com/DivyanshuTak/BrainIAC.git
cd brainiac

# Create conda environment
conda create -n brainiac python=3.9
conda activate brainiac
pip install -r requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Model Checkpoints

Download BrainIAC and downstream model checkpoints: [Model Checkpoints](https://drive.google.com/drive/folders/13xMyLS8vy07dNgWHyXDR4A-O_m7hgZZQ?usp=sharing)

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

## Saliency Map Visualization

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

## Downstream Tasks

- [Brain Age Prediction](DownstreamTasks/Brainage/README.md)
- [IDH Mutation Classification](DownstreamTasks/IDHprediction/README.md)
- [MCI Classification](DownstreamTasks/MCIclassification/README.md)
- [Overall Survival Prediction](DownstreamTasks/OverallSurvival/README.md)
- [Sequence Classification](DownstreamTasks/SequenceClassification/README.md)
- [Time to Stroke Prediction](DownstreamTasks/timetostroke/README.md)

## Citation

```bibtex
@article{tak2024brainiac,
    title={BrainIAC: A Foundation Model for Generalized Brain MRI Analysis},
    author={Tak, Divyanshu and others},
    journal={medRxiv},
    year={2024},
    doi={10.1101/2024.12.02.24317992}
}
```

## License

BrainIAC is released under the MIT License. See [LICENSE](LICENSE) for details.


---


