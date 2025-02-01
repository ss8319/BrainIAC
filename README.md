# BrainIAC: Generalize Vision Foundation Model for Brain MRI 

<p align="center">
  <img src="github_brainiac_logo.png" width="400" alt="BrainIAC Logo"/>
</p>

## Overview

BrainIAC is a state-of-the-art foundation model for brain MRI analysis, built on a modified ResNet-50 architecture specifically optimized for 3D neuroimaging data. It provides powerful feature extraction capabilities and interpretable saliency maps to support clinical decision making and neuroscience research.

## 🌟 Key Features

- **Robust Feature Extraction**: Extract 2048-dimensional feature vectors from brain MRI scans
- **Interpretable Results**: Generate saliency maps to visualize model attention
- **Clinical Integration**: Easy-to-use interface for clinical workflows
- **Research Ready**: Flexible architecture for transfer learning and adaptation
- **Production Grade**: Optimized for both research and clinical deployment

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/brainiac.git
cd brainiac

# Create conda environment
conda create -n brainiac python=3.9
conda activate brainiac

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from load_brainiac import load_brainiac

# Load pre-trained model
model = load_brainiac("path/to/checkpoint", device="cuda")

# Extract features
features = model(input_scan)
```

## 📊 Feature Extraction

Extract meaningful features from brain MRI scans:

```bash
python get_brainiac_features.py \
    --checkpoint /path/to/model/checkpoint \
    --input_csv input_scans.csv \
    --output_csv features.csv \
    --root_dir /path/to/data
```

## 🎯 Saliency Maps

Generate interpretable saliency maps:

```bash
python get_brainiac_saliencymap.py \
    --checkpoint /path/to/model/checkpoint \
    --input_csv input_scans.csv \
    --output_dir /path/to/output \
    --root_dir /path/to/data
```

## 📋 Model Architecture

BrainIAC is built on a modified ResNet-50 architecture:
- Input: 3D MRI scans (1 channel)
- Backbone: Modified ResNet-50 with 3D convolutions
- Output: 2048-dimensional feature vector
- Additional: Guided backpropagation for saliency maps

## 🔬 Research Applications

BrainIAC has been successfully applied to:
- Disease classification
- Longitudinal analysis
- Biomarker discovery
- Treatment response prediction

## 📚 Citation

If you use BrainIAC in your research, please cite:

```bibtex
@article{brainiac2024,
  title={BrainIAC: A Foundation Model for Generalized Brain MRI Analysis},
  author={Your Team},
  year={2025}
}
```


## 📄 License

BrainIAC is released under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

We thank our collaborators and the open-source community for their valuable contributions.


---


