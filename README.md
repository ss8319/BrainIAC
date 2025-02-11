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

## Installation

```bash
# Clone and setup
git clone https://github.com/DivyanshuTak/BrainIAC.git
cd brainiac

# Create conda environment
conda create -n brainiac python=3.9
conda activate brainiac
pip install -r requirements.txt
```

## Model Checkpoints

Download BrainIAC and downstream model checkpoints: [Model Checkpoints](https://drive.google.com/drive/folders/13xMyLS8vy07dNgWHyXDR4A-O_m7hgZZQ?usp=sharing) and place them in the [checkpoints](./src/BrainIAC/checkpoints) directory.

## Quick Start

See [quickstart.ipynb](./src/BrainIAC/quickstart.ipynb) to get started of how to preprocess data, load BrainIAC to extract features, generate and visualize saliency maps. We provide a sample dataset from publicly available [UPENN-GBM](https://www.cancerimagingarchive.net/collection/upenn-gbm/), [License](https://creativecommons.org/licenses/by/4.0/) (with no modifications to the provided preprocessed images)in the [sample_data](src/BrainIAC/data/sample/processed/) directory and the corresponding csv file in [input_scans.csv](src/BrainIAC/data/csvs/input_scans.csv) to run BrainIAC and downstream tasks.


## Train Downstream Models

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


