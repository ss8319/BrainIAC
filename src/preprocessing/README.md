# Preprocessing Pipelines

## Key preprocessing steps
### 1. Convert DICOM to Nifti files
- DICOM files include 1 .dcm per slice in image
- 1 Nifti file has full 3D image
- image will be in a new directory

### 2. Preprocess Nifti files
- Registration to a common anatomical template
- Brain extraction (segmentation) using HD-BET segmentation model

The preprocessing pipeline consists of two main stages:

### Stage 1: DICOM to NIfTI Conversion
- **Input**: DICOM files (1 `.dcm` file per slice)
- **Output**: NIfTI files (1 file containing the full 3D image)
- **Purpose**: Convert raw DICOM data into a standardized neuroimaging format

### Stage 2: NIfTI Preprocessing
- **Registration**: Align images to a common anatomical template
- **Brain Extraction**: Segment brain tissue using HD-BET segmentation model
- **Purpose**: Standardize images for consistent analysis

## Usage

### 1. DICOM to NIfTI Conversion

```bash
python dicomtonifti_adni.py --input /path/to/raw/dicom --output /path/to/nifti/output
```

**Example:**
```bash
python dicomtonifti_adni.py --input /home/ssim0068/data/raw/AD_CN_MRI_final --output /home/ssim0068/data/nifti/AD_CN_MRI_final
```

**Notes:**
- Modified from `dicomtonifti_2.py` with improvements for ADNI directory organization
- Preserves original directory structure in output

### 2. NIfTI Preprocessing

**Option A: Direct execution**
```bash
python mri_preprocess_adni.py --temp_img /path/to/template --input_dir /path/to/nifti --output_dir /path/to/preprocessed
```

**Option B: SLURM job (recommended for compute-intensive tasks)**
```bash
bash preprocess.sh
```

**Notes:**
- Lightly modified from `mri_preprocess_3d_simple.py`
- Use SLURM job submission for large datasets due to computational intensity

## Command-line Arguments

### `mri_preprocess_adni.py` Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--temp_img` | Required | Path to the common anatomical atlas template image for registration |
| `--input_dir` | Required | Path to the source NIfTI images (output from Stage 1) |
| `--output_dir` | Required | Path where preprocessed images are saved |
| `--resume` | Optional | Allows processing to resume from where it previously stopped (for recovery) |
| `--debug` | Optional | Enables verbose logging for troubleshooting |
| `--limit` | Optional | Limits the number of images processed (primarily for testing and debugging) |

## File Organization

```
preprocessing/
├── README.md                    # This file
├── dicomtonifti_adni.py        # DICOM to NIfTI conversion script
├── mri_preprocess_adni.py      # NIfTI preprocessing script
├── mri_preprocess_3d_simple.py # Original preprocessing script
├── dicomtonifti_2.py           # Original conversion script
└── preprocess.sh               # SLURM job submission script
```


