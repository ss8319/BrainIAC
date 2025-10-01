# BrainIAC Foundation Model
Light adapted from BrainIAC model

## Code Organisation
**Purpose** 
Prepare preprocessed dataset into the format and organisation that is expected to run linear probe experiments
```bash
python convert_adni_to_brainiac_v2.py
```
| Argument | Default | Description |
|----------|---------|-------------|
| `--copy-files` | False | Copy NIfTI files to output |
| `--metadata-path` | `/home/ssim0068/data/AD_CN_train_v2/metadata.csv` | Input metadata CSV |
| `--nifti-base-dir` | `/home/ssim0068/data/v2_preprocessed_icbm152` | NIfTI files location |
| `--output-base-dir` | `/home/ssim0068/data/ADNI_v2_icbm152` | Output directory |

### Input Format
**Metadata CSV:**
```csv
Subject,Sex,Age,Group,ScanPath
023_S_0139,M,75,CN,023_S_0139/MPRAGE/2006-02-06_12_11_27.0/I10861
```

**NIfTI Structure:**
```
nifti-base-dir/023_S_0139/MPRAGE/2006-02-06_12_11_27.0/2006-02-06_12_11_27.0.nii.gz
```

### Output

**CSV files:** `mci_train.csv`, `mci_val.csv`, `mci_test.csv`
**Format:** `pat_id,label,Sex,Age` (label: 0=CN, 1=AD)

**Features:** Balanced splits by sex + disease status, stratified sampling


```bash
python convert_adni_to_brainiac_v1.py
```
- Uses pre-split CSV files that define train and test split
- Search through preprocessed image directory to find those scans

### Bash scripts
```bash
sbatch train.sh
```

```bash
sbatch test_inference.sh
```