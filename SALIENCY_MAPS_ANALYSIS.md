# Analysis: Saliency Map Generation Scripts

## Overview
There are **6 distinct saliency map generation scripts** in the BrainIAC codebase. While they share the core `extract_attention_map()` function, they differ significantly in model architecture, input modalities, and task-specific requirements.

---

## 1. `get_brainiac_saliencymap.py` - **Generic Batch Processor**

### Purpose
Generic script for batch processing multiple images from a CSV file using the base BrainIAC model.

### Key Characteristics
- **Model Loading**: Uses `load_brainiac()` helper function
- **Model Access Path**: `model.backbone` (direct access)
- **Input**: Single 3D MRI volume per sample
- **Processing**: Batch processing via DataLoader
- **Dataset**: Uses `BrainAgeDataset` with CSV input
- **Interface**: Command-line arguments (argparse)
- **Output**: Batch processing with success/error counting

### Why Distinct
- **Different model structure**: Loads base BrainIAC model, not task-specific Lightning modules
- **Batch processing**: Designed for processing entire datasets, not single images
- **Simpler architecture**: No Lightning wrapper, direct model access

---

## 2. `generate_brainage_vit_saliency.py` - **Brain Age Regression**

### Purpose
Generate saliency maps for brain age prediction (regression task).

### Key Characteristics
- **Model**: `BrainAgeLightningModule` (from `train_lightning_brainage.py`)
- **Model Access Path**: `model.backbone.backbone` (Lightning → Model → Backbone)
- **Input**: Single 3D MRI volume
- **Task**: Regression (predicts continuous age value)
- **Model Loading**: Uses Lightning's `load_from_checkpoint()` with config
- **Processing**: Single image processing
- **Interface**: Hardcoded paths at top of file

### Why Distinct
- **Task-specific model**: Brain age prediction requires specific architecture
- **Lightning module**: Uses PyTorch Lightning checkpoint format
- **Regression task**: Different from classification tasks

---

## 3. `generate_idh_vit_saliency.py` - **IDH Mutation Prediction (Dual Input)**

### Purpose
Generate saliency maps for IDH mutation status prediction using **two input modalities**.

### Key Characteristics
- **Model**: `DualInputBinaryClassificationLightningModule` (from `train_lightning_idh.py`)
- **Model Access Path**: `model.backbone.backbone`
- **Input**: **Dual input** - T2F and T1CE sequences
- **Architecture**: `SingleScanModelBP` (Bilateral Processing)
- **Task**: Binary classification (IDH mutated vs. wildtype)
- **Model Loading**: Manual checkpoint loading (PyTorch 2.6 compatibility)
- **PyTorch 2.6 Compatibility**: Includes MetaTensor and numpy safe globals

### Why Distinct
- **Multi-modal input**: Requires two images (T2F + T1CE) - **CRITICAL DIFFERENCE**
- **Different model architecture**: `SingleScanModelBP` processes two inputs separately through shared backbone
- **Cannot use single-image scripts**: The model expects two inputs, so single-image scripts won't work

### Architecture Note
```python
# Model processes two images:
features1 = self.backbone(image1)  # T2F
features2 = self.backbone(image2)  # T1CE
# Then merges features for classification
```

---

## 4. `generate_mci_stroke_vit_saliency.py` - **MCI/Stroke Classification**

### Purpose
Generate saliency maps for MCI (Mild Cognitive Impairment) and Stroke classification.

### Key Characteristics
- **Model**: `MCIClassificationLightningModule` (from `train_lightning_mci.py`)
- **Model Access Path**: `model.backbone.backbone`
- **Input**: Single 3D MRI volume
- **Task**: Classification (MCI/Stroke categories)
- **Model Loading**: Manual checkpoint loading (PyTorch 2.6 compatibility)
- **PyTorch 2.6 Compatibility**: Includes MetaTensor and numpy safe globals

### Why Distinct
- **Task-specific model**: MCI/Stroke classification requires specific training
- **Different checkpoint format**: May have different state_dict structure
- **PyTorch compatibility**: Includes compatibility code for newer PyTorch versions

---

## 5. `generate_multiclass_vit_saliency.py` - **Multi-Class Sequence Classification**

### Purpose
Generate saliency maps for multi-class sequence classification (multiple MRI sequence types).

### Key Characteristics
- **Model**: `MultiClassSequenceLightningModule` (from `train_lightning_multiclass.py`)
- **Model Access Path**: `model.backbone.backbone`
- **Input**: Single 3D MRI volume
- **Task**: Multi-class classification (multiple sequence types)
- **Model Loading**: Manual checkpoint loading (PyTorch 2.6 compatibility)
- **PyTorch 2.6 Compatibility**: Includes MetaTensor and numpy safe globals

### Why Distinct
- **Multi-class task**: Different from binary classification
- **Sequence classification**: Classifies different MRI sequence types
- **Different number of output classes**: Affects model architecture

---

## 6. `generate_os_vit_saliency.py` - **Overall Survival Prediction (Quad Input)**

### Purpose
Generate saliency maps for overall survival prediction using **four input modalities**.

### Key Characteristics
- **Model**: `QuadInputBinaryClassificationLightningModule` (from `train_lightning_os.py`)
- **Model Access Path**: `model.backbone.backbone`
- **Input**: **Quad input** - 4 MRI sequences (T1CE, T1N, T2W, T2F)
- **Architecture**: `SingleScanModelQuad` (Quad Processing)
- **Task**: Binary classification (survival prediction)
- **Model Loading**: Manual checkpoint loading (PyTorch 2.6 compatibility)
- **Preprocessing**: Includes `ScaleIntensityd` (unlike others)
- **PyTorch 2.6 Compatibility**: Includes MetaTensor and numpy safe globals

### Why Distinct
- **Four input modalities**: Requires 4 images - **CRITICAL DIFFERENCE**
- **Different model architecture**: `SingleScanModelQuad` processes four inputs
- **Cannot use single/dual-image scripts**: Model expects exactly 4 inputs
- **Different preprocessing**: Includes intensity scaling

### Architecture Note
```python
# Model processes four images:
features1 = self.backbone(image1)  # T1CE
features2 = self.backbone(image2)  # T1N
features3 = self.backbone(image3)  # T2W
features4 = self.backbone(image4)  # T2F
# Then merges all features for classification
```

---

## Key Differences Summary

| Script | Input Modalities | Model Architecture | Model Access | Task Type | Batch Processing |
|--------|-----------------|-------------------|--------------|-----------|------------------|
| `get_brainiac_saliencymap.py` | 1 (Single) | Base BrainIAC | `model.backbone` | Generic | ✅ Yes (CSV) |
| `generate_brainage_vit_saliency.py` | 1 (Single) | `SingleScanModel` | `model.backbone.backbone` | Regression | ❌ No (Single) |
| `generate_idh_vit_saliency.py` | **2 (Dual)** | `SingleScanModelBP` | `model.backbone.backbone` | Binary Classification | ❌ No (Single) |
| `generate_mci_stroke_vit_saliency.py` | 1 (Single) | `SingleScanModel` | `model.backbone.backbone` | Classification | ❌ No (Single) |
| `generate_multiclass_vit_saliency.py` | 1 (Single) | `SingleScanModel` | `model.backbone.backbone` | Multi-Class | ❌ No (Single) |
| `generate_os_vit_saliency.py` | **4 (Quad)** | `SingleScanModelQuad` | `model.backbone.backbone` | Binary Classification | ❌ No (Single) |

---

## Why They Cannot Be Unified

### 1. **Different Model Architectures**
- **Single input**: `SingleScanModel` (BrainAge, MCI, MultiClass)
- **Dual input**: `SingleScanModelBP` (IDH) - processes 2 images
- **Quad input**: `SingleScanModelQuad` (OS) - processes 4 images
- **Base model**: Direct `ViTBackboneNet` (generic script)

### 2. **Different Input Requirements**
- Single-image scripts cannot handle dual/quad inputs
- Dual-input script cannot handle single/quad inputs
- Quad-input script cannot handle single/dual inputs

### 3. **Different Model Loading Methods**
- Generic script: Uses `load_brainiac()` helper
- BrainAge: Uses Lightning's `load_from_checkpoint()`
- Others: Manual checkpoint loading with PyTorch 2.6 compatibility

### 4. **Different Lightning Modules**
Each task has its own Lightning module:
- `BrainAgeLightningModule`
- `DualInputBinaryClassificationLightningModule`
- `MCIClassificationLightningModule`
- `MultiClassSequenceLightningModule`
- `QuadInputBinaryClassificationLightningModule`

### 5. **Different Preprocessing**
- OS script includes `ScaleIntensityd` transform
- Others may have different normalization strategies

### 6. **Different Checkpoint Formats**
- Lightning checkpoints have `state_dict` nested structure
- Base BrainIAC may have different checkpoint format
- PyTorch 2.6 compatibility requires different loading strategies

---

## Code Duplication

### Shared Code
All scripts share the **exact same** `extract_attention_map()` function:
- Attention weight extraction
- CLS token attention selection
- 3D reshaping and upsampling
- Normalization

### Potential Refactoring
The `extract_attention_map()` function could be extracted to a shared module to reduce duplication. However, the scripts must remain separate due to:
1. Different model architectures
2. Different input modalities
3. Different loading mechanisms
4. Task-specific requirements

---

## Recommendations

### 1. **Extract Common Function**
Create a shared module (e.g., `attention_utils.py`) with:
```python
def extract_attention_map(vit_model, image, layer_idx=-1, ...):
    # Shared implementation
```

### 2. **Unify Model Loading**
Create a factory function that handles different checkpoint formats:
```python
def load_model_for_saliency(checkpoint_path, model_type, config_path=None):
    # Handle different model types and loading strategies
```

### 3. **Keep Scripts Separate**
Maintain separate scripts for:
- Different input modalities (single/dual/quad)
- Different tasks (regression/classification)
- Different use cases (batch vs. single image)

### 4. **Documentation**
Add clear documentation explaining:
- Which script to use for which task
- Input format requirements
- Model architecture differences

---

## Conclusion

The scripts are **necessarily distinct** due to fundamental architectural differences:
- **Input modalities**: Single (1), Dual (2), or Quad (4) images
- **Model architectures**: Different wrapper models for different input types
- **Task requirements**: Regression vs. classification, binary vs. multi-class
- **Loading mechanisms**: Different checkpoint formats and loading strategies

While there is code duplication in `extract_attention_map()`, the scripts serve different purposes and cannot be unified without significant architectural changes that would break existing functionality.

