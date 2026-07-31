# Saliency Map Generation Procedure - Detailed Explanation

## Overview
This document explains how saliency maps are generated from Vision Transformer (ViT) attention weights in `get_brainiac_saliencymap.py`. The method uses **CLS token attention** to visualize which image regions the model focuses on.

---

## Core Procedure: Step-by-Step

### **Step 1: Hook into Attention Modules** (Lines 29-55)

The first step is to intercept attention weights during the forward pass. Since ViT attention modules don't automatically return attention weights, we wrap them:

```python
# Lines 30-50: AttentionWithWeights wrapper class
class AttentionWithWeights(torch.nn.Module):
    def __init__(self, original_attn_module):
        super().__init__()
        self.original_attn_module = original_attn_module
        self.attn_weights = None  # Will store attention weights here

    def forward(self, x):
        # Run the original attention module
        output = self.original_attn_module(x)
        
        # Recompute attention weights manually
        if hasattr(self.original_attn_module, 'qkv'):
            # Extract Q, K, V from fused qkv layer
            qkv = self.original_attn_module.qkv(x)
            batch_size, seq_len, _ = x.shape
            
            # Reshape: (batch, seq_len, 3*num_heads*head_dim) 
            #      -> (batch, seq_len, 3, num_heads, head_dim)
            qkv = qkv.reshape(batch_size, seq_len, 3, 
                             self.original_attn_module.num_heads, -1)
            
            # Permute: (3, batch, num_heads, seq_len, head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # Split into Q, K, V
            
            # Compute attention: Q @ K^T, scaled, then softmax
            attn = (q @ k.transpose(-2, -1)) * self.original_attn_module.scale
            self.attn_weights = attn.softmax(dim=-1)  # Shape: [batch, heads, seq_len, seq_len]
        
        return output
```

**What this does:**
- Wraps each attention module in the ViT
- During forward pass, manually computes attention weights from Q, K, V
- Stores attention weights in `self.attn_weights`

**Why needed:**
- Standard ViT attention modules don't return attention weights
- We need these weights to see which patches the model attends to

---

### **Step 2: Replace All Attention Modules** (Lines 52-55)

```python
# Replace the attention module in each block with our wrapper
for i, block in enumerate(vit_model.blocks):
    if hasattr(block, 'attn'):
        block.attn = AttentionWithWeights(block.attn)
```

**What this does:**
- Iterates through all transformer blocks in the ViT
- Replaces each attention module with our wrapper
- Now all attention weights will be captured during forward pass

---

### **Step 3: Forward Pass to Capture Attention** (Lines 57-64)

```python
# Perform a forward pass to execute the wrapped modules and capture weights
with torch.no_grad():  # No gradients needed for inference
    _ = vit_model(image)  # Forward pass triggers attention computation

# Collect the captured attention weights from each block
for i, block in enumerate(vit_model.blocks):
    if hasattr(block.attn, 'attn_weights') and block.attn.attn_weights is not None:
        attention_maps[f"layer_{i}"] = block.attn.attn_weights.detach()
```

**What this does:**
- Runs the image through the model (triggers attention computation)
- Collects attention weights from all layers
- Stores them in `attention_maps` dictionary: `{"layer_0": weights, "layer_1": weights, ...}`

**Attention weights shape:**
- `[batch_size, num_heads, seq_len, seq_len]`
- Each `[i, j]` entry = attention from token `i` to token `j`
- `seq_len` = number of patches + 1 (for CLS token)

---

### **Step 4: Select Target Layer** (Lines 69-76)

```python
# Select the attention map from the specified layer
if layer_idx < 0:
    layer_idx = len(attention_maps) + layer_idx  # Handle negative indexing (-1 = last layer)
layer_name = f"layer_{layer_idx}"
if layer_name not in attention_maps:
    raise ValueError(f"Layer {layer_idx} not found...")

layer_attn = attention_maps[layer_name]  # Shape: [batch, heads, seq_len, seq_len]
```

**What this does:**
- Selects attention from a specific transformer layer (default: last layer)
- Last layer typically has the most task-relevant attention patterns

---

### **Step 5: Extract CLS Token Attention** (Lines 77-80)

```python
# Average attention across all heads
head_attn = layer_attn[0].mean(dim=0)  # [batch, heads, seq_len, seq_len] -> [seq_len, seq_len]
# Get attention from the [CLS] token to all other image patches
cls_attn = head_attn[0, 1:]  # [seq_len] -> [num_patches]
```

**What this does:**
1. **Average across heads**: Multi-head attention has multiple attention patterns. We average them:
   - `layer_attn[0]`: First (and only) sample in batch
   - `.mean(dim=0)`: Average across all attention heads
   - Result: `[seq_len, seq_len]` attention matrix

2. **Extract CLS token attention**: 
   - `head_attn[0]`: CLS token's attention to all tokens (including itself)
   - `head_attn[0, 1:]`: CLS token's attention to **image patches only** (skip CLS→CLS)
   - Result: `[num_patches]` vector where each value = how much CLS attends to that patch

**Why CLS token?**
- CLS token aggregates global information from all patches
- Its attention weights show which patches contribute most to the final prediction
- This is the standard approach for ViT saliency maps

---

### **Step 6: Reshape to 3D Spatial Grid** (Lines 82-96)

```python
# Reshape the 1D attention vector into a 3D volume
patches_per_dim = img_size[0] // patch_size  # 96 // 16 = 6 patches per dimension
total_patches = patches_per_dim ** 3  # 6^3 = 216 patches total

# Pad or truncate if the number of patches doesn't align
if cls_attn.shape[0] != total_patches:
    if cls_attn.shape[0] > total_patches:
        cls_attn = cls_attn[:total_patches]  # Truncate if too many
    else:
        padded = torch.zeros(total_patches, device=cls_attn.device)
        padded[:cls_attn.shape[0]] = cls_attn  # Pad with zeros if too few
        cls_attn = padded

# Reshape: [216] -> [6, 6, 6] (3D grid matching spatial layout)
cls_attn_3d = cls_attn.reshape(patches_per_dim, patches_per_dim, patches_per_dim)
cls_attn_3d = cls_attn_3d.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims: [1, 1, 6, 6, 6]
```

**What this does:**
- Converts 1D patch attention vector into 3D spatial grid
- For 96×96×96 image with 16×16×16 patches: 6×6×6 grid
- Handles mismatches (padding/truncation) if patch count doesn't align

**Spatial mapping:**
- Patch 0 → Position (0, 0, 0) in 3D grid
- Patch 1 → Position (0, 0, 1)
- ...
- Patch 215 → Position (5, 5, 5)

---

### **Step 7: Upsample to Full Image Resolution** (Lines 98-104)

```python
# Upsample the attention map to the full image resolution
upsampled_attn = torch.nn.functional.interpolate(
    cls_attn_3d,           # [1, 1, 6, 6, 6] - coarse attention map
    size=img_size,          # (96, 96, 96) - target size
    mode='trilinear',        # 3D interpolation
    align_corners=False
).squeeze()  # Remove batch and channel dims: [96, 96, 96]
```

**What this does:**
- Upsamples from 6×6×6 patch grid to 96×96×96 full resolution
- Uses trilinear interpolation to smoothly fill in between patches
- Result: Full-resolution 3D attention map matching input image size

**Why upsample?**
- Attention is computed at patch level (coarse)
- For visualization, we want pixel-level resolution
- Interpolation creates smooth heatmap overlay

---

### **Step 8: Normalize for Visualization** (Lines 106-109)

```python
# Normalize the map to [0, 1] for visualization
upsampled_attn = upsampled_attn.cpu().numpy()  # Convert to numpy
upsampled_attn = (upsampled_attn - upsampled_attn.min()) / (upsampled_attn.max() - upsampled_attn.min())
return upsampled_attn  # Shape: [96, 96, 96], values in [0, 1]
```

**What this does:**
- Min-max normalization: maps values to [0, 1] range
- 0 = lowest attention (model ignores this region)
- 1 = highest attention (model focuses most on this region)
- Ready for visualization as heatmap overlay

---

## Complete Flow Diagram

```
Input Image (96×96×96)
    ↓
ViT Forward Pass
    ↓
[Step 1-2] Hook attention modules → Capture attention weights
    ↓
[Step 3] Forward pass → attention_maps = {layer_0: [B,H,S,S], layer_1: [B,H,S,S], ...}
    ↓
[Step 4] Select layer (e.g., last layer)
    ↓
[Step 5] Extract CLS token attention → cls_attn = [216] (one value per patch)
    ↓
[Step 6] Reshape to 3D grid → cls_attn_3d = [6, 6, 6]
    ↓
[Step 7] Upsample → upsampled_attn = [96, 96, 96]
    ↓
[Step 8] Normalize [0, 1] → Saliency Map
    ↓
Save as NIfTI file
```

---

## Key Concepts

### **1. CLS Token Attention**
- CLS token is a special token added to the input sequence
- It aggregates information from all image patches
- Its attention weights indicate which patches are most important for the prediction

### **2. Multi-Head Attention Averaging**
- ViT uses multiple attention heads (typically 12)
- Each head may focus on different aspects
- Averaging gives a unified view of what the model attends to

### **3. Patch-to-Spatial Mapping**
- ViT divides image into non-overlapping patches (16×16×16 voxels each)
- Attention is computed at patch level, not pixel level
- Reshaping maps patch indices back to 3D spatial coordinates

### **4. Upsampling**
- Attention maps are coarse (6×6×6 for 96×96×96 image)
- Upsampling creates smooth, full-resolution visualization
- Trilinear interpolation ensures smooth transitions

---

## Usage in Main Function

```python
# Lines 134-141: Generate saliency map for each sample
saliency_map = extract_attention_map(
    vit_model,           # ViT backbone model
    input_tensor,        # Single 3D MRI volume [1, 1, 96, 96, 96]
    layer_idx=layer_idx, # Which transformer layer (-1 = last)
    img_size=(96, 96, 96),
    patch_size=16
)

# Lines 146-154: Save as NIfTI file
saliency_nifti = nib.Nifti1Image(saliency_map, np.eye(4))
nib.save(saliency_nifti, output_path)
```

---

## Interpretation

The saliency map shows:
- **High values (bright regions)**: Areas the model focuses on for prediction
- **Low values (dark regions)**: Areas the model ignores
- **Spatial localization**: Which anatomical regions are important

This is useful for:
- **Model interpretability**: Understanding what the model "sees"
- **Clinical validation**: Checking if model focuses on clinically relevant regions
- **Debugging**: Identifying if model attends to artifacts or irrelevant regions

---

## Limitations

1. **Patch-level granularity**: Attention is computed at patch level (16×16×16), not pixel level
2. **Single layer**: Only one transformer layer is visualized (typically last)
3. **CLS token only**: Doesn't show patch-to-patch attention, only CLS→patch
4. **No task-specific attention**: Uses generic attention, not task-specific gradients

---

## Summary

The saliency map generation process:
1. **Intercepts** attention weights during ViT forward pass
2. **Extracts** CLS token attention to image patches
3. **Reshapes** 1D patch attention to 3D spatial grid
4. **Upsamples** to full image resolution
5. **Normalizes** for visualization

The result is a 3D heatmap showing which brain regions the model focuses on when making predictions.

