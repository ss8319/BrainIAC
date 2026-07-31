# What is CLS Token Attention? A Complete Explanation

## The Big Picture

**CLS token** (Classification token) is a special token added to Vision Transformers that acts as a "summary" of the entire image. Its attention weights tell us which image patches the model considers most important.

---

## 1. What is a CLS Token?

### In Natural Language Processing (Origin)
CLS token was first used in BERT (a language model):
- Added at the **beginning** of the input sequence
- Example: `[CLS] The cat sat on the mat [SEP]`
- After processing, the CLS token's embedding contains a **summary** of the entire sentence
- Used for classification tasks (sentiment, etc.)

### In Vision Transformers (ViT)
Same concept, but for images:
- Image is divided into **patches** (e.g., 16×16×16 voxels each)
- Each patch becomes a "token" in the sequence
- **CLS token is added as the first token** in the sequence

### Visual Representation

```
Input Image (96×96×96) divided into patches:
┌─────┬─────┬─────┬─────┬─────┬─────┐
│ P0  │ P1  │ P2  │ P3  │ ... │ P215│
└─────┴─────┴─────┴─────┴─────┴─────┘

After adding CLS token:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ CLS │ P0  │ P1  │ P2  │ P3  │ ... │ P215│
└─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  ↑
  Special token that aggregates information
```

**Sequence length**: 1 (CLS) + 216 (patches) = 217 tokens

---

## 2. How Does Attention Work with CLS Token?

### Self-Attention Mechanism

In a transformer, each token can "attend" to (look at) all other tokens, including itself.

**Attention Matrix** (for one head):
```
        CLS   P0   P1   P2   ...  P215
CLS    [0.1   0.3  0.2  0.1   ...  0.05]  ← CLS attends to patches
P0     [0.05  0.1  0.2  0.3   ...  0.1 ]
P1     [0.1   0.2  0.15 0.25  ...  0.15]
...
P215   [0.05  0.1  0.15 0.2   ...  0.1 ]
```

**Key insight**: The **first row** (`CLS → all tokens`) shows how much CLS token attends to each patch.

### What Does "Attention" Mean?

Attention is a **weighted sum**:
- Higher attention weight = CLS token "pays more attention" to that patch
- Lower attention weight = CLS token "ignores" that patch

**Mathematically:**
```
CLS_embedding = Σ(attention_weight[i] × patch_embedding[i])
```

Where:
- `attention_weight[i]` = how much CLS attends to patch `i`
- `patch_embedding[i]` = features from patch `i`
- Sum is over all patches

---

## 3. Why Use CLS Token?

### Problem: How to Get a Single Prediction from Multiple Patches?

An image has **many patches** (216 in our case), but we need **one prediction** (e.g., "AD" or "CN").

**Solution Options:**

1. **Average all patch embeddings** ❌
   - Loses spatial information
   - All patches treated equally

2. **Use last patch's embedding** ❌
   - Arbitrary choice
   - May not be representative

3. **Use CLS token** ✅
   - Learns to aggregate information from all patches
   - Can selectively focus on important patches
   - Standard approach in transformers

### How CLS Token Learns

During training:
1. CLS token starts as a **learnable embedding** (random initialization)
2. Through attention, it learns to **aggregate information** from relevant patches
3. The final CLS embedding is used for classification:
   ```
   CLS_embedding → Classifier → Prediction
   ```

---

## 4. CLS Token Attention for Saliency Maps

### The Key Idea

**If CLS token's attention to a patch is high, that patch is important for the prediction.**

### Why This Works

1. **CLS token aggregates information** from all patches
2. **Attention weights** show which patches contribute most
3. **High attention** = patch is important for the final prediction
4. **Low attention** = patch is less relevant

### Example

For an AD (Alzheimer's Disease) classification task:
- CLS token might have **high attention** to:
  - Hippocampus (memory region)
  - Temporal lobes (affected in AD)
- CLS token might have **low attention** to:
  - Skull (not relevant)
  - Background (empty space)

---

## 5. Code Walkthrough

### Step 1: Extract Attention Matrix

```python
# After forward pass, we have attention weights
layer_attn = attention_maps["layer_11"]  # Last layer
# Shape: [batch=1, heads=12, seq_len=217, seq_len=217]
```

### Step 2: Average Across Heads

```python
head_attn = layer_attn[0].mean(dim=0)
# Shape: [seq_len=217, seq_len=217]
# Average of 12 attention heads
```

### Step 3: Extract CLS Token's Attention

```python
cls_attn = head_attn[0, 1:]  # First row, skip CLS→CLS
# Shape: [216] - one value per patch
# Values: [0.3, 0.1, 0.05, 0.4, ...]
#         ↑    ↑    ↑     ↑
#        P0   P1   P2    P3  (attention weights)
```

**Interpretation:**
- `cls_attn[0] = 0.3` → CLS attends to patch 0 with weight 0.3
- `cls_attn[3] = 0.4` → CLS attends to patch 3 with weight 0.4 (more important!)

### Step 4: Map to Spatial Locations

```python
# Reshape to 3D grid
cls_attn_3d = cls_attn.reshape(6, 6, 6)
# Now we know which spatial regions are important
```

---

## 6. Visual Example

### Input Image
```
3D MRI Brain (96×96×96)
├─ Patch 0: [0:16, 0:16, 0:16]   (background)
├─ Patch 1: [16:32, 0:16, 0:16]  (background)
├─ Patch 50: [0:16, 32:48, 48:64] (hippocampus) ← Important!
├─ Patch 100: [32:48, 48:64, 32:48] (temporal lobe) ← Important!
└─ ...
```

### CLS Token Attention Weights
```
Patch 0:  0.01  (background - ignored)
Patch 1:  0.02  (background - ignored)
Patch 50: 0.15  (hippocampus - HIGH attention!)
Patch 100: 0.12 (temporal lobe - HIGH attention!)
...
```

### Resulting Saliency Map
```
Bright regions = High CLS attention = Important for prediction
Dark regions = Low CLS attention = Less relevant
```

---

## 7. Why Not Use Patch-to-Patch Attention?

You might wonder: "Why not visualize how patches attend to each other?"

### CLS Token Attention (What we use) ✅
- **Single vector**: One attention value per patch
- **Task-relevant**: CLS token is trained for the specific task
- **Interpretable**: Directly shows importance for prediction
- **Standard**: Widely used in ViT interpretability

### Patch-to-Patch Attention ❌
- **Complex matrix**: 216×216 attention values
- **Hard to visualize**: Need to aggregate somehow
- **Less task-specific**: Shows general relationships, not task importance

---

## 8. Limitations

### 1. **Single Layer**
- We only visualize one layer (typically last)
- Earlier layers might have different attention patterns

### 2. **Patch-Level Granularity**
- Attention is at patch level (16×16×16 voxels)
- Can't see pixel-level details
- Upsampling creates smooth visualization but doesn't add real detail

### 3. **Averaging Across Heads**
- Each attention head might focus on different aspects
- Averaging might lose some information

### 4. **Not Task-Specific Gradients**
- Attention is a forward-pass mechanism
- Doesn't use gradients (like Grad-CAM)
- May not perfectly align with what actually influences the prediction

---

## 9. Comparison with Other Methods

### CLS Token Attention (This method)
- ✅ Forward-pass only (fast)
- ✅ No gradients needed
- ✅ Shows what model "looks at"
- ❌ Patch-level granularity
- ❌ Single layer only

### Grad-CAM / Gradient-based methods
- ✅ Pixel-level granularity
- ✅ Task-specific (uses gradients)
- ❌ Requires backward pass (slower)
- ❌ More computationally expensive

### PCA Feature Maps (Your other method)
- ✅ Works with any model (not just ViT)
- ✅ Shows feature variations
- ❌ Less directly interpretable
- ❌ Doesn't show "importance" directly

---

## 10. Summary

**CLS Token Attention** is:
1. A **special token** added to the input sequence
2. That **aggregates information** from all image patches
3. Through **attention weights** that show which patches are important
4. Used for **saliency maps** by visualizing these attention weights
5. A **standard approach** in Vision Transformer interpretability

**Key Takeaway:**
> CLS token attention weights tell us which image regions the model considers most important for its prediction. High attention = important region, Low attention = less relevant region.

---

## Further Reading

- **Original ViT Paper**: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
- **BERT Paper**: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- **Attention Visualization**: "Attention Is All You Need" (Vaswani et al., 2017)

