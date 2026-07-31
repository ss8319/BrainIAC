# Why Are Saliency Maps Different Across Tasks?

## The Key Question

> "They all share the same ViT foundation model, though they often have different task heads. So why are the saliency maps different?"

## The Answer: Task-Specific Fine-Tuning

Even though all models use the **same ViT backbone architecture**, the **backbone weights are different** because they were **fine-tuned on different tasks**. This causes different attention patterns, resulting in different saliency maps.

---

## 1. The Architecture is the Same, But Weights Are Different

### Shared Architecture ✅
All models use the same ViT backbone structure:
- Same number of layers (e.g., 12 transformer blocks)
- Same attention heads (e.g., 12 heads)
- Same embedding dimensions (e.g., 768)
- Same patch size (16×16×16)

### Different Weights ❌
But the **weights** (parameters) are different:
- **Brain Age model**: Backbone fine-tuned for age prediction
- **AD Classification model**: Backbone fine-tuned for AD/CN classification
- **IDH model**: Backbone fine-tuned for IDH mutation prediction

### Code Evidence

```python
# All scripts extract from backbone:
vit_model = model.backbone.backbone  # Same architecture

# But the backbone weights are task-specific!
# Brain Age checkpoint → Brain Age backbone weights
# AD checkpoint → AD backbone weights
# IDH checkpoint → IDH backbone weights
```

---

## 2. How Fine-Tuning Changes Attention Patterns

### Pre-Training (Foundation Model)
```
ViT Backbone (pre-trained on large dataset)
├─ Learns general image features
├─ Attention patterns: General object recognition
└─ Weights: W_pretrained
```

### Fine-Tuning for Brain Age (Regression)
```
ViT Backbone + Age Regressor
├─ Fine-tunes backbone weights: W_pretrained → W_brainage
├─ Attention patterns: Focus on age-related features
│  └─ May attend to: Brain volume, atrophy patterns
└─ Saliency map: Highlights age-relevant regions
```

### Fine-Tuning for AD Classification
```
ViT Backbone + AD Classifier
├─ Fine-tunes backbone weights: W_pretrained → W_ad
├─ Attention patterns: Focus on AD-related features
│  └─ May attend to: Hippocampus, temporal lobes, amyloid deposits
└─ Saliency map: Highlights AD-relevant regions
```

### Fine-Tuning for IDH Mutation
```
ViT Backbone + IDH Classifier
├─ Fine-tunes backbone weights: W_pretrained → W_idh
├─ Attention patterns: Focus on IDH-related features
│  └─ May attend to: Tumor regions, specific mutations
└─ Saliency map: Highlights IDH-relevant regions
```

---

## 3. Why Backbone Weights Change During Fine-Tuning

### The Training Process

When you fine-tune a model:

1. **Start with pre-trained weights**: `W_pretrained`
2. **Add task head**: Classifier or regressor
3. **Train end-to-end**: 
   - Task head learns to predict from features
   - **Backbone weights are updated** to produce better features for the task
4. **Result**: `W_pretrained` → `W_task_specific`

### Example: Brain Age vs AD Classification

**Brain Age (Regression):**
```
Loss = MSE(predicted_age, true_age)
Backbone learns to extract features that correlate with age:
- Brain volume changes
- Atrophy patterns
- General aging markers
```

**AD Classification:**
```
Loss = CrossEntropy(predicted_class, true_class)
Backbone learns to extract features that distinguish AD from CN:
- Hippocampal atrophy
- Temporal lobe changes
- Disease-specific patterns
```

**Result**: Different features → Different attention patterns → Different saliency maps

---

## 4. Attention Patterns Are Task-Specific

### What Attention Represents

Attention weights show **which patches the model focuses on** to make its prediction. Different tasks require focusing on different regions:

### Brain Age Prediction
- **Focus**: Age-related changes throughout the brain
- **Attention**: May be more uniform, focusing on overall brain structure
- **Saliency map**: Highlights general aging markers

### AD Classification
- **Focus**: Disease-specific regions (hippocampus, temporal lobes)
- **Attention**: Highly focused on specific anatomical regions
- **Saliency map**: Highlights AD-affected regions

### IDH Mutation Prediction
- **Focus**: Tumor regions and mutation-specific features
- **Attention**: Focused on tumor boundaries and characteristics
- **Saliency map**: Highlights tumor-relevant regions

---

## 5. Code Verification

### All Scripts Extract from Backbone

```python
# get_brainiac_saliencymap.py (Line 116)
vit_model = model.backbone  # Base BrainIAC backbone

# generate_brainage_vit_saliency.py (Line 144)
vit_model = model.backbone.backbone  # Brain Age fine-tuned backbone

# generate_idh_vit_saliency.py (Line 165)
vit_model = model.backbone.backbone  # IDH fine-tuned backbone
```

**Key Point**: They all extract attention from the backbone, but:
- The backbone weights are **different** (task-specific fine-tuning)
- The attention patterns are **different** (learned for different tasks)
- The saliency maps are **different** (reflect task-specific focus)

---

## 6. Visual Example

### Same Image, Different Tasks

**Input**: Same 3D MRI brain scan

**Brain Age Model:**
```
Attention Pattern:
- Moderate attention to: Whole brain volume
- High attention to: Ventricular size (age marker)
- Low attention to: Specific disease regions

Saliency Map:
[Uniform highlighting of brain volume]
```

**AD Classification Model:**
```
Attention Pattern:
- High attention to: Hippocampus (AD marker)
- High attention to: Temporal lobes (AD marker)
- Low attention to: Other regions

Saliency Map:
[Bright spots in hippocampus and temporal lobes]
```

**IDH Mutation Model:**
```
Attention Pattern:
- High attention to: Tumor boundaries
- High attention to: Mutation-specific features
- Low attention to: Normal brain tissue

Saliency Map:
[Bright spots in tumor regions]
```

---

## 7. Why Not Just Use the Task Head?

### You Might Wonder:
> "If the task head is different, shouldn't we visualize the task head's gradients instead?"

### The Answer:
**Attention-based saliency maps use the backbone's attention**, not the task head, because:

1. **Attention is interpretable**: Shows spatial focus directly
2. **Backbone contains spatial information**: Task head is just a classifier
3. **Standard practice**: CLS token attention is the standard ViT interpretability method

### Alternative: Gradient-Based Methods

If you wanted task-head-specific saliency:
- Use **Grad-CAM** or **Integrated Gradients**
- These use gradients from the task head
- Show what actually influences the prediction
- But computationally more expensive

---

## 8. The Foundation Model vs Fine-Tuned Model

### Foundation Model (Pre-Trained)
```
ViT Backbone (BrainIAC pre-trained)
├─ Weights: W_foundation
├─ Attention: General medical image features
└─ Saliency: Generic anatomical regions
```

### Fine-Tuned Models (Task-Specific)
```
ViT Backbone (Fine-tuned for task)
├─ Weights: W_foundation → W_task (updated during fine-tuning)
├─ Attention: Task-specific features
└─ Saliency: Task-relevant regions
```

**Key Insight**: Fine-tuning **updates the backbone weights**, not just the task head!

---

## 9. What If We Used the Same Backbone Weights?

### Hypothetical: Same Weights, Different Task Heads

If we **froze the backbone** and only trained task heads:

```python
# Freeze backbone during training (from train_lightning_*.py)
if config['train']['freeze'] == 'yes':
    for param in backbone.parameters():
        param.requires_grad = False
```

**Result**:
- Same backbone weights → Same attention patterns
- Different task heads → Different predictions
- **Same saliency maps** (because attention comes from backbone)

### In Practice: Backbone is Usually Fine-Tuned

**Most training configs** set `freeze: 'no'` (or don't specify), meaning:
- Backbone weights ARE updated during fine-tuning
- This causes different attention patterns
- This is why saliency maps differ across tasks

**Exception**: If `freeze: 'yes'` (linear probing):
- Backbone weights are frozen
- Only task head is trained
- Saliency maps would be the same (but task performance likely worse)

---

## 10. Summary

### Why Saliency Maps Differ Across Tasks

1. **Same architecture, different weights**: All use ViT backbone, but weights are task-specific
2. **Fine-tuning updates backbone**: Not just the task head, but the backbone too
3. **Task-specific attention**: Backbone learns to attend to task-relevant regions
4. **Different focus regions**: Each task requires focusing on different brain regions

### The Key Takeaway

> **Even though the architecture is the same, the learned attention patterns are different because the backbone weights are fine-tuned for each specific task. This is why saliency maps differ across tasks.**

### Code Confirmation

All scripts extract attention from `model.backbone.backbone`, but:
- Brain Age model → Brain Age fine-tuned backbone → Age-focused attention
- AD model → AD fine-tuned backbone → Disease-focused attention
- IDH model → IDH fine-tuned backbone → Tumor-focused attention

**The backbone is the same structure, but the weights (and thus attention patterns) are task-specific!**

---

## Further Reading

- **Transfer Learning**: How fine-tuning updates pre-trained weights
- **Attention Mechanisms**: How attention patterns emerge during training
- **Task-Specific Representations**: How models learn task-relevant features

