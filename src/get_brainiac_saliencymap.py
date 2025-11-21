import torch
import numpy as np
import random
import yaml
import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import nibabel as nib
from dataset import BrainAgeDataset, get_validation_transform
from load_brainiac import load_brainiac

# Fix random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def extract_attention_map(vit_model, image, layer_idx=-1, img_size=(96, 96, 96), patch_size=16):
    """
    Extracts the attention map from a Vision Transformer (ViT) model.
    
    This function wraps the attention blocks of the ViT to capture the attention
    weights during a forward pass. It then processes these weights to generate
    a 3D saliency map corresponding to the model's focus on the input image.
    """
    attention_maps = {}

    # A wrapper class to intercept and store attention weights from a ViT block.
    class AttentionWithWeights(torch.nn.Module):
        def __init__(self, original_attn_module):
            super().__init__()
            self.original_attn_module = original_attn_module
            self.attn_weights = None

        def forward(self, x):
            # MONAI ViT with save_attn=True stores attention matrices in att_mat
            # First, run the forward pass to populate att_mat
            output = self.original_attn_module(x)
            
            # Try to get saved attention matrix from MONAI ViT
            if hasattr(self.original_attn_module, 'att_mat'):
                att_mat = self.original_attn_module.att_mat
                if att_mat is not None:
                    # att_mat shape: [batch, heads, seq_len, seq_len] or [heads, seq_len, seq_len]
                    self.attn_weights = att_mat.detach() if isinstance(att_mat, torch.Tensor) else att_mat
                    if len(self.attn_weights.shape) == 3:
                        # Add batch dimension if missing: [heads, seq_len, seq_len] -> [1, heads, seq_len, seq_len]
                        self.attn_weights = self.attn_weights.unsqueeze(0)
                    return output
            
            # Fallback: recompute attention if att_mat not available
            batch_size, seq_len, _ = x.shape
            
            # qkv is a nn.Module (Linear layer), not a callable function
            if hasattr(self.original_attn_module, 'qkv'):
                try:
                    qkv = self.original_attn_module.qkv(x)  # qkv is a Linear layer, call it directly
                    qkv = qkv.reshape(batch_size, seq_len, 3, self.original_attn_module.num_heads, -1)
                    qkv = qkv.permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    scale = self.original_attn_module.scale if hasattr(self.original_attn_module, 'scale') else (q.shape[-1] ** -0.5)
                    attn = (q @ k.transpose(-2, -1)) * scale
                    self.attn_weights = attn.softmax(dim=-1)
                except Exception as e:
                    # If qkv computation fails, attn_weights remains None
                    pass
            
            return output

    # Replace the attention module in each block with our wrapper
    for i, block in enumerate(vit_model.blocks):
        if hasattr(block, 'attn'):
            block.attn = AttentionWithWeights(block.attn)

    # Perform a forward pass to execute the wrapped modules and capture weights
    with torch.no_grad():
        _ = vit_model(image)

    # Collect the captured attention weights from each block
    for i, block in enumerate(vit_model.blocks):
        if hasattr(block.attn, 'attn_weights') and block.attn.attn_weights is not None:
            attention_maps[f"layer_{i}"] = block.attn.attn_weights.detach()

    if not attention_maps:
        raise RuntimeError("Could not extract any attention maps. Please check the ViT model structure.")

    # Select the attention map from the specified layer
    if layer_idx < 0:
        layer_idx = len(attention_maps) + layer_idx
    layer_name = f"layer_{layer_idx}"
    if layer_name not in attention_maps:
        raise ValueError(f"Layer {layer_idx} not found. Available layers: {list(attention_maps.keys())}")

    layer_attn = attention_maps[layer_name]
    # Average attention across all heads
    head_attn = layer_attn[0].mean(dim=0)
    # Get attention from the [CLS] token to all other image patches
    cls_attn = head_attn[0, 1:]

    # Reshape the 1D attention vector into a 3D volume
    patches_per_dim = img_size[0] // patch_size
    total_patches = patches_per_dim ** 3
    
    # Pad or truncate if the number of patches doesn't align
    if cls_attn.shape[0] != total_patches:
        if cls_attn.shape[0] > total_patches:
            cls_attn = cls_attn[:total_patches]
        else:
            padded = torch.zeros(total_patches, device=cls_attn.device)
            padded[:cls_attn.shape[0]] = cls_attn
            cls_attn = padded

    cls_attn_3d = cls_attn.reshape(patches_per_dim, patches_per_dim, patches_per_dim)
    cls_attn_3d = cls_attn_3d.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

    # Upsample the attention map to the full image resolution
    upsampled_attn = torch.nn.functional.interpolate(
        cls_attn_3d,
        size=img_size,
        mode='trilinear',
        align_corners=False
    ).squeeze()

    # Normalize the map to [0, 1] for visualization
    upsampled_attn = upsampled_attn.cpu().numpy()
    upsampled_attn = (upsampled_attn - upsampled_attn.min()) / (upsampled_attn.max() - upsampled_attn.min())
    return upsampled_attn

def generate_saliency_maps(model, data_loader, output_dir, device, layer_idx=-1):
    """Generate saliency maps using ViT attention mechanism"""
    model.eval()
    
    # Extract the ViT backbone from the BrainIAC model
    vit_model = model.backbone
    
    success_count = 0
    error_count = 0
    
    for sample in tqdm(data_loader, desc="Generating ViT attention maps"):
        inputs = sample['image'].to(device)
        labels = sample['label']
        
        # Get patient ID from the file path if available
        # This is a simplified approach - you might need to modify based on your dataset structure
        batch_size = inputs.shape[0]
        
        for i in range(batch_size):
            input_tensor = inputs[i:i+1]  # Single sample
            label = labels[i].item()
            
            # Generate attention map
            try:
                saliency_map = extract_attention_map(
                    vit_model, 
                    input_tensor, 
                    layer_idx=layer_idx, 
                    img_size=(96, 96, 96), 
                    patch_size=16
                )
                
                # Save input image and saliency map
                inputs_np = input_tensor.squeeze().cpu().detach().numpy()
                
                input_nifti = nib.Nifti1Image(inputs_np, np.eye(4))
                saliency_nifti = nib.Nifti1Image(saliency_map, np.eye(4))
                
                # Create unique filename
                filename_base = f"sample_{i:04d}_label_{label:.2f}"
                
                # Save files
                nib.save(input_nifti, os.path.join(output_dir, f"{filename_base}_image.nii.gz"))
                nib.save(saliency_nifti, os.path.join(output_dir, f"{filename_base}_saliencymap_layer{layer_idx}.nii.gz"))
                success_count += 1
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                error_count += 1
                continue
    
    print(f"\nSummary: {success_count} saliency maps generated successfully, {error_count} errors")

def main():
    parser = argparse.ArgumentParser(description='Generate ViT attention-based saliency maps for medical images')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to the ViT BrainIAC model checkpoint (default: checkpoints/BrainIAC.ckpt)')
    parser.add_argument('--input_csv', type=str, required=True,
                      help='Path to the input CSV file containing image paths')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save saliency maps')
    parser.add_argument('--root_dir', type=str, required=True,
                      help='Root directory containing the image data')
    parser.add_argument('--layer', type=int, default=-1,
                      help='Transformer layer index to visualize (-1 for last layer)')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for inference (default: 1)')
    parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of workers for data loading (default: 1)')
    
    args = parser.parse_args()
    device = torch.device("cpu")  # Use CPU for saliency generation to avoid memory issues
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize dataset and dataloader with validation transforms
    dataset = BrainAgeDataset(
        csv_path=args.input_csv,
        root_dir=args.root_dir,
        transform=get_validation_transform()
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False  # Set to False for CPU processing
    )
    
    # Load ViT BrainIAC model
    model = load_brainiac(args.checkpoint, device)
    model = model.to(device)
    
    # Generate saliency maps
    generate_saliency_maps(model, dataloader, args.output_dir, device, args.layer)

if __name__ == "__main__":
    main()