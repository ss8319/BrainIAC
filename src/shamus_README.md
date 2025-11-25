# Simple .md to keep handy commands

## Generate Saliency Maps

Run from the BrainIAC root directory:

```bash
uv run python src/get_brainiac_saliencymap.py \
    --checkpoint src/checkpoints/BrainIAC.ckpt \
    --input_csv /home/ssim0068/data/multimodal-dataset/all_icbm.csv \
    --output_dir src/saliency_map \
    --root_dir /home/ssim0068/data/multimodal-dataset/all_icbm/images
```

### Optional Arguments:
- `--layer`: Layer index to visualize (default: -1 for last layer)
- `--batch_size`: Batch size for inference (default: 1)
- `--num_workers`: Number of workers for data loading (default: 1)