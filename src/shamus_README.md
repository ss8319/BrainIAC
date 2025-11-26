# Simple .md to keep handy commands

## Generate Saliency Maps

Run from the BrainIAC root directory:
```bash
cd src/mri/BrainIAC
```


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

sample_0000 match Row 1 of your CSV (126_S_0606, Label 1.00).
sample_0001 match Row 2 (031_S_1209, Label 1.00).
sample_0002 match Row 3 (023_S_0926, Label 0.00).