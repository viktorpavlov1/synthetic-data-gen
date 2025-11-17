# Quick Start Guide

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install the package** (optional, but recommended):
   ```bash
   pip install -e .
   ```

## Quick Test

### Option 1: Using the CLI (Recommended)

Generate a small number of images and classify them:

```bash
python -m synthetic_data_gen.interface.cli run \
    --generation-model stable_diffusion \
    --num-images 5 \
    --action classify \
    --classification-model resnet50 \
    --image-size 256 \
    --batch-size 2
```

**Note**: The first run will download model weights, which may take some time.

### Option 2: Using Kedro directly

```bash
# Run data generation only
kedro run --pipeline data_generation --params image_generation.num_images=5

# Then run classification
kedro run --pipeline classification
```

## Test with Minimal Resources

For a quick test with minimal GPU memory usage:

```bash
python -m synthetic_data_gen.interface.cli run \
    --generation-model stable_diffusion \
    --num-images 2 \
    --action classify \
    --classification-model resnet50 \
    --image-size 256 \
    --batch-size 1
```

## Troubleshooting

### If you get "Module not found" errors:
```bash
# Make sure you're in the project root directory
cd c:\Users\Victor\Desktop\synthetic-data-gen

# Install in development mode
pip install -e .
```

### If image generation fails:
- Make sure you have enough GPU memory (or use CPU mode)
- Try reducing `--image-size` to 256 or `--batch-size` to 1
- Check that you have internet connection for downloading models

### If classification fails:
- The models will download ImageNet pretrained weights automatically
- Make sure PyTorch is installed correctly: `python -c "import torch; print(torch.__version__)"`

## Expected Output

When running successfully, you should see:
1. Model loading messages
2. Progress bars for image generation
3. Classification results with confidence scores
4. Summary statistics

