# How to Run and Test the Application

## Step 1: Install Dependencies

First, make sure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

If you encounter version conflicts (like tokenizers), you may need to upgrade:

```bash
pip install --upgrade transformers tokenizers diffusers
```

## Step 2: Install the Package

Install the package in development mode so Python can find it:

```bash
pip install -e .
```

## Step 3: Quick Test

Run a simple test with just 2 images to verify everything works:

```bash
python -m synthetic_data_gen.interface.cli run --generation-model stable_diffusion --num-images 2 --action classify --classification-model resnet50 --image-size 256 --batch-size 1
```

**What this does:**
- Generates 2 synthetic skin lesion images using Stable Diffusion
- Classifies them using ResNet50
- Shows classification results

**Note:** The first run will download model weights (can be several GB), so it may take a while.

## Step 4: Test with More Images

Once the basic test works, try generating more images:

```bash
python -m synthetic_data_gen.interface.cli run --generation-model stable_diffusion --num-images 10 --action classify --classification-model resnet50
```

## Step 5: Test Retraining

To test the retraining functionality:

```bash
python -m synthetic_data_gen.interface.cli run --generation-model stable_diffusion --num-images 20 --action retrain --classification-model resnet50 --num-epochs 5
```

## Alternative: Using Kedro Directly

You can also use Kedro commands directly:

```bash
# Generate images
kedro run --pipeline data_generation --params image_generation.num_images=5

# Classify images
kedro run --pipeline classification

# Train model
kedro run --pipeline training --params training.num_epochs=5
```

## Troubleshooting

### "Module not found" errors
- Make sure you ran `pip install -e .`
- Check that you're in the project root directory

### GPU out of memory
- Reduce `--image-size` to 256 or 128
- Reduce `--batch-size` to 1
- The models will run on CPU if GPU is not available (slower)

### Model download issues
- Make sure you have internet connection
- First download can take 10-20 minutes depending on connection
- Models are cached after first download

### Check if everything is installed correctly
```bash
python test_run.py
```

This will verify all imports work correctly.

