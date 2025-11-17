# Fine-Tuning Guide: Using Your Dataset to Improve Image Generation

## Overview

You can use your 10,000 image dataset to fine-tune the image generation models (Stable Diffusion, QWEN, Flux) for better quality. Fine-tuning adapts the models to generate images more similar to your specific dataset.

## Quick Start

### Option 1: Using the Web Interface (Recommended)

1. **Prepare your dataset:**
   - Download your dataset from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
   - Extract it to: `data/00_external/dataset/`
   - Organize images in one of the supported structures (see below)

2. **Launch the interface:**
   ```bash
   python run_interface.py
   ```

3. **Fine-tune the model:**
   - Go to the **"🎨 Fine-Tune Generator"** tab
   - Set dataset path: `data/00_external/dataset`
   - Select dataset structure
   - Click **"📂 Load Dataset"** to preview
   - Configure settings (epochs, learning rate, etc.)
   - Click **"▶️ Start Fine-Tuning"**

4. **Use the fine-tuned model:**
   - In the same tab, select your fine-tuned model
   - Go to **"🚀 Run Pipeline"** tab
   - Generate images - they'll automatically use the fine-tuned model

### Option 2: Using the Command Line

```bash
# Prepare dataset
python prepare_dataset.py --dataset-path data/00_external/dataset --structure flat --preview

# Fine-tune using Kedro
kedro run --pipeline fine_tuning --params fine_tuning.dataset_path=data/00_external/dataset
```

## Dataset Structures

The loader supports three common structures:

### 1. Flat Structure (All images in one folder)
```
data/00_external/dataset/
  ├── image1.jpg
  ├── image2.jpg
  ├── image3.jpg
  └── ... (10,000 images)
```

### 2. By Class Structure
```
data/00_external/dataset/
  ├── benign/
  │   ├── image1.jpg
  │   └── ...
  ├── malignant/
  │   ├── image2.jpg
  │   └── ...
  └── ...
```

### 3. Train/Val/Test Structure
```
data/00_external/dataset/
  ├── train/
  │   └── images...
  ├── val/
  │   └── images...
  └── test/
      └── images...
```

## Fine-Tuning Parameters

### Recommended Settings for 10,000 Images

**For Testing (Faster):**
- Max Images: 1,000-2,000
- Epochs: 5-10
- Learning Rate: 0.0001
- LoRA Rank: 4

**For Production (Better Quality):**
- Max Images: 10,000 (all)
- Epochs: 10-20
- Learning Rate: 0.0001
- LoRA Rank: 8

### Parameter Explanations

- **Number of Epochs**: How many times the model sees the entire dataset
  - More epochs = better quality but longer training
  - Start with 10, increase if needed

- **Learning Rate**: How fast the model learns
  - Lower = more stable but slower
  - 0.0001 is a good starting point

- **LoRA Rank**: Number of trainable parameters
  - Higher rank = more capacity but slower
  - 4-8 is a good range

- **Max Images**: Limit dataset size for faster training
  - Use subset for testing, all images for production

## Important Notes

### Current Implementation

The current fine-tuning implementation provides:
- ✅ Dataset loading and preparation
- ✅ Configuration management
- ✅ Framework for fine-tuning
- ⚠️ **Simplified training loop** (for full training, use diffusers scripts)

### For Production Fine-Tuning

For actual fine-tuning with 10,000 images, you may want to use the official diffusers training scripts:

```bash
# Example using diffusers training script
accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="data/00_external/dataset" \
  --output_dir="data/06_models/fine_tuned_sd" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --max_train_steps=10000 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0
```

The web interface saves the configuration, which you can use with these scripts.

## Benefits of Fine-Tuning

1. **Better Quality**: Generated images will be more similar to your dataset
2. **Domain-Specific**: Models learn medical imaging characteristics
3. **Improved Classification**: Better synthetic data improves classifier training
4. **Consistency**: More consistent with real medical images

## Troubleshooting

### Dataset Not Loading
- Check the path is correct
- Verify images are in supported formats (.jpg, .png, .bmp)
- Try different structure options

### Out of Memory
- Reduce batch size to 1
- Use fewer images for testing
- Reduce image size to 256

### Fine-Tuning Takes Too Long
- Use a subset of images (1000-2000) for testing
- Reduce number of epochs
- Use lower LoRA rank

## Next Steps

After fine-tuning:
1. Compare generated images before/after fine-tuning
2. Use fine-tuned model for generating training data
3. Retrain classification models with fine-tuned synthetic data
4. Evaluate improvement in classification accuracy

