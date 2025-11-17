# Dataset Setup Guide

## Using Your 10,000 Image Dataset

You can use your dataset to fine-tune the image generation models for better quality. The dataset from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) can be integrated into the pipeline.

## Dataset Organization

The dataset loader supports three common structures:

### 1. Flat Structure (All images in one folder)
```
dataset/
  ├── image1.jpg
  ├── image2.jpg
  ├── image3.jpg
  └── ...
```

### 2. By Class Structure (Images organized by class)
```
dataset/
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
dataset/
  ├── train/
  │   └── images...
  ├── val/
  │   └── images...
  └── test/
      └── images...
```

## Setup Steps

### 1. Download Your Dataset

Download the dataset from the Harvard Dataverse link and extract it to:
```
data/00_external/dataset/
```

### 2. Organize Images (if needed)

If your dataset isn't in one of the supported structures, organize it accordingly.

### 3. Use the Web Interface

1. Launch the interface: `python run_interface.py`
2. Go to the **"🎨 Fine-Tune Generator"** tab
3. Set your dataset path (e.g., `data/00_external/dataset`)
4. Select the dataset structure
5. Click **"📂 Load Dataset"** to preview
6. Configure fine-tuning settings
7. Click **"▶️ Start Fine-Tuning"**

### 4. Use Fine-Tuned Model

After fine-tuning:
1. In the **"🎨 Fine-Tune Generator"** tab, select your fine-tuned model
2. Go back to **"🚀 Run Pipeline"** tab
3. Generate images - they will use the fine-tuned model automatically

## Fine-Tuning Parameters

- **Number of Epochs**: 10-20 recommended for good results
- **Learning Rate**: 0.0001 is a good starting point
- **LoRA Rank**: 4-8 for balance between quality and speed
- **Max Images**: Start with 1000-2000 for testing, use all 10,000 for production

## Important Notes

1. **Full Fine-Tuning**: The current implementation provides the framework. For production fine-tuning, you may need to use diffusers training scripts directly.

2. **GPU Memory**: Fine-tuning requires significant GPU memory. Use smaller batch sizes if needed.

3. **Time**: Fine-tuning 10,000 images can take several hours depending on your hardware.

4. **Quality Improvement**: Fine-tuned models will generate images more similar to your dataset, which should improve classification accuracy.

## Alternative: Using the Dataset for Classification Training

You can also use your dataset directly for training classification models:

1. Place your dataset in the appropriate structure
2. The training pipeline can be extended to load real images alongside synthetic ones
3. This improves model performance by training on real medical images

## Dataset Format Support

The loader supports:
- `.jpg`, `.jpeg`, `.png`, `.bmp` image formats
- ISIC dataset format (with metadata CSV)
- Custom structures (can be extended)

