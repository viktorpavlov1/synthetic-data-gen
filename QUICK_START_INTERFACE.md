# Quick Start: Web Interface

## 🚀 Launch the Interface

Just run this one command:

```bash
python run_interface.py
```

The web interface will automatically open in your browser at `http://localhost:8501`

## 📋 What You Can Do

### 1. **Select Image Generation Model**
   - Stable Diffusion (default, most reliable)
   - QWEN
   - Flux

### 2. **Configure Image Generation**
   - Number of images (1-100)
   - Image size (256, 512, or 768)
   - Batch size (1-8)

### 3. **Choose Action**
   - **Classify Only**: Generate images and classify them
   - **Retrain Model**: Generate images, classify them, AND retrain the model

### 4. **Select Classification Model**
   - VGG-16
   - Inception v3
   - ResNet50 (default)

### 5. **Training Settings** (if retraining)
   - Number of epochs
   - Learning rate
   - Batch size

## 🎯 Example Workflow

1. **Quick Test** (Classify Only):
   - Generation Model: Stable Diffusion
   - Number of Images: 5
   - Image Size: 256
   - Action: Classify Only
   - Click "Start Pipeline"

2. **Full Training** (Retrain Model):
   - Generation Model: Stable Diffusion
   - Number of Images: 20
   - Image Size: 512
   - Action: Retrain Model
   - Classification Model: ResNet50
   - Epochs: 5
   - Click "Start Pipeline"

## 📊 View Results

After running, switch to the **"Results"** tab to see:
- Classification results
- Charts and statistics
- Training metrics (if you retrained)

## 💾 Saved Models

Check the **"Saved Models"** tab to see all your retrained models with their metrics.

---

**That's it!** The interface is intuitive and handles everything for you.

