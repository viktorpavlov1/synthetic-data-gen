# Web Interface Guide

## Quick Start

### 1. Install Streamlit (if not already installed)

```bash
pip install streamlit
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

### 2. Launch the Interface

Simply run:

```bash
python run_interface.py
```

This will:
- Start a Streamlit web server
- Automatically open your browser to `http://localhost:8501`
- Show the interactive interface

## Using the Interface

### Configuration (Sidebar)

**Image Generation:**
- **Generation Model**: Choose between Stable Diffusion, QWEN, or Flux
- **Number of Images**: How many images to generate (1-100)
- **Image Size**: 256, 512, or 768 pixels (larger = better quality but slower)
- **Batch Size**: Number of images to generate at once (1-8)

**Classification & Training:**
- **Action**: 
  - "Classify Only" - Just classify the generated images
  - "Retrain Model" - Classify AND retrain the model with new data
- **Classification Model**: Choose VGG-16, Inception v3, or ResNet50

**Training Settings** (only shown when "Retrain Model" is selected):
- **Number of Epochs**: How many training epochs (1-50)
- **Learning Rate**: Training learning rate (0.0001-0.01)
- **Training Batch Size**: Batch size for training (8-64)

### Running the Pipeline

1. Configure your settings in the sidebar
2. Click the **"▶️ Start Pipeline"** button
3. Watch the progress as it:
   - Generates images
   - Classifies them
   - (Optionally) Trains the model

### Viewing Results

Switch to the **"📊 Results"** tab to see:
- Summary statistics (total images, average confidence, class distribution)
- Detailed results table
- Charts showing class and confidence distributions
- Training metrics (if you retrained)

### Saved Models

The **"💾 Saved Models"** tab shows:
- All previously saved retrained models
- Model metrics and training parameters
- Model file paths

## Tips

1. **Start Small**: For testing, use 5-10 images with size 256
2. **GPU Memory**: If you get out-of-memory errors, reduce batch size or image size
3. **First Run**: Model downloads happen automatically (may take 10-20 minutes)
4. **Training**: Retraining takes longer - start with 5 epochs for testing

## Troubleshooting

### Interface won't start
- Make sure Streamlit is installed: `pip install streamlit`
- Check that you're in the project root directory

### Browser doesn't open automatically
- Manually go to: `http://localhost:8501`

### Out of memory errors
- Reduce image size to 256
- Reduce batch size to 1
- Generate fewer images

### Models won't load
- Check your internet connection (models download from Hugging Face)
- First download can take 10-20 minutes

