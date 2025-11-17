# HAM10000 Dataset Integration Guide

## Overview

The system now automatically integrates with the HAM10000 dataset to improve image generation quality through:
1. **Metadata-based prompt generation** - Uses actual patient data (age, sex, localization, diagnosis) to create realistic prompts
2. **Automatic fine-tuning** - One-click setup to fine-tune models on HAM10000 images
3. **Distribution matching** - Generates prompts that match the actual distribution of lesion types in the dataset

## Dataset Structure

Your HAM10000 dataset should be located at:
```
data/00_external/HAM10000/
├── HAM10000_metadata          # Metadata file (required)
├── images/                    # Image directory
│   ├── ISIC_0024306.jpg
│   ├── ISIC_0024307.jpg
│   └── ...
└── ...
```

## Dataset Statistics

The HAM10000 dataset contains:
- **10,015 total images**
- **7 lesion types:**
  - `nv` (melanocytic nevus): 6,705 images
  - `mel` (melanoma): 1,113 images
  - `bkl` (benign keratosis): 1,099 images
  - `bcc` (basal cell carcinoma): 514 images
  - `akiec` (actinic keratosis): 327 images
  - `vasc` (vascular lesion): 142 images
  - `df` (dermatofibroma): 115 images

- **Age range:** 0-85 years
- **Sex distribution:** 5,406 male, 4,552 female, 57 unknown
- **Localizations:** back, lower extremity, trunk, upper extremity, abdomen, face, chest, foot, neck, scalp, ear

## Automatic Features

### 1. Metadata-Based Prompt Generation

When HAM10000 is detected, the system automatically:
- Uses actual diagnosis codes from the dataset
- Includes localization information (e.g., "on the back", "on the face")
- Adds demographic context (age group, sex) when available
- Matches the distribution of lesion types in the dataset

**Example prompts generated:**
- "A high-quality dermoscopic image of a melanoma on the back in a middle-aged male, medical photography, detailed, professional, clinical quality, dermoscopy"
- "A high-quality dermoscopic image of a melanocytic nevus on the trunk in a young female, medical photography, detailed, professional, clinical quality, dermoscopy"

### 2. Automatic Detection

The system automatically detects HAM10000 when:
- The dataset exists at `data/00_external/HAM10000`
- The metadata file `HAM10000_metadata` is present

No manual configuration needed!

### 3. Fine-Tuning Integration

The web interface provides:
- **Quick setup button** for HAM10000 fine-tuning
- **Dataset statistics** visualization
- **One-click fine-tuning** with recommended settings

## Usage

### Step 1: Verify Dataset

Run the setup script to verify your dataset:
```bash
python setup_ham10000.py
```

This will:
- Check if the dataset exists
- Load and display statistics
- Verify metadata is accessible

### Step 2: Use in Web Interface

1. **Launch the interface:**
   ```bash
   python run_interface.py
   ```

2. **Automatic prompt generation:**
   - Go to "Run Pipeline" tab
   - The system automatically detects HAM10000
   - Prompts are generated using HAM10000 metadata
   - You'll see: "✅ Using HAM10000 metadata for X prompts"

3. **Fine-tuning (optional):**
   - Go to "Fine-Tune Generator" tab
   - You'll see HAM10000 dataset statistics
   - Click "🎯 Auto-Setup HAM10000 Fine-Tuning"
   - Configure epochs and max images
   - Start fine-tuning

### Step 3: Generate Images

When generating images:
- Prompts automatically use HAM10000 metadata
- Distribution matches the actual dataset
- More realistic and diverse prompts
- Better quality synthetic images

## Configuration

### Parameters (conf/base/parameters.yml)

```yaml
ham10000:
  dataset_path: "data/00_external/HAM10000"
  use_ham10000_prompts: true  # Enable metadata-based prompts
  auto_fine_tune: false  # Auto fine-tune on startup
  fine_tuned_model_path: "data/06_models/fine_tuned_sd_ham10000"
```

### Using HAM10000 Prompts Programmatically

```python
from synthetic_data_gen.pipelines.data_generation.ham10000_prompts import generate_ham10000_prompts

prompts = generate_ham10000_prompts(
    dataset_path="data/00_external/HAM10000",
    num_prompts=10,
    lesion_types=['mel', 'nv'],  # Optional filter
    match_distribution=True
)
```

### Loading HAM10000 Images

```python
from synthetic_data_gen.utils.ham10000_loader import HAM10000Loader

loader = HAM10000Loader("data/00_external/HAM10000")
images = loader.load_images_with_metadata(
    max_images=1000,
    lesion_types=['mel', 'nv'],
    localization='back'
)

# Each image has rich metadata:
for img_data in images:
    print(f"Diagnosis: {img_data['diagnosis_full']}")
    print(f"Localization: {img_data['localization']}")
    print(f"Age: {img_data['age']}")
    print(f"Sex: {img_data['sex']}")
```

## Benefits

1. **More Realistic Prompts:**
   - Based on actual medical data
   - Includes relevant clinical context
   - Matches real-world distribution

2. **Better Image Quality:**
   - Prompts are more specific and detailed
   - Context helps generation models
   - Results are more clinically relevant

3. **Easy Fine-Tuning:**
   - One-click setup
   - Automatic dataset preparation
   - Optimized for HAM10000 structure

4. **Distribution Matching:**
   - Generated images match real distribution
   - Better for training classifiers
   - More balanced datasets

## Troubleshooting

### Dataset Not Detected

- Check that `data/00_external/HAM10000` exists
- Verify `HAM10000_metadata` file is present
- Run `python setup_ham10000.py` to diagnose

### Metadata Loading Errors

- Ensure metadata file is CSV format
- Check file encoding (should be UTF-8)
- Verify column names match expected format

### Image Loading Issues

- Check that `images/` directory exists
- Verify image files are `.jpg` format
- Ensure image IDs match metadata

## Next Steps

1. **Generate images** using HAM10000 prompts
2. **Fine-tune models** on HAM10000 images
3. **Compare results** before/after HAM10000 integration
4. **Train classifiers** with improved synthetic data

## Technical Details

### Prompt Generation Process

1. Load HAM10000 metadata
2. Filter by lesion types (if specified)
3. Sample according to distribution
4. Create prompts with:
   - Diagnosis (full name)
   - Localization
   - Age group
   - Sex
   - Clinical descriptors

### Fine-Tuning Process

1. Load HAM10000 images with metadata
2. Prepare images (resize, normalize)
3. Create prompts for each image
4. Fine-tune Stable Diffusion using LoRA
5. Save fine-tuned model

The fine-tuned model learns:
- HAM10000-specific features
- Medical imaging characteristics
- Better lesion representation

