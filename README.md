# Synthetic Skin Lesion Generation and Classification Pipeline

A Kedro-based application for generating synthetic skin cancer lesion images using state-of-the-art image generation models (Stable Diffusion, QWEN, Flux) and classifying them using pre-trained models from the ENHANCE repository (VGG-16, Inception v3, ResNet50).

## Features

- **Image Generation**: Generate synthetic skin lesion images using:
  - Stable Diffusion
  - QWEN
  - FLUX
  
- **Classification**: Classify generated images using pre-trained ENHANCE models:
  - VGG-16
  - Inception v3
  - ResNet50

- **Model Training**: Retrain classification models with synthetic data to improve performance

- **Model Storage**: Save and version retrained models with metadata

## Project Structure

```
synthetic-data-gen/
├── conf/
│   ├── base/
│   │   ├── catalog.yml          # Data catalog
│   │   ├── parameters.yml      # Pipeline parameters
│   │   └── logging.yml         # Logging configuration
│   └── local/                  # Local environment configs
├── data/
│   ├── 01_raw/                 # Raw generated images
│   ├── 02_intermediate/        # Processed images
│   ├── 03_primary/             # Training-ready datasets
│   ├── 04_feature/             # Feature-engineered data
│   ├── 05_model_input/         # Model input data
│   ├── 06_models/              # Saved models
│   └── 07_model_output/        # Classification results
├── src/
│   └── synthetic_data_gen/
│       ├── pipelines/          # Kedro pipelines
│       ├── utils/              # Utility functions
│       └── interface/          # CLI interface
├── notebooks/                  # Jupyter notebooks
├── requirements.txt
└── README.md
```

## Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd synthetic-data-gen
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

## Usage

### Command-Line Interface

The application provides a CLI interface for easy interaction.

#### Generate and Classify Images

Generate synthetic images and classify them using a pre-trained model:

```bash
python -m synthetic_data_gen.interface.cli run \
    --generation-model stable_diffusion \
    --num-images 20 \
    --action classify \
    --classification-model resnet50
```

#### Generate and Retrain Model

Generate synthetic images and retrain a classification model:

```bash
python -m synthetic_data_gen.interface.cli run \
    --generation-model flux \
    --num-images 50 \
    --action retrain \
    --classification-model resnet50 \
    --num-epochs 15 \
    --learning-rate 0.001
```

#### List Saved Models

View all saved retrained models:

```bash
python -m synthetic_data_gen.interface.cli list-models
```

### CLI Options

- `--generation-model`: Choose image generation model (`stable_diffusion`, `qwen`, `flux`)
- `--num-images`: Number of images to generate
- `--action`: Action to perform (`classify` or `retrain`)
- `--classification-model`: Classification model to use (`vgg16`, `inception_v3`, `resnet50`)
- `--image-size`: Size of generated images (default: 512)
- `--batch-size`: Batch size for image generation (default: 4)
- `--num-epochs`: Number of training epochs for retraining (default: 10)
- `--learning-rate`: Learning rate for training (default: 0.001)
- `--output-dir`: Output directory for results (default: `data/07_model_output`)

### Using Kedro Directly

You can also use Kedro commands directly:

```bash
# Run data generation pipeline
kedro run --pipeline data_generation

# Run classification pipeline
kedro run --pipeline classification

# Run training pipeline
kedro run --pipeline training

# Run all pipelines
kedro run
```

## Configuration

### Parameters

Edit `conf/base/parameters.yml` to customize default parameters:

- Image generation settings (model, number of images, prompts)
- Classification settings (model, batch size)
- Training settings (learning rate, epochs, data splits)

### Data Catalog

Edit `conf/base/catalog.yml` to configure data storage locations and formats.

## ENHANCE Integration

This project integrates with the [ENHANCE repository](https://github.com/raumannsr/ENHANCE) for classification models. The models (VGG-16, Inception v3, ResNet50) are loaded using PyTorch's torchvision models with ImageNet pretrained weights, adapted for skin lesion classification.

To use custom ENHANCE weights, you can modify the model loading code in `src/synthetic_data_gen/utils/model_loaders.py`.

## Model Storage

Retrained models are automatically saved to `data/06_models/` with:
- Model weights (`.pth` file)
- Metadata JSON file with training parameters and metrics
- Timestamped filenames for versioning

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for image generation and training)

## Notes

- **QWEN Model**: The QWEN implementation may need adjustment based on the specific Qwen model available on Hugging Face. The current implementation includes a fallback to Stable Diffusion if QWEN fails.

- **GPU Memory**: Image generation models require significant GPU memory. Adjust batch sizes if you encounter out-of-memory errors.

- **Model Weights**: The classification models use ImageNet pretrained weights by default. For best results with skin lesions, consider fine-tuning on the ENHANCE dataset first.

## License

[Add your license information here]

## Acknowledgments

- ENHANCE dataset and models: https://github.com/raumannsr/ENHANCE
- Hugging Face for model hosting
- Kedro for pipeline orchestration
