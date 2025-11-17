"""Automatic HAM10000 dataset setup and fine-tuning."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def setup_ham10000_fine_tuning(
    ham10000_path: str = "data/00_external/HAM10000",
    output_dir: str = "data/06_models/fine_tuned_sd_ham10000",
    max_images: Optional[int] = None
) -> Dict[str, Any]:
    """
    Set up automatic fine-tuning using HAM10000 dataset.
    
    Args:
        ham10000_path: Path to HAM10000 dataset
        output_dir: Where to save fine-tuned model
        max_images: Maximum images to use (None for all)
        
    Returns:
        Configuration dictionary for fine-tuning
    """
    from synthetic_data_gen.utils.ham10000_loader import HAM10000Loader
    
    ham10000_path = Path(ham10000_path)
    
    if not ham10000_path.exists():
        raise ValueError(f"HAM10000 dataset not found at: {ham10000_path}")
    
    logger.info(f"Setting up HAM10000 fine-tuning from: {ham10000_path}")
    
    # Load dataset
    loader = HAM10000Loader(str(ham10000_path))
    
    # Get statistics
    stats = loader.get_statistics()
    logger.info(f"Dataset statistics: {stats}")
    
    # Load images
    images = loader.load_images_with_metadata(max_images=max_images)
    
    if not images:
        raise ValueError("No images loaded from HAM10000 dataset")
    
    logger.info(f"Loaded {len(images)} images for fine-tuning")
    
    # Create configuration
    config = {
        "dataset_path": str(ham10000_path),
        "output_dir": output_dir,
        "num_images": len(images),
        "statistics": stats,
        "images": images
    }
    
    return config


def auto_fine_tune_ham10000(
    ham10000_path: str = "data/00_external/HAM10000",
    output_dir: str = "data/06_models/fine_tuned_sd_ham10000",
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    max_images: Optional[int] = 2000,  # Use subset for faster training
    lora_rank: int = 4
) -> str:
    """
    Automatically fine-tune Stable Diffusion on HAM10000 dataset.
    
    Args:
        ham10000_path: Path to HAM10000 dataset
        output_dir: Where to save fine-tuned model
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        max_images: Maximum images to use
        lora_rank: LoRA rank
        
    Returns:
        Path to fine-tuned model
    """
    from synthetic_data_gen.pipelines.fine_tuning.nodes import (
        prepare_training_dataset,
        fine_tune_stable_diffusion
    )
    
    # Setup
    config = setup_ham10000_fine_tuning(
        ham10000_path=ham10000_path,
        output_dir=output_dir,
        max_images=max_images
    )
    
    images = config["images"]
    
    # Prepare dataset
    logger.info("Preparing training dataset...")
    prepared_dataset = prepare_training_dataset(
        dataset_images=images,
        output_size=512,
        max_images=max_images
    )
    
    # Fine-tune
    logger.info("Starting fine-tuning...")
    model_path = fine_tune_stable_diffusion(
        dataset=prepared_dataset,
        base_model="runwayml/stable-diffusion-v1-5",
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=1,
        lora_rank=lora_rank,
        lora_alpha=lora_rank * 8
    )
    
    logger.info(f"Fine-tuning complete. Model saved to: {model_path}")
    return model_path

