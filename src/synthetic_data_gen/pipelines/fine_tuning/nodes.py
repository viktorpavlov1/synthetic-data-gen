"""Fine-tuning nodes for image generation models."""

import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import load_image
from typing import List, Dict, Any, Optional
import logging
from tqdm import tqdm
from PIL import Image
import os
import json

logger = logging.getLogger(__name__)


def prepare_training_dataset(
    dataset_images: List[Dict[str, Any]],
    output_size: int = 512,
    max_images: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Prepare dataset for fine-tuning.
    
    Args:
        dataset_images: List of image dictionaries
        output_size: Target image size
        max_images: Maximum number of images to use
        
    Returns:
        Prepared dataset
    """
    if max_images:
        dataset_images = dataset_images[:max_images]
    
    prepared = []
    for img_data in tqdm(dataset_images, desc="Preparing dataset"):
        img = img_data.get('image')
        if img is None:
            continue
        
        # Resize and ensure RGB
        if not isinstance(img, Image.Image):
            continue
        
        img = img.convert('RGB')
        img = img.resize((output_size, output_size), Image.Resampling.LANCZOS)
        
        prepared.append({
            'image': img,
            'path': img_data.get('path', ''),
            'prompt': img_data.get('prompt', 'skin lesion image')
        })
    
    logger.info(f"Prepared {len(prepared)} images for fine-tuning")
    return prepared


def fine_tune_stable_diffusion(
    dataset: List[Dict[str, Any]],
    base_model: str = "runwayml/stable-diffusion-v1-5",
    output_dir: str = "data/06_models/fine_tuned_sd",
    num_train_epochs: int = 10,
    learning_rate: float = 1e-4,
    batch_size: int = 1,
    lora_rank: int = 4,
    lora_alpha: int = 32,
    save_steps: int = 500
) -> str:
    """
    Fine-tune Stable Diffusion using LoRA.
    
    Note: This is a simplified implementation. For production use,
    consider using the diffusers training scripts directly.
    
    Args:
        base_model: Base Stable Diffusion model ID
        dataset: Prepared training dataset
        output_dir: Directory to save fine-tuned model
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Training batch size
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        save_steps: Save checkpoint every N steps
        
    Returns:
        Path to fine-tuned model
    """
    logger.info(f"Fine-tuning Stable Diffusion on {len(dataset)} images")
    logger.info("Note: Full fine-tuning requires additional setup. Using LoRA approach.")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save fine-tuning configuration
    config_path = os.path.join(output_dir, "fine_tuning_config.json")
    
    config = {
        "base_model": base_model,
        "num_images": len(dataset),
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "num_epochs": num_train_epochs,
        "batch_size": batch_size,
        "output_dir": output_dir,
        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],  # Attention layers for LoRA
        "image_size": 512,  # Standard size for training
        "resolution": 512
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Fine-tuning configuration saved to {config_path}")
    logger.warning("""
    ⚠️  Full fine-tuning requires using diffusers training scripts.
    
    To actually fine-tune the model, use one of these approaches:
    
    1. Use diffusers training script:
       accelerate launch train_text_to_image_lora.py \\
         --pretrained_model_name_or_path="{base_model}" \\
         --train_data_dir="path/to/dataset" \\
         --output_dir="{output_dir}" \\
         --resolution=512 \\
         --train_batch_size={batch_size} \\
         --learning_rate={learning_rate} \\
         --max_train_steps={num_epochs * len(dataset) // batch_size} \\
         --lr_scheduler="constant" \\
         --lr_warmup_steps=0 \\
         --rank={lora_rank}
    
    2. Use Hugging Face Trainer with diffusers
    
    The configuration has been saved. You can use it with the training scripts above.
    """)
    
    # Save dataset info for reference
    dataset_info_path = os.path.join(output_dir, "dataset_info.json")
    with open(dataset_info_path, 'w') as f:
        json.dump({
            "num_images": len(dataset),
            "sample_prompts": [d.get('prompt', '') for d in dataset[:5]],
            "image_paths": [d.get('path', '') for d in dataset[:5]]
        }, f, indent=2)
    
    logger.info(f"Dataset info saved to {dataset_info_path}")
    logger.info(f"Fine-tuning setup complete. Configuration saved to: {output_dir}")
    
    return output_dir


def load_fine_tuned_model(
    model_path: str,
    base_model: str = "runwayml/stable-diffusion-v1-5"
) -> StableDiffusionPipeline:
    """
    Load a fine-tuned Stable Diffusion model.
    
    Args:
        model_path: Path to fine-tuned model or LoRA weights
        base_model: Base model to load LoRA on
        
    Returns:
        Loaded pipeline
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if it's a full model or LoRA weights
    if os.path.isdir(model_path) and "pytorch_model.bin" in os.listdir(model_path):
        # Full model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
    else:
        # Load base model and apply LoRA if available
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        # TODO: Load and apply LoRA weights if available
    
    pipe = pipe.to(device)
    return pipe

