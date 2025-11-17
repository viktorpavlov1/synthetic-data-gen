"""Image-guided generation using HAM10000 reference images."""

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from typing import List, Dict, Any, Optional
import logging
from tqdm import tqdm
import random

logger = logging.getLogger(__name__)


def generate_with_reference_images(
    prompts: List[str],
    reference_images: List[Image.Image],
    num_images: int,
    strength: float = 0.75,
    batch_size: int = 1,
    seed: Optional[int] = None,
    model_id: str = "runwayml/stable-diffusion-v1-5"
) -> List[Dict[str, Any]]:
    """
    Generate images using reference images from HAM10000 dataset.
    
    Uses image-to-image generation where HAM10000 images guide the generation.
    
    Args:
        prompts: List of text prompts
        reference_images: List of PIL Images from HAM10000 to use as references
        num_images: Number of images to generate
        strength: How much to transform the reference image (0.0-1.0)
                  Lower = more similar to reference, Higher = more creative
        batch_size: Batch size for generation
        seed: Random seed
        model_id: Hugging Face model identifier
        
    Returns:
        List of generated images with metadata
    """
    logger.info(f"Loading Stable Diffusion Img2Img pipeline: {model_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    
    if seed is not None:
        torch.manual_seed(seed)
    
    generated_images = []
    
    # Generate images in batches
    num_batches = (num_images + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Generating images with references"):
        batch_prompts = []
        batch_references = []
        
        for i in range(batch_size):
            if len(generated_images) >= num_images:
                break
            
            # Select prompt and reference image
            prompt_idx = len(generated_images) % len(prompts)
            ref_idx = len(generated_images) % len(reference_images)
            
            batch_prompts.append(prompts[prompt_idx])
            
            # Prepare reference image (resize to 600x450 for HAM10000 format)
            ref_img = reference_images[ref_idx].copy()
            ref_img = ref_img.convert('RGB')
            # Resize to generation size (600x448 for SD compatibility)
            ref_img = ref_img.resize((600, 448), Image.Resampling.LANCZOS)
            batch_references.append(ref_img)
        
        if not batch_prompts:
            break
        
        try:
            with torch.no_grad():
                # Generate using image-to-image
                # Note: Img2Img requires same number of prompts and images
                images = []
                for prompt, ref_img in zip(batch_prompts, batch_references):
                    result = pipe(
                        prompt=prompt,
                        image=ref_img,
                        strength=strength,
                        num_inference_steps=50,
                        guidance_scale=7.5
                    ).images[0]
                    
                    # Resize to exact HAM10000 format (600x450)
                    if result.size != (600, 450):
                        result = result.resize((600, 450), Image.Resampling.LANCZOS)
                    
                    images.append(result)
                
                for img, prompt, ref_img in zip(images, batch_prompts, batch_references):
                    generated_images.append({
                        "image": img,
                        "prompt": prompt,
                        "reference_image": ref_img,
                        "model": "stable_diffusion_img2img",
                        "model_id": model_id,
                        "strength": strength,
                        "image_size": (600, 450)
                    })
                
                if len(generated_images) >= num_images:
                    break
                    
        except Exception as e:
            logger.error(f"Error generating batch {batch_idx}: {e}")
            continue
    
    logger.info(f"Generated {len(generated_images)} images using reference images")
    return generated_images


def get_reference_images_from_ham10000(
    ham10000_path: str,
    diagnosis: Optional[str] = None,
    num_references: int = 10
) -> List[Image.Image]:
    """
    Get reference images from HAM10000 dataset.
    
    Args:
        ham10000_path: Path to HAM10000 dataset
        diagnosis: Filter by diagnosis code (optional)
        num_references: Number of reference images to get
        
    Returns:
        List of PIL Images
    """
    from synthetic_data_gen.utils.ham10000_loader import HAM10000Loader
    
    loader = HAM10000Loader(ham10000_path)
    
    # Load images with optional diagnosis filter
    lesion_types = [diagnosis] if diagnosis else None
    images_data = loader.load_images_with_metadata(
        max_images=num_references * 2,  # Load more to have variety
        lesion_types=lesion_types
    )
    
    # Extract PIL Images
    reference_images = []
    for img_data in images_data[:num_references]:
        img = img_data.get('image')
        if img and isinstance(img, Image.Image):
            reference_images.append(img)
    
    logger.info(f"Loaded {len(reference_images)} reference images from HAM10000")
    return reference_images

