"""Image generation nodes for synthetic skin lesion images."""

import torch
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionImg2ImgPipeline, 
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusion3Pipeline,
    FluxPipeline,
    DiffusionPipeline
)
try:
    from diffusers import StableDiffusion3Img2ImgPipeline
except ImportError:
    StableDiffusion3Img2ImgPipeline = None  # May not be available
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from tqdm import tqdm
import random

logger = logging.getLogger(__name__)


def generate_stable_diffusion_images(
    prompts: List[str],
    num_images: int,
    image_size: int = 512,
    batch_size: int = 4,
    seed: Optional[int] = None,
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    reference_images: Optional[List[Image.Image]] = None,
    img2img_strength: float = 0.75
) -> List[Dict[str, Any]]:
    """
    Generate images using Stable Diffusion.
    
    Args:
        prompts: List of text prompts for image generation
        num_images: Total number of images to generate
        image_size: Size of generated images
        batch_size: Number of images to generate per batch
        seed: Random seed for reproducibility
        model_id: Hugging Face model identifier
        reference_images: Optional list of reference images for image-to-image generation
        img2img_strength: Strength for image-to-image (0.0-1.0), only used if reference_images provided
        
    Returns:
        List of dictionaries containing images and metadata
    """
    # Use image-to-image if reference images are provided
    if reference_images and len(reference_images) > 0:
        # Check if reference_images is a tuple with reference pool
        if isinstance(reference_images, tuple) and len(reference_images) == 3:
            # New format: (prompts, paired_references, reference_pool)
            prompts_list, paired_refs, reference_pool = reference_images
            logger.info(f"Using image-to-image generation with {len(reference_pool)} reference images in pool")
            return generate_stable_diffusion_img2img(
                prompts=prompts_list,
                reference_images=paired_refs,
                num_images=num_images,
                batch_size=batch_size,
                seed=seed,
                model_id=model_id,
                strength=img2img_strength,
                reference_pool=reference_pool
            )
        else:
            # Old format: just list of reference images
            logger.info(f"Using image-to-image generation with {len(reference_images)} reference images")
            return generate_stable_diffusion_img2img(
                prompts=prompts,
                reference_images=reference_images,
                num_images=num_images,
                batch_size=batch_size,
                seed=seed,
                model_id=model_id,
                strength=img2img_strength
            )
    
    # Standard text-to-image generation
    logger.info(f"Loading Stable Diffusion model: {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Clear CUDA cache before loading
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Use CPU offloading if available to save VRAM
    if device == "cuda":
        try:
            pipe.enable_model_cpu_offload()  # Offload to CPU to save VRAM
        except Exception as e:
            logger.warning(f"Could not enable CPU offloading: {e}. Loading to GPU directly.")
            pipe = pipe.to(device)
        
        # Enable memory optimizations
        try:
            pipe.enable_attention_slicing(slice_size="auto")
            logger.info("Enabled attention slicing for memory optimization")
        except Exception as e:
            logger.warning(f"Could not enable attention slicing: {e}")
        
        try:
            pipe.enable_vae_tiling()
            logger.info("Enabled VAE tiling for memory optimization")
        except Exception as e:
            logger.warning(f"Could not enable VAE tiling: {e}")
    else:
        pipe = pipe.to(device)
    
    if seed is not None:
        torch.manual_seed(seed)
    
    generated_images = []
    
    # Validate prompts
    if not prompts:
        logger.error("No prompts provided to generate_stable_diffusion_images")
        return []
    
    # Generate images in batches
    num_batches = (num_images + batch_size - 1) // batch_size
    logger.info(f"Generating {num_images} images in {num_batches} batches of size {batch_size}")
    
    for batch_idx in tqdm(range(num_batches), desc="Generating images"):
        batch_prompts = []
        for i in range(batch_size):
            if len(generated_images) >= num_images:
                break
            prompt_idx = len(generated_images) % len(prompts)
            batch_prompts.append(prompts[prompt_idx])
        
        if not batch_prompts:
            logger.warning(f"Batch {batch_idx} has no prompts, stopping")
            break
        
        try:
            with torch.no_grad():
                # HAM10000 format: 600x450 (4:3 aspect ratio)
                # Stable Diffusion requires dimensions divisible by 8
                # Using 600x448 (closest to 600x450 that meets the requirement)
                # 600/8=75, 448/8=56
                height = 448  # Adjusted from 450 to be divisible by 8
                width = 600   # Already divisible by 8
                
                # Strong negative prompt to exclude body parts
                negative_prompt = "face, facial features, eyes, nose, mouth, lips, teeth, tongue, head, person, body, torso, chest, back, limbs, arms, legs, hands, feet, fingers, toes, full body, portrait, human figure, human face, facial expression, smile, frown, clothing, clothes, background, landscape, artistic, painting, drawing, illustration, cartoon, anime, 3d render, blurry, low quality, distorted, deformed, anatomy, human anatomy, body parts, facial anatomy, oral cavity, dental, teeth close-up, lipstick, makeup, beauty, cosmetic"
                
                images = pipe(
                    batch_prompts,
                    negative_prompt=[negative_prompt] * len(batch_prompts),
                    height=height,
                    width=width,
                    num_inference_steps=50,
                    guidance_scale=9.0  # Higher guidance scale for better prompt adherence
                ).images
            
            for img, prompt in zip(images, batch_prompts):
                # Generate at 600x448 (divisible by 8), then resize to 600x450 (HAM10000 format)
                if img.size != (600, 450):
                    img = img.resize((600, 450), Image.Resampling.LANCZOS)
                
                generated_images.append({
                    "image": img,
                    "prompt": prompt,
                    "model": "stable_diffusion",
                    "model_id": model_id,
                    "image_size": (600, 450)  # Final size matches HAM10000
                })
            
            if len(generated_images) >= num_images:
                break
                    
        except Exception as e:
            logger.error(f"Error generating batch {batch_idx}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Don't continue silently - raise if it's a critical error
            # But allow a few failures before giving up
            if batch_idx == 0:  # If first batch fails, it's likely a critical error
                raise
            continue
    
    logger.info(f"Generated {len(generated_images)} images using Stable Diffusion")
    return generated_images


def generate_stable_diffusion_img2img(
    prompts: List[str],
    reference_images: List[Image.Image],
    num_images: int,
    batch_size: int = 1,
    seed: Optional[int] = None,
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    strength: float = 0.75,
    reference_pool: Optional[List[Image.Image]] = None
) -> List[Dict[str, Any]]:
    """
    Generate images using image-to-image with HAM10000 reference images.
    
    Args:
        prompts: List of text prompts
        reference_images: List of reference images from HAM10000 (for backward compatibility)
        num_images: Number of images to generate
        batch_size: Batch size (typically 1 for img2img)
        seed: Random seed
        model_id: Model identifier
        strength: How much to transform reference (0.0-1.0)
        reference_pool: Optional larger pool of reference images to randomly select from
        
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
    
    # Enable memory optimizations
    if device == "cuda":
        try:
            pipe.enable_attention_slicing(slice_size="auto")
            logger.info("Enabled attention slicing for memory optimization")
        except Exception as e:
            logger.warning(f"Could not enable attention slicing: {e}")
        
        try:
            pipe.enable_vae_tiling()
            logger.info("Enabled VAE tiling for memory optimization")
        except Exception as e:
            logger.warning(f"Could not enable VAE tiling: {e}")
    
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    
    # Use reference pool if provided, otherwise use reference_images
    if reference_pool and len(reference_pool) > 0:
        available_references = reference_pool
        logger.info(f"Using reference pool with {len(available_references)} images")
    else:
        available_references = reference_images
        logger.info(f"Using {len(available_references)} reference images")
    
    generated_images = []
    
    for i in tqdm(range(num_images), desc="Generating images with references"):
        # Select prompt
        prompt = prompts[i % len(prompts)]
        
        # Randomly select a reference image from the pool
        ref_img = random.choice(available_references).copy()
        
        # Prepare reference image
        ref_img = ref_img.convert('RGB')
        # Resize to generation size (600x448 for SD compatibility)
        ref_img = ref_img.resize((600, 448), Image.Resampling.LANCZOS)
        
        try:
            with torch.no_grad():
                # Strong negative prompt to exclude body parts and ensure close-up view
                negative_prompt = "face, facial features, eyes, nose, mouth, lips, teeth, tongue, head, person, body, torso, chest, back, limbs, arms, legs, hands, feet, fingers, toes, full body, portrait, human figure, human face, facial expression, smile, frown, clothing, clothes, background, landscape, artistic, painting, drawing, illustration, cartoon, anime, 3d render, blurry, low quality, distorted, deformed, wide shot, full view, body context, anatomy, human anatomy, body parts, facial anatomy, oral cavity, dental, teeth close-up, lipstick, makeup, beauty, cosmetic"
                
                result = pipe(
                    prompt=prompt,
                    image=ref_img,
                    negative_prompt=negative_prompt,
                    strength=strength,
                    num_inference_steps=50,
                    guidance_scale=9.0  # Higher guidance scale for better prompt adherence
                ).images[0]
                
                # Resize to exact HAM10000 format (600x450)
                if result.size != (600, 450):
                    result = result.resize((600, 450), Image.Resampling.LANCZOS)
                
                generated_images.append({
                    "image": result,
                    "prompt": prompt,
                    "reference_image": ref_img,
                    "model": "stable_diffusion_img2img",
                    "model_id": model_id,
                    "strength": strength,
                    "image_size": (600, 450)
                })
        except Exception as e:
            logger.error(f"Error generating image {i}: {e}")
            continue
    
    logger.info(f"Generated {len(generated_images)} images using {len(available_references)} reference images from pool")
    return generated_images


def generate_sdxl_images(
    prompts: List[str],
    num_images: int,
    image_size: int = 512,
    batch_size: int = 4,
    seed: Optional[int] = None,
    model_id: str = "stabilityai/stable-diffusion-3.5-large",
    reference_images: Optional[List[Image.Image]] = None,
    img2img_strength: float = 0.75
) -> List[Dict[str, Any]]:
    """
    Generate images using Stable Diffusion 3.5 Large (replaces SDXL).
    
    Args:
        prompts: List of text prompts for image generation
        num_images: Total number of images to generate
        image_size: Size of generated images
        batch_size: Number of images to generate per batch
        seed: Random seed for reproducibility
        model_id: Hugging Face model identifier
        reference_images: Optional list of reference images for image-to-image generation
        img2img_strength: Strength for image-to-image (0.0-1.0), only used if reference_images provided
        
    Returns:
        List of dictionaries containing images and metadata
    """
    # Use image-to-image if reference images are provided
    if reference_images and len(reference_images) > 0:
        # Check if reference_images is a tuple with reference pool
        if isinstance(reference_images, tuple) and len(reference_images) == 3:
            # New format: (prompts, paired_references, reference_pool)
            prompts_list, paired_refs, reference_pool = reference_images
            logger.info(f"Using SDXL image-to-image generation with {len(reference_pool)} reference images in pool")
            return generate_sdxl_img2img(
                prompts=prompts_list,
                reference_images=paired_refs,
                num_images=num_images,
                batch_size=batch_size,
                seed=seed,
                model_id=model_id,
                strength=img2img_strength,
                reference_pool=reference_pool
            )
        else:
            # Old format: just list of reference images
            logger.info(f"Using SDXL image-to-image generation with {len(reference_images)} reference images")
            return generate_sdxl_img2img(
                prompts=prompts,
                reference_images=reference_images,
                num_images=num_images,
                batch_size=batch_size,
                seed=seed,
                model_id=model_id,
                strength=img2img_strength
            )
    
    # Standard text-to-image generation
    logger.info(f"Loading Stable Diffusion 3.5 Large model: {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Clear CUDA cache before loading
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    
    # Use CPU offloading if available to save VRAM
    if device == "cuda":
        try:
            pipe.enable_model_cpu_offload()  # Offload to CPU to save VRAM
        except Exception as e:
            logger.warning(f"Could not enable CPU offloading: {e}. Loading to GPU directly.")
            pipe = pipe.to(device)
        
        # Enable memory optimizations
        try:
            pipe.enable_attention_slicing(slice_size="auto")
            logger.info("Enabled attention slicing for memory optimization")
        except Exception as e:
            logger.warning(f"Could not enable attention slicing: {e}")
        
        try:
            pipe.enable_vae_tiling()
            logger.info("Enabled VAE tiling for memory optimization")
        except Exception as e:
            logger.warning(f"Could not enable VAE tiling: {e}")
    else:
        pipe = pipe.to(device)
    
    if seed is not None:
        torch.manual_seed(seed)
    
    generated_images = []
    
    # Validate prompts
    if not prompts:
        logger.error("No prompts provided to generate_sdxl_images")
        return []
    
    # Generate images in batches
    num_batches = (num_images + batch_size - 1) // batch_size
    logger.info(f"Generating {num_images} images in {num_batches} batches of size {batch_size}")
    
    for batch_idx in tqdm(range(num_batches), desc="Generating images"):
        batch_prompts = []
        for i in range(batch_size):
            if len(generated_images) >= num_images:
                break
            prompt_idx = len(generated_images) % len(prompts)
            batch_prompts.append(prompts[prompt_idx])
        
        if not batch_prompts:
            logger.warning(f"Batch {batch_idx} has no prompts, stopping")
            break
        
        try:
            with torch.no_grad():
                # HAM10000 format: 600x450 (4:3 aspect ratio)
                # SD3.5 requires dimensions divisible by 16
                # Using 592x448 (closest to 600x450 that meets the requirement)
                height = 448  # Divisible by 16 (448/16=28)
                width = 592   # Divisible by 16 (592/16=37)
                
                # Strong negative prompt to exclude body parts
                negative_prompt = "face, facial features, eyes, nose, mouth, lips, teeth, tongue, head, person, body, torso, chest, back, limbs, arms, legs, hands, feet, fingers, toes, full body, portrait, human figure, human face, facial expression, smile, frown, clothing, clothes, background, landscape, artistic, painting, drawing, illustration, cartoon, anime, 3d render, blurry, low quality, distorted, deformed, anatomy, human anatomy, body parts, facial anatomy, oral cavity, dental, teeth close-up, lipstick, makeup, beauty, cosmetic"
                
                # SD3.5 supports single prompt or list of prompts
                if len(batch_prompts) == 1:
                    result = pipe(
                        prompt=batch_prompts[0],
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width,
                        num_inference_steps=28,
                        guidance_scale=3.5
                    )
                    images = [result.images[0]]
                else:
                    # Generate one at a time for batch
                    images = []
                    for prompt in batch_prompts:
                        result = pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            height=height,
                            width=width,
                            num_inference_steps=28,
                            guidance_scale=3.5
                        )
                        images.append(result.images[0])
            
            for img, prompt in zip(images, batch_prompts):
                # Generate at 600x448 (divisible by 8), then resize to 600x450 (HAM10000 format)
                if img.size != (600, 450):
                    img = img.resize((600, 450), Image.Resampling.LANCZOS)
                
                generated_images.append({
                    "image": img,
                    "prompt": prompt,
                    "model": "stable_diffusion_xl",
                    "model_id": model_id,
                    "image_size": (600, 450)  # Final size matches HAM10000
                })
            
            if len(generated_images) >= num_images:
                break
                    
        except Exception as e:
            logger.error(f"Error generating batch {batch_idx}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            if batch_idx == 0:
                raise
            continue
    
    logger.info(f"Generated {len(generated_images)} images using Stable Diffusion XL")
    return generated_images


def generate_sdxl_img2img(
    prompts: List[str],
    reference_images: List[Image.Image],
    num_images: int,
    batch_size: int = 1,
    seed: Optional[int] = None,
    model_id: str = "stabilityai/stable-diffusion-3.5-large",
    strength: float = 0.75,
    reference_pool: Optional[List[Image.Image]] = None
) -> List[Dict[str, Any]]:
    """
    Generate images using Stable Diffusion 3.5 Large image-to-image with HAM10000 reference images.
    
    Note: SD3.5 may not have a dedicated img2img pipeline. If not available, falls back to text-to-image.
    
    Args:
        prompts: List of text prompts
        reference_images: List of reference images from HAM10000 (for backward compatibility)
        num_images: Number of images to generate
        batch_size: Batch size (typically 1 for img2img)
        seed: Random seed
        model_id: Model identifier
        strength: How much to transform reference (0.0-1.0)
        reference_pool: Optional larger pool of reference images to randomly select from
        
    Returns:
        List of generated images with metadata
    """
    logger.info(f"Loading Stable Diffusion 3.5 Large Img2Img pipeline: {model_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Note: SD3.5 may not have a dedicated img2img pipeline
    # We'll try to use the base pipeline with image conditioning if available
    # Otherwise, fall back to text-to-image
    use_img2img = False
    if StableDiffusion3Img2ImgPipeline is not None:
        try:
            # Try to load SD3.5 img2img if available (may not exist)
            pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
            )
            pipe = pipe.to(device)
            use_img2img = True
        except Exception as e:
            logger.warning(f"SD3.5 img2img pipeline not available: {e}. Using text-to-image with reference guidance.")
            # Fall back to text-to-image (img2img not available for SD3.5)
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
            )
            pipe = pipe.to(device)
    else:
        logger.warning("SD3.5 img2img pipeline not available in this diffusers version. Using text-to-image.")
        # Fall back to text-to-image (img2img not available for SD3.5)
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        pipe = pipe.to(device)
    
    # Enable memory optimizations
    if device == "cuda":
        try:
            pipe.enable_attention_slicing(slice_size="auto")
            logger.info("Enabled attention slicing for memory optimization")
        except Exception as e:
            logger.warning(f"Could not enable attention slicing: {e}")
        
        try:
            pipe.enable_vae_tiling()
            logger.info("Enabled VAE tiling for memory optimization")
        except Exception as e:
            logger.warning(f"Could not enable VAE tiling: {e}")
    
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    
    # Use reference pool if provided, otherwise use reference_images
    if reference_pool and len(reference_pool) > 0:
        available_references = reference_pool
        logger.info(f"Using reference pool with {len(available_references)} images")
    else:
        available_references = reference_images
        logger.info(f"Using {len(available_references)} reference images")
    
    generated_images = []
    
    for i in tqdm(range(num_images), desc="Generating images with references (SDXL)"):
        # Select prompt
        prompt = prompts[i % len(prompts)]
        
        # Randomly select a reference image from the pool
        ref_img = random.choice(available_references).copy()
        
        # Prepare reference image
        ref_img = ref_img.convert('RGB')
        # Resize to generation size (592x448 for SD3.5 compatibility - divisible by 16)
        ref_img = ref_img.resize((592, 448), Image.Resampling.LANCZOS)
        
        try:
            with torch.no_grad():
                # Strong negative prompt to exclude body parts and ensure close-up view
                negative_prompt = "face, facial features, eyes, nose, mouth, lips, teeth, tongue, head, person, body, torso, chest, back, limbs, arms, legs, hands, feet, fingers, toes, full body, portrait, human figure, human face, facial expression, smile, frown, clothing, clothes, background, landscape, artistic, painting, drawing, illustration, cartoon, anime, 3d render, blurry, low quality, distorted, deformed, wide shot, full view, body context, anatomy, human anatomy, body parts, facial anatomy, oral cavity, dental, teeth close-up, lipstick, makeup, beauty, cosmetic"
                
                if use_img2img:
                    result = pipe(
                        prompt=prompt,
                        image=ref_img,
                        negative_prompt=negative_prompt,
                        strength=strength,
                        num_inference_steps=28,
                        guidance_scale=3.5
                    ).images[0]
                else:
                    # Fallback: use text-to-image (reference image not used)
                    logger.warning("SD3.5 img2img not available, using text-to-image only")
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=448,
                        width=592,
                        num_inference_steps=28,
                        guidance_scale=3.5
                    ).images[0]
                
                # Resize to exact HAM10000 format (600x450)
                if result.size != (600, 450):
                    result = result.resize((600, 450), Image.Resampling.LANCZOS)
                
                generated_images.append({
                    "image": result,
                    "prompt": prompt,
                    "reference_image": ref_img if use_img2img else None,
                    "model": "stable_diffusion_3.5_img2img" if use_img2img else "stable_diffusion_3.5",
                    "model_id": model_id,
                    "strength": strength if use_img2img else None,
                    "image_size": (600, 450)
                })
        except Exception as e:
            logger.error(f"Error generating image {i}: {e}")
            continue
    
    logger.info(f"Generated {len(generated_images)} images using {len(available_references)} reference images from pool (SD3.5)")
    return generated_images


def generate_qwen_images(
    prompts: List[str],
    num_images: int,
    image_size: int = 512,
    batch_size: int = 1,
    seed: Optional[int] = None,
    model_id: str = "Qwen/Qwen-Image"
) -> List[Dict[str, Any]]:
    """
    Generate images using Qwen-Image model.
    
    Args:
        prompts: List of text prompts for image generation
        num_images: Total number of images to generate
        image_size: Size of generated images
        batch_size: Number of images to generate per batch
        seed: Random seed for reproducibility
        model_id: Hugging Face model identifier
        
    Returns:
        List of dictionaries containing images and metadata
    """
    logger.info(f"Loading Qwen-Image model: {model_id}")
    
    # Import torch locally to avoid scoping issues when we modify torch.compiler
    import torch as torch_module
    
    device = "cuda" if torch_module.cuda.is_available() else "cpu"
    
    # Clear CUDA cache before loading
    if device == "cuda":
        torch_module.cuda.empty_cache()
        torch_module.cuda.synchronize()
    
    # Handle torch.compiler compatibility issues with older PyTorch versions
    import os
    original_torch_compile = os.environ.get("TORCH_COMPILE_DISABLE", None)
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    
    # Patch torch.compiler if it doesn't have is_compiling attribute
    # Use local torch_module reference to avoid Python scoping issues
    try:
        # Check if torch.compiler exists
        if hasattr(torch_module, 'compiler'):
            if not hasattr(torch_module.compiler, 'is_compiling'):
                # Add a dummy is_compiling function for compatibility
                def is_compiling():
                    return False
                setattr(torch_module.compiler, 'is_compiling', is_compiling)
                logger.info("Patched torch.compiler.is_compiling for compatibility")
        else:
            # Create a dummy compiler module if it doesn't exist
            import types
            compiler_module = types.ModuleType('compiler')
            def is_compiling():
                return False
            compiler_module.is_compiling = is_compiling
            setattr(torch_module, 'compiler', compiler_module)
            logger.info("Created torch.compiler module for compatibility")
    except Exception as patch_error:
        logger.warning(f"Could not patch torch.compiler: {patch_error}")
    
    # Patch scaled_dot_product_attention to filter out unsupported parameters
    # This handles PyTorch version compatibility where enable_gqa may not be supported
    try:
        original_sdpa = torch_module.nn.functional.scaled_dot_product_attention
        
        def patched_sdpa(*args, **kwargs):
            # Always remove enable_gqa if present - it's not supported in older PyTorch versions
            # We'll catch and retry if needed, but proactively removing it is safer
            had_enable_gqa = 'enable_gqa' in kwargs
            if had_enable_gqa:
                logger.debug("Removing 'enable_gqa' parameter from scaled_dot_product_attention for compatibility")
                kwargs = {k: v for k, v in kwargs.items() if k != 'enable_gqa'}
            
            try:
                return original_sdpa(*args, **kwargs)
            except TypeError as e:
                # If we still get a TypeError about unexpected keyword, it might be a different param
                error_str = str(e).lower()
                if 'unexpected keyword' in error_str:
                    # Try to identify which parameter is problematic
                    logger.debug(f"TypeError in scaled_dot_product_attention: {e}")
                    # For now, just re-raise - the error handling in the generation loop will catch it
                    raise
                raise
        
        torch_module.nn.functional.scaled_dot_product_attention = patched_sdpa
        logger.info("Patched scaled_dot_product_attention to handle enable_gqa compatibility")
    except Exception as patch_error:
        logger.warning(f"Could not patch scaled_dot_product_attention: {patch_error}")
    
    try:
        # Load Qwen-Image pipeline
        logger.info("Loading Qwen-Image pipeline...")
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_module.bfloat16 if device == "cuda" else torch_module.float32
        )
        logger.info("Pipeline loaded, setting up device...")
        
        # QWEN is a very large model (~20GB+), so use CPU offloading from the start
        # This keeps model components on CPU and only moves them to GPU when needed
        if device == "cuda":
            logger.info("Using CPU offloading for QWEN (large model, prevents OOM)")
            try:
                pipe.enable_model_cpu_offload()
                logger.info("Enabled CPU offloading successfully")
            except Exception as e:
                logger.warning(f"Could not enable CPU offloading: {e}. Loading to GPU (may OOM)...")
                pipe = pipe.to(device)
            
            # Enable memory optimizations
            try:
                pipe.enable_attention_slicing(slice_size="auto")
                logger.info("Enabled attention slicing for memory optimization")
            except Exception as e:
                logger.warning(f"Could not enable attention slicing: {e}")
            
            try:
                pipe.enable_vae_tiling()
                logger.info("Enabled VAE tiling for memory optimization")
            except Exception as e:
                logger.warning(f"Could not enable VAE tiling: {e}")
        else:
            pipe = pipe.to(device)
            logger.info("Model loaded to CPU")
        
        if seed is not None:
            torch_module.manual_seed(seed)
        
        generated_images = []
        
        # Qwen-Image requires dimensions divisible by 16
        # For 600x450 (4:3 aspect ratio), we'll use dimensions divisible by 16
        # Target: 600x450, closest divisible by 16: 608x448 (maintains ~4:3 ratio)
        # Or use Qwen-Image's supported aspect ratios (all divisible by 16)
        aspect_ratios = {
            "1:1": (1344, 1344),      # Divisible by 16
            "16:9": (1664, 928),       # Divisible by 16
            "9:16": (928, 1664),       # Divisible by 16
            "4:3": (1472, 1104),       # Divisible by 16 (was 1140, now 1104)
            "3:4": (1104, 1472),       # Divisible by 16
            "3:2": (1584, 1056),       # Divisible by 16
            "2:3": (1056, 1584),       # Divisible by 16
        }
        # Use 4:3 aspect ratio (closest to 600x450)
        # 1472x1104 is close to 4:3 and divisible by 16
        width, height = aspect_ratios["4:3"]
        
        for i in tqdm(range(num_images), desc="Generating images with Qwen-Image"):
            prompt = prompts[i % len(prompts)]
            logger.info(f"Starting generation for image {i+1}/{num_images}")
            logger.debug(f"Prompt: {prompt[:100]}...")
            
            try:
                with torch_module.no_grad():
                    # Qwen-Image uses true_cfg_scale instead of guidance_scale
                    # Handle various PyTorch compatibility issues
                    # Reduce steps for faster generation (can be increased later if needed)
                    inference_steps = 30  # Reduced from 50 for faster generation
                    logger.info(f"Calling pipe with width={width}, height={height}, steps={inference_steps}")
                    try:
                        image = pipe(
                            prompt=prompt,
                            negative_prompt="",
                            width=width,
                            height=height,
                            num_inference_steps=inference_steps,
                            true_cfg_scale=4.0,
                            generator=torch_module.Generator(device=device).manual_seed(seed + i) if seed is not None else None
                        ).images[0]
                        logger.info(f"Successfully generated image {i+1}")
                    except (AttributeError, RuntimeError, TypeError) as attr_error:
                        error_str = str(attr_error).lower()
                        if "is_compiling" in error_str or "torch.compiler" in error_str:
                            # PyTorch compiler compatibility issue
                            logger.warning(f"PyTorch compiler compatibility issue: {attr_error}")
                            logger.info("Trying with scaled_dot_product_attention disabled...")
                            # Disable flash attention / SDPA
                            try:
                                torch_module.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
                            except:
                                pass
                            # Retry with reduced steps
                            inference_steps = 30
                            image = pipe(
                                prompt=prompt,
                                negative_prompt="",
                                width=width,
                                height=height,
                                num_inference_steps=inference_steps,
                                true_cfg_scale=4.0,
                                generator=torch_module.Generator(device=device).manual_seed(seed + i) if seed is not None else None
                            ).images[0]
                        elif "enable_gqa" in error_str or "unexpected keyword argument" in error_str:
                            # PyTorch version doesn't support enable_gqa parameter
                            logger.warning(f"PyTorch version compatibility issue (enable_gqa): {attr_error}")
                            logger.info("Trying with different attention settings...")
                            # Try to disable problematic attention mechanisms
                            try:
                                # Disable all SDPA optimizations
                                torch_module.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
                            except:
                                pass
                            # Set environment variable to disable GQA if possible
                            import os
                            original_env = os.environ.get("PYTORCH_ENABLE_GQA", None)
                            os.environ["PYTORCH_ENABLE_GQA"] = "0"
                            try:
                                # Retry generation with reduced steps
                                inference_steps = 30
                                image = pipe(
                                    prompt=prompt,
                                    negative_prompt="",
                                    width=width,
                                    height=height,
                                    num_inference_steps=inference_steps,
                                    true_cfg_scale=4.0,
                                    generator=torch_module.Generator(device=device).manual_seed(seed + i) if seed is not None else None
                                ).images[0]
                            finally:
                                # Restore environment
                                if original_env is None:
                                    os.environ.pop("PYTORCH_ENABLE_GQA", None)
                                else:
                                    os.environ["PYTORCH_ENABLE_GQA"] = original_env
                        else:
                            raise
                    
                    # Resize to exact HAM10000 format (600x450)
                    if image.size != (600, 450):
                        image = image.resize((600, 450), Image.Resampling.LANCZOS)
                    
                    generated_images.append({
                        "image": image,
                        "prompt": prompt,
                        "model": "qwen",
                        "model_id": model_id,
                        "image_size": (600, 450)
                    })
                    
            except Exception as e:
                logger.error(f"Error generating image {i} with QWEN: {e}")
                continue
        
        logger.info(f"Generated {len(generated_images)} images using Qwen-Image")
        
        # Clear CUDA cache after generation
        if device == "cuda":
            torch_module.cuda.empty_cache()
        
        # Restore original torch compile setting
        if original_torch_compile is None:
            os.environ.pop("TORCH_COMPILE_DISABLE", None)
        else:
            os.environ["TORCH_COMPILE_DISABLE"] = original_torch_compile
        
        return generated_images
        
    except torch_module.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA out of memory error with Qwen-Image: {e}")
        # Clear cache before trying fallback
        if device == "cuda":
            torch_module.cuda.empty_cache()
            torch_module.cuda.synchronize()
        # Restore original torch compile setting
        if original_torch_compile is None:
            os.environ.pop("TORCH_COMPILE_DISABLE", None)
        else:
            os.environ["TORCH_COMPILE_DISABLE"] = original_torch_compile
        logger.warning("Qwen-Image failed due to OOM. Cannot fallback to other GPU models. Please free GPU memory or use CPU.")
        raise RuntimeError(
            "CUDA out of memory. Please:\n"
            "1. Free GPU memory by closing other applications\n"
            "2. Reduce batch_size or num_images\n"
            "3. Restart the application to clear memory\n"
            f"Original error: {e}"
        )
    except Exception as e:
        logger.error(f"Error loading Qwen-Image model: {e}")
        # Clear cache before trying fallback
        if device == "cuda":
            torch_module.cuda.empty_cache()
            torch_module.cuda.synchronize()
        # Restore original torch compile setting
        if original_torch_compile is None:
            os.environ.pop("TORCH_COMPILE_DISABLE", None)
        else:
            os.environ["TORCH_COMPILE_DISABLE"] = original_torch_compile
        
        # Only fallback if it's not a memory error
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            logger.warning("Memory error detected. Cannot fallback to other GPU models.")
            raise RuntimeError(
                "GPU memory error. Please free GPU memory or reduce batch size.\n"
                f"Original error: {e}"
            )
        
        logger.info("Falling back to Stable Diffusion")
        try:
            return generate_stable_diffusion_images(
                prompts, num_images, image_size, batch_size, seed
            )
        except Exception as fallback_error:
            logger.error(f"Fallback to Stable Diffusion also failed: {fallback_error}")
            raise RuntimeError(
                f"Both Qwen-Image and fallback model failed.\n"
                f"Qwen-Image error: {e}\n"
                f"Fallback error: {fallback_error}"
            )


def generate_flux_images(
    prompts: List[str],
    num_images: int,
    image_size: int = 512,
    batch_size: int = 4,
    seed: Optional[int] = None,
    model_id: str = "black-forest-labs/FLUX.1-dev"
) -> List[Dict[str, Any]]:
    """
    Generate images using FLUX model.
    
    Args:
        prompts: List of text prompts for image generation
        num_images: Total number of images to generate
        image_size: Size of generated images
        batch_size: Number of images to generate per batch
        seed: Random seed for reproducibility
        model_id: Hugging Face model identifier
        
    Returns:
        List of dictionaries containing images and metadata
    """
    logger.info(f"Loading FLUX model: {model_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Clear CUDA cache before loading
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    try:
        # Load FLUX pipeline
        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        
        # Always use CPU offloading for FLUX to save VRAM (it's a large model)
        if device == "cuda":
            try:
                pipe.enable_model_cpu_offload()  # Save VRAM by offloading to CPU
            except Exception as e:
                logger.warning(f"Could not enable CPU offloading: {e}. Loading to GPU directly.")
                pipe = pipe.to(device)
            
            # Enable memory optimizations
            try:
                pipe.enable_attention_slicing(slice_size="auto")
                logger.info("Enabled attention slicing for memory optimization")
            except Exception as e:
                logger.warning(f"Could not enable attention slicing: {e}")
            
            try:
                pipe.enable_vae_tiling()
                logger.info("Enabled VAE tiling for memory optimization")
            except Exception as e:
                logger.warning(f"Could not enable VAE tiling: {e}")
        else:
            pipe = pipe.to(device)
        
        if seed is not None:
            torch.manual_seed(seed)
        
        generated_images = []
        num_batches = (num_images + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Generating images"):
            batch_prompts = []
            for i in range(batch_size):
                if len(generated_images) >= num_images:
                    break
                prompt_idx = len(generated_images) % len(prompts)
                batch_prompts.append(prompts[prompt_idx])
            
            if not batch_prompts:
                break
            
            try:
                with torch.no_grad():
                    # HAM10000 format: 600x450
                    # Stable Diffusion requires dimensions divisible by 8
                    # Using 600x448 for generation, will resize to 600x450
                    height = 448  # Divisible by 8
                    width = 600   # Divisible by 8
                    
                    # FLUX supports batch generation
                    if len(batch_prompts) == 1:
                        result = pipe(
                            prompt=batch_prompts[0],
                            height=height,
                            width=width,
                            guidance_scale=3.5,
                            num_inference_steps=50,
                            max_sequence_length=512,
                            generator=torch.Generator(device=device).manual_seed(seed + batch_idx) if seed is not None else None
                        )
                        images = [result.images[0]]
                    else:
                        # Generate one at a time for batch
                        images = []
                        for prompt in batch_prompts:
                            result = pipe(
                                prompt=prompt,
                                height=height,
                                width=width,
                                guidance_scale=3.5,
                                num_inference_steps=50,
                                max_sequence_length=512,
                                generator=torch.Generator(device=device).manual_seed(seed + len(images)) if seed is not None else None
                            )
                            images.append(result.images[0])
                
                    for img, prompt in zip(images, batch_prompts):
                        # Ensure image is exactly 600x450 (HAM10000 format)
                        if img.size != (600, 450):
                            img = img.resize((600, 450), Image.Resampling.LANCZOS)
                        
                        generated_images.append({
                            "image": img,
                            "prompt": prompt,
                            "model": "flux",
                            "model_id": model_id,
                            "image_size": (600, 450)
                        })
                    
                    if len(generated_images) >= num_images:
                        break
                        
            except Exception as e:
                logger.error(f"Error generating batch {batch_idx}: {e}")
                continue
        
        logger.info(f"Generated {len(generated_images)} images using FLUX")
        return generated_images
        
    except Exception as e:
        logger.error(f"Error loading FLUX model: {e}")
        logger.info("Falling back to Stable Diffusion")
        return generate_stable_diffusion_images(
            prompts, num_images, image_size, batch_size, seed
        )


def generate_images(
    model_name: str,
    prompts: List[str],
    num_images: int,
    image_size: int = 512,
    batch_size: int = 4,
    seed: Optional[int] = None,
    reference_images: Optional[List[Image.Image]] = None,
    img2img_strength: float = 0.75
) -> List[Dict[str, Any]]:
    """
    Generate images using the specified model.
    
    Args:
        model_name: Name of the generation model ('stable_diffusion', 'sdxl', 'qwen', or 'flux')
        prompts: List of text prompts
        num_images: Number of images to generate
        image_size: Size of generated images
        batch_size: Batch size for generation
        seed: Random seed
        reference_images: Optional reference images for image-guided generation
        img2img_strength: Strength for image-to-image (0.0-1.0)
        
    Returns:
        List of generated images with metadata
    """
    # Validate inputs
    if not prompts:
        logger.error("No prompts provided for image generation")
        raise ValueError("Prompts list cannot be empty")
    
    if num_images <= 0:
        logger.error(f"Invalid num_images: {num_images}")
        raise ValueError(f"num_images must be positive, got {num_images}")
    
    logger.info(f"Generating {num_images} images using {model_name} with {len(prompts)} prompts")
    if reference_images:
        logger.info(f"Using {len(reference_images)} reference images for image-guided generation")
    
    model_name = model_name.lower()
    
    if model_name == "stable_diffusion":
        return generate_stable_diffusion_images(
            prompts, num_images, image_size, batch_size, seed,
            reference_images=reference_images,
            img2img_strength=img2img_strength
        )
    elif model_name == "sd3.5":
        return generate_sdxl_images(
            prompts, num_images, image_size, batch_size, seed,
            reference_images=reference_images,
            img2img_strength=img2img_strength
        )
    elif model_name == "qwen":
        return generate_qwen_images(
            prompts, num_images, image_size, batch_size, seed
        )
    elif model_name == "flux":
        return generate_flux_images(
            prompts, num_images, image_size, batch_size, seed
        )
    else:
        raise ValueError(f"Unknown generation model: {model_name}. "
                       f"Supported: stable_diffusion, sd3.5, qwen, flux")


def create_prompts(
    lesion_types: List[str], 
    prompt_template: str, 
    num_images: int,
    ham10000_dataset_path: Optional[str] = None,
    use_ham10000_prompts: bool = False
) -> List[str]:
    """
    Create prompts for image generation.
    
    Args:
        lesion_types: List of lesion type names
        prompt_template: Template string with {lesion_type} placeholder
        num_images: Number of images to generate
        ham10000_dataset_path: Path to HAM10000 dataset (optional)
        use_ham10000_prompts: Whether to use HAM10000-based prompts
        
    Returns:
        List of formatted prompts
    """
    # Use HAM10000 prompts if available and requested
    if use_ham10000_prompts and ham10000_dataset_path:
        try:
            from synthetic_data_gen.pipelines.data_generation.ham10000_prompts import generate_ham10000_prompts
            
            # Map lesion types to HAM10000 codes
            ham10000_codes = []
            for lt in lesion_types:
                lt_lower = lt.lower()
                if 'melanoma' in lt_lower or 'malignant' in lt_lower:
                    ham10000_codes.append('mel')
                elif 'benign' in lt_lower or 'nevus' in lt_lower:
                    ham10000_codes.append('nv')
                elif 'keratosis' in lt_lower:
                    ham10000_codes.append('bkl')
                else:
                    ham10000_codes.extend(['nv', 'mel', 'bkl'])  # Default to common types
            
            prompts = generate_ham10000_prompts(
                dataset_path=ham10000_dataset_path,
                num_prompts=num_images,
                lesion_types=ham10000_codes if ham10000_codes else None,
                match_distribution=True
            )
            logger.info(f"Generated {len(prompts)} prompts using HAM10000 metadata")
            return prompts
        except Exception as e:
            logger.warning(f"Could not use HAM10000 prompts, falling back to template: {e}")
    
    # Fallback to template-based prompts
    prompts = []
    for i in range(num_images):
        lesion_type = lesion_types[i % len(lesion_types)]
        prompt = prompt_template.format(lesion_type=lesion_type)
        prompts.append(prompt)
    
    return prompts

