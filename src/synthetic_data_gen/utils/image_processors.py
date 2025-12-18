"""Image preprocessing utilities for the pipeline."""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Union, Tuple
import logging

logger = logging.getLogger(__name__)


def get_preprocessing_transform(image_size: int = 224, normalize: bool = True) -> transforms.Compose:
    """
    Get preprocessing transform for classification models.
    
    Args:
        image_size: Target image size (default: 224 for most models)
        normalize: Whether to normalize with ImageNet statistics
        
    Returns:
        Composed transform
    """
    transform_list = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        )
    
    return transforms.Compose(transform_list)


def preprocess_image(image: Union[Image.Image, np.ndarray, str], 
                    image_size: int = 224) -> torch.Tensor:
    """
    Preprocess a single image for model input.
    
    Args:
        image: PIL Image, numpy array, or path to image file
        image_size: Target image size
        
    Returns:
        Preprocessed tensor
    """
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
    elif not isinstance(image, Image.Image):
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    transform = get_preprocessing_transform(image_size=image_size)
    return transform(image)


def preprocess_batch(images: List[Union[Image.Image, np.ndarray, str]], 
                    image_size: int = 224) -> torch.Tensor:
    """
    Preprocess a batch of images.
    
    Args:
        images: List of images (PIL, numpy, or file paths)
        image_size: Target image size
        
    Returns:
        Batch tensor of shape (N, 3, H, W)
    """
    processed = [preprocess_image(img, image_size) for img in images]
    return torch.stack(processed)


def resize_image(image: Union[Image.Image, np.ndarray, str], 
                size: Tuple[int, int]) -> Image.Image:
    """
    Resize an image to specified dimensions.
    
    Args:
        image: PIL Image, numpy array, or path to image file
        size: Target size (width, height)
        
    Returns:
        Resized PIL Image
    """
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
    
    return image.resize(size, Image.Resampling.LANCZOS)


def save_image(image: Union[Image.Image, np.ndarray], path: str) -> None:
    """
    Save an image to disk.
    
    Args:
        image: PIL Image or numpy array
        path: Output file path
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image.save(path)
    logger.debug(f"Saved image to {path}")

