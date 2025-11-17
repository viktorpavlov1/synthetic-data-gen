"""Dataset loading utilities for training image generation models."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class SkinLesionDatasetLoader:
    """Loader for skin lesion image datasets."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize dataset loader.
        
        Args:
            dataset_path: Path to dataset directory
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    def load_images_from_directory(
        self,
        image_dir: str,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp'),
        max_images: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load images from a directory.
        
        Args:
            image_dir: Subdirectory or path relative to dataset_path
            extensions: Allowed image file extensions
            max_images: Maximum number of images to load (None for all)
            
        Returns:
            List of image dictionaries with 'image' and 'path' keys
        """
        image_path = self.dataset_path / image_dir if not Path(image_dir).is_absolute() else Path(image_dir)
        
        if not image_path.exists():
            logger.warning(f"Image directory not found: {image_path}")
            return []
        
        images = []
        image_files = []
        
        for ext in extensions:
            image_files.extend(list(image_path.glob(f"*{ext}")))
            image_files.extend(list(image_path.glob(f"*{ext.upper()}")))
        
        if max_images:
            image_files = image_files[:max_images]
        
        logger.info(f"Loading {len(image_files)} images from {image_path}")
        
        for img_file in image_files:
            try:
                img = Image.open(img_file).convert('RGB')
                images.append({
                    'image': img,
                    'path': str(img_file),
                    'filename': img_file.name
                })
            except Exception as e:
                logger.warning(f"Could not load image {img_file}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(images)} images")
        return images
    
    def load_from_ham10000_format(
        self,
        images_dir: str = "images",
        metadata_file: str = "HAM10000_metadata",
        max_images: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load images from HAM10000 dataset format.
        
        Args:
            images_dir: Directory containing images (relative to dataset_path)
            metadata_file: Name of metadata file
            max_images: Maximum number of images to load
            
        Returns:
            List of image dictionaries with HAM10000 metadata
        """
        from synthetic_data_gen.utils.ham10000_loader import HAM10000Loader
        
        loader = HAM10000Loader(str(self.dataset_path))
        return loader.load_images_with_metadata(max_images=max_images)
    
    def load_from_isic_format(
        self,
        images_dir: str = "images",
        metadata_file: Optional[str] = None,
        max_images: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load images from ISIC dataset format.
        
        Args:
            images_dir: Directory containing images
            metadata_file: Optional CSV file with metadata
            max_images: Maximum number of images to load
            
        Returns:
            List of image dictionaries with metadata
        """
        images = self.load_images_from_directory(images_dir, max_images=max_images)
        
        # Load metadata if available
        if metadata_file:
            metadata_path = self.dataset_path / metadata_file
            if metadata_path.exists():
                try:
                    df = pd.read_csv(metadata_path)
                    logger.info(f"Loaded metadata from {metadata_path}")
                    
                    # Merge metadata with images based on filename
                    for img_data in images:
                        filename = img_data['filename']
                        # Try to match with metadata
                        matches = df[df.iloc[:, 0].astype(str).str.contains(filename, case=False, na=False)]
                        if not matches.empty:
                            # Add metadata columns
                            for col in df.columns:
                                if col not in img_data:
                                    img_data[col] = matches.iloc[0][col] if len(matches) > 0 else None
                except Exception as e:
                    logger.warning(f"Could not load metadata: {e}")
        
        return images
    
    def load_from_structured_format(
        self,
        structure: str = "flat",  # "flat", "by_class", "train_val_test"
        max_images: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load images from common dataset structures.
        
        Args:
            structure: Dataset structure type
            max_images: Maximum number of images to load
            
        Returns:
            List of image dictionaries
        """
        all_images = []
        
        if structure == "flat":
            # All images in one directory
            all_images = self.load_images_from_directory(".", max_images=max_images)
        
        elif structure == "by_class":
            # Images organized by class folders
            for class_dir in self.dataset_path.iterdir():
                if class_dir.is_dir():
                    class_images = self.load_images_from_directory(
                        str(class_dir.relative_to(self.dataset_path)),
                        max_images=None if max_images is None else max_images // 10
                    )
                    for img in class_images:
                        img['class'] = class_dir.name
                    all_images.extend(class_images)
                    
                    if max_images and len(all_images) >= max_images:
                        all_images = all_images[:max_images]
                        break
        
        elif structure == "train_val_test":
            # Standard train/val/test split
            for split in ['train', 'val', 'test']:
                split_path = self.dataset_path / split
                if split_path.exists():
                    split_images = self.load_images_from_directory(
                        split,
                        max_images=None if max_images is None else max_images // 3
                    )
                    for img in split_images:
                        img['split'] = split
                    all_images.extend(split_images)
        
        if max_images and len(all_images) > max_images:
            all_images = all_images[:max_images]
        
        return all_images

