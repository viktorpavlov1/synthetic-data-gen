"""HAM10000 dataset loader with metadata integration."""

import pandas as pd
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional
import logging
import random

logger = logging.getLogger(__name__)

# HAM10000 diagnosis codes to full names
DX_MAPPING = {
    'nv': 'melanocytic nevus',
    'mel': 'melanoma',
    'bkl': 'benign keratosis',
    'bcc': 'basal cell carcinoma',
    'akiec': 'actinic keratosis',
    'vasc': 'vascular lesion',
    'df': 'dermatofibroma'
}

# Localization descriptions
LOCALIZATION_DESCRIPTIONS = {
    'back': 'on the back',
    'lower extremity': 'on the lower extremity',
    'trunk': 'on the trunk',
    'upper extremity': 'on the upper extremity',
    'abdomen': 'on the abdomen',
    'face': 'on the face',
    'chest': 'on the chest',
    'foot': 'on the foot',
    'neck': 'on the neck',
    'scalp': 'on the scalp',
    'ear': 'on the ear',
    'unknown': ''
}


class HAM10000Loader:
    """Loader for HAM10000 dataset with metadata."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize HAM10000 loader.
        
        Args:
            dataset_path: Path to HAM10000 dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.metadata_path = self.dataset_path / "HAM10000_metadata"
        self.images_dir = self.dataset_path / "images"
        
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Load metadata
        self.metadata = None
        self._load_metadata()
    
    def _load_metadata(self):
        """Load HAM10000 metadata."""
        if self.metadata_path.exists():
            try:
                self.metadata = pd.read_csv(self.metadata_path)
                logger.info(f"Loaded metadata for {len(self.metadata)} entries")
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
                self.metadata = None
        else:
            logger.warning(f"Metadata file not found: {self.metadata_path}")
            self.metadata = None
    
    def load_images_with_metadata(
        self,
        max_images: Optional[int] = None,
        lesion_types: Optional[List[str]] = None,
        localization: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load images with their metadata.
        
        Args:
            max_images: Maximum number of images to load
            lesion_types: Filter by lesion types (e.g., ['mel', 'nv'])
            localization: Filter by localization (e.g., 'back')
            
        Returns:
            List of image dictionaries with metadata
        """
        images = []
        
        # Filter metadata if needed
        metadata = self.metadata.copy() if self.metadata is not None else None
        
        if metadata is not None:
            if lesion_types:
                metadata = metadata[metadata['dx'].isin(lesion_types)]
            
            if localization:
                metadata = metadata[metadata['localization'] == localization]
            
            # Limit number of images
            if max_images:
                metadata = metadata.head(max_images)
            
            logger.info(f"Loading {len(metadata)} images with metadata")
            
            for _, row in metadata.iterrows():
                image_id = row['image_id']
                image_path = self.images_dir / f"{image_id}.jpg"
                
                if image_path.exists():
                    try:
                        img = Image.open(image_path).convert('RGB')
                        
                        # Create rich metadata dictionary
                        img_data = {
                            'image': img,
                            'path': str(image_path),
                            'filename': image_path.name,
                            'image_id': image_id,
                            'lesion_id': row.get('lesion_id', ''),
                            'diagnosis': row.get('dx', ''),
                            'diagnosis_full': DX_MAPPING.get(row.get('dx', ''), row.get('dx', '')),
                            'dx_type': row.get('dx_type', ''),
                            'age': row.get('age', None),
                            'sex': row.get('sex', 'unknown'),
                            'localization': row.get('localization', 'unknown'),
                            'dataset': row.get('dataset', '')
                        }
                        
                        images.append(img_data)
                    except Exception as e:
                        logger.warning(f"Could not load image {image_path}: {e}")
                        continue
        else:
            # Fallback: load images without metadata
            logger.warning("No metadata available, loading images without metadata")
            image_files = list(self.images_dir.glob("*.jpg"))
            
            if max_images:
                image_files = image_files[:max_images]
            
            for img_path in image_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append({
                        'image': img,
                        'path': str(img_path),
                        'filename': img_path.name
                    })
                except Exception as e:
                    logger.warning(f"Could not load image {img_path}: {e}")
                    continue
        
        logger.info(f"Successfully loaded {len(images)} images")
        return images

    def sample_images(
        self,
        count: int,
        lesion_types: Optional[List[str]] = None,
        localization: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Efficiently sample random images matching criteria without loading everything.
        
        Args:
            count: Number of images to sample
            lesion_types: Filter by lesion types
            localization: Filter by localization
            
        Returns:
            List of image dictionaries with metadata
        """
        if self.metadata is None:
            return []
            
        df = self.metadata.copy()
        
        if lesion_types:
            df = df[df['dx'].isin(lesion_types)]
            
        if localization:
            df = df[df['localization'] == localization]
            
        if len(df) == 0:
            return []
            
        # Sample metadata first
        sample_size = min(count, len(df))
        sampled_df = df.sample(n=sample_size)
        
        images = []
        for _, row in sampled_df.iterrows():
            image_id = row['image_id']
            image_path = self.images_dir / f"{image_id}.jpg"
            
            if image_path.exists():
                try:
                    img = Image.open(image_path).convert('RGB')
                    img_data = {
                        'image': img,
                        'path': str(image_path),
                        'filename': image_path.name,
                        'image_id': image_id,
                        'diagnosis': row.get('dx', ''),
                        'diagnosis_full': DX_MAPPING.get(row.get('dx', ''), row.get('dx', '')),
                        'localization': row.get('localization', 'unknown')
                    }
                    images.append(img_data)
                except Exception as e:
                    logger.warning(f"Could not load image {image_path}: {e}")
                    
        return images
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if self.metadata is None:
            return {}
        
        stats = {
            'total_images': len(self.metadata),
            'diagnosis_distribution': self.metadata['dx'].value_counts().to_dict(),
            'localization_distribution': self.metadata['localization'].value_counts().to_dict(),
            'age_range': {
                'min': float(self.metadata['age'].min()) if not self.metadata['age'].isna().all() else None,
                'max': float(self.metadata['age'].max()) if not self.metadata['age'].isna().all() else None,
                'mean': float(self.metadata['age'].mean()) if not self.metadata['age'].isna().all() else None
            },
            'sex_distribution': self.metadata['sex'].value_counts().to_dict()
        }
        
        return stats


def create_ham10000_prompt(img_data: Dict[str, Any]) -> str:
    """
    Create a detailed prompt based on HAM10000 metadata.
    
    Args:
        img_data: Image data dictionary with metadata
        
    Returns:
        Detailed prompt string
    """
    diagnosis = img_data.get('diagnosis_full', 'skin lesion')
    localization = img_data.get('localization', '')
    age = img_data.get('age', None)
    sex = img_data.get('sex', '')
    
    # Build prompt components
    prompt_parts = [
        "A high-quality dermoscopic image of a",
        diagnosis,
    ]
    
    # Add localization if available
    if localization and localization != 'unknown':
        loc_desc = LOCALIZATION_DESCRIPTIONS.get(localization, f"on the {localization}")
        if loc_desc:
            prompt_parts.append(loc_desc)
    
    # Add demographic info if available
    if age is not None and not pd.isna(age):
        age_group = "elderly" if age >= 65 else "middle-aged" if age >= 40 else "young"
        prompt_parts.append(f"in a {age_group} {sex}" if sex != 'unknown' else f"in a {age_group} patient")
    
    prompt_parts.extend([
        "medical photography",
        "detailed",
        "professional",
        "clinical quality"
    ])
    
    return ", ".join(prompt_parts)


def create_prompts_from_ham10000(
    dataset_images: List[Dict[str, Any]],
    num_prompts: int
) -> List[str]:
    """
    Create prompts based on HAM10000 metadata distribution.
    
    Args:
        dataset_images: List of images with HAM10000 metadata
        num_prompts: Number of prompts to generate
        
    Returns:
        List of prompts
    """
    prompts = []
    
    # Sample from dataset to match distribution
    for i in range(num_prompts):
        # Randomly sample an image to get realistic metadata
        sample_img = random.choice(dataset_images)
        prompt = create_ham10000_prompt(sample_img)
        prompts.append(prompt)
    
    return prompts

