"""HAM10000-based prompt generation for improved image generation."""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import random
import logging

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


def load_ham10000_metadata(dataset_path: str) -> Optional[pd.DataFrame]:
    """Load HAM10000 metadata."""
    metadata_path = Path(dataset_path) / "HAM10000_metadata"
    if metadata_path.exists():
        try:
            return pd.read_csv(metadata_path)
        except Exception as e:
            logger.warning(f"Could not load HAM10000 metadata: {e}")
    return None


def create_metadata_based_prompt(
    diagnosis: str,
    localization: Optional[str] = None,
    age: Optional[float] = None,
    sex: Optional[str] = None
) -> str:
    """
    Create a detailed prompt based on HAM10000 metadata.
    
    Args:
        diagnosis: Diagnosis code (e.g., 'mel', 'nv', 'bkl')
        localization: Body location
        age: Patient age
        sex: Patient sex
        
    Returns:
        Detailed prompt string
    """
    diagnosis_full = DX_MAPPING.get(diagnosis, diagnosis)
    
    prompt_parts = [
        "A high-quality dermoscopic image of a",
        diagnosis_full,
    ]
    
    if localization and localization != 'unknown':
        loc_desc = LOCALIZATION_DESCRIPTIONS.get(localization, f"on the {localization}")
        if loc_desc:
            prompt_parts.append(loc_desc)
    
    if age is not None and not pd.isna(age):
        age_group = "elderly" if age >= 65 else "middle-aged" if age >= 40 else "young"
        if sex and sex != 'unknown':
            prompt_parts.append(f"in a {age_group} {sex}")
        else:
            prompt_parts.append(f"in a {age_group} patient")
    
    prompt_parts.extend([
        "medical photography",
        "detailed",
        "professional",
        "clinical quality",
        "dermoscopy"
    ])
    
    return ", ".join(prompt_parts)


def generate_ham10000_prompts(
    dataset_path: str,
    num_prompts: int,
    lesion_types: Optional[List[str]] = None,
    match_distribution: bool = True
) -> List[str]:
    """
    Generate prompts based on HAM10000 dataset distribution.
    
    Args:
        dataset_path: Path to HAM10000 dataset
        num_prompts: Number of prompts to generate
        lesion_types: Filter by specific lesion types (optional)
        match_distribution: Whether to match the actual distribution in dataset
        
    Returns:
        List of prompts
    """
    metadata = load_ham10000_metadata(dataset_path)
    
    if metadata is None:
        logger.warning("HAM10000 metadata not found, using default prompts")
        return [
            "A high-quality dermoscopic image of a skin lesion, medical photography, detailed, professional"
        ] * num_prompts
    
    # Filter by lesion types if specified
    if lesion_types:
        metadata = metadata[metadata['dx'].isin(lesion_types)]
    
    if len(metadata) == 0:
        logger.warning("No matching metadata found, using default prompts")
        return [
            "A high-quality dermoscopic image of a skin lesion, medical photography, detailed, professional"
        ] * num_prompts
    
    prompts = []
    
    if match_distribution:
        # Sample according to actual distribution
        for _ in range(num_prompts):
            row = metadata.sample(n=1).iloc[0]
            prompt = create_metadata_based_prompt(
                diagnosis=row.get('dx', 'nv'),
                localization=row.get('localization', None),
                age=row.get('age', None),
                sex=row.get('sex', None)
            )
            prompts.append(prompt)
    else:
        # Uniform sampling
        sampled = metadata.sample(n=min(num_prompts, len(metadata)), replace=True)
        for _, row in sampled.iterrows():
            prompt = create_metadata_based_prompt(
                diagnosis=row.get('dx', 'nv'),
                localization=row.get('localization', None),
                age=row.get('age', None),
                sex=row.get('sex', None)
            )
            prompts.append(prompt)
    
    logger.info(f"Generated {len(prompts)} prompts based on HAM10000 metadata")
    return prompts

