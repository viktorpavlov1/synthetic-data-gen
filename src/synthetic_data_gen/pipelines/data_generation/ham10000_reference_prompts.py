"""Generate prompts with HAM10000 reference images for image-guided generation."""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import random
import logging
from PIL import Image

logger = logging.getLogger(__name__)


def truncate_prompt_for_clip(prompt: str, max_tokens: int = 77) -> str:
    """
    Truncate prompt to fit within CLIP's token limit while keeping most important parts.
    
    Args:
        prompt: Full prompt string
        max_tokens: Maximum tokens (77 for CLIP)
        
    Returns:
        Truncated prompt that fits within token limit
    """
    # Simple estimation: ~1 token ≈ 4-5 characters for English text
    # 77 tokens ≈ 300-350 characters
    # Use a conservative estimate to ensure we're under the limit
    char_limit = 300  # Conservative estimate for 77 tokens
    
    if len(prompt) <= char_limit:
        return prompt
    
    # Truncate at word boundary to avoid cutting words
    truncated = prompt[:char_limit]
    last_comma = truncated.rfind(',')
    if last_comma > char_limit * 0.8:  # If we find a comma in the last 20% of the string
        truncated = truncated[:last_comma]
    
    logger.warning(f"Prompt truncated from {len(prompt)} to {len(truncated)} characters (estimated {len(truncated)//4} tokens)")
    return truncated

# HAM10000 diagnosis codes to full names
DX_MAPPING = {
    'akiec': {
        'name': 'actinic keratoses / intraepithelial carcinoma',
        'short': 'actinic keratoses',
        'features': 'scaly surface, irregular borders, erythematous base'
    },
    'bcc': {
        'name': 'basal cell carcinoma',
        'short': 'basal cell carcinoma',
        'features': 'rolled borders, telangiectasias, pearly appearance'
    },
    'bkl': {
        'name': 'benign keratosis-like lesions',
        'short': 'benign keratosis',
        'features': 'well-defined borders, scaly surface, uniform color'
    },
    'df': {
        'name': 'dermatofibroma',
        'short': 'dermatofibroma',
        'features': 'central dimple, firm texture, brownish color'
    },
    'nv': {
        'name': 'melanocytic nevi',
        'short': 'melanocytic nevus',
        'features': 'regular borders, uniform pigment network, symmetric pattern'
    },
    'mel': {
        'name': 'melanoma',
        'short': 'melanoma',
        'features': 'irregular borders, asymmetric pattern, variegated color, atypical pigment network'
    },
    'vasc': {
        'name': 'vascular lesions',
        'short': 'vascular lesion',
        'features': 'reddish color, lacunar structures, vascular patterns'
    }
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


def create_improved_prompt_with_reference(
    diagnosis: str,
    localization: Optional[str] = None
) -> str:
    """
    Create an improved prompt based on best practices.
    
    Args:
        diagnosis: Diagnosis code (e.g., 'mel', 'nv', 'bkl')
        localization: Body location (optional)
        
    Returns:
        High-quality prompt string
    """
    dx_info = DX_MAPPING.get(diagnosis, {'name': diagnosis, 'short': diagnosis, 'features': ''})
    lesion_name = dx_info['name']
    
    # Base prompt structure - ULTRA CONCISE for CLIP 77 token limit
    # PRIORITIZED: Most important parts first (will be kept if truncated)
    # Target: ~60-70 tokens to be safe
    
    # Core elements (most important)
    prompt_parts = [
        "dermoscopic skin lesion",
        lesion_name,
        "extreme close-up macro",
        "ISIC-style",
        "skin only, no body parts",
        "dermatoscope",
        "pink skin background",
        "lesion visible",
        "clinical lighting",
        "sharp focus"
    ]
    
    # Add diagnosis-specific color (very concise)
    if diagnosis == 'mel':
        prompt_parts.append("irregular, variegated, dark brown")
    elif diagnosis == 'nv':
        prompt_parts.append("regular borders, brown")
    elif diagnosis == 'bcc':
        prompt_parts.append("pearly white")
    elif diagnosis == 'vasc':
        prompt_parts.append("reddish")
    
    # Add localization if provided (very short)
    if localization and localization != 'unknown':
        # Shorten localization names
        loc_short = localization.replace('lower extremity', 'leg').replace('upper extremity', 'arm')
        prompt_parts.append(f"on {loc_short}")
    
    # Final quality (minimal)
    prompt_parts.append("clinical photo")
    
    prompt = ", ".join(prompt_parts)
    
    # Truncate if needed to fit CLIP's 77 token limit
    return truncate_prompt_for_clip(prompt, max_tokens=77)


def get_negative_prompt() -> str:
    """
    Get negative prompt to exclude unwanted content.
    
    Returns:
        Negative prompt string
    """
    return ", ".join([
        "face",
        "facial features",
        "eyes",
        "nose",
        "mouth",
        "head",
        "person",
        "body",
        "torso",
        "chest",
        "back",
        "limbs",
        "arms",
        "legs",
        "hands",
        "feet",
        "full body",
        "portrait",
        "human figure",
        "clothing",
        "clothes",
        "background",
        "landscape",
        "artistic",
        "painting",
        "drawing",
        "illustration",
        "cartoon",
        "anime",
        "3d render",
        "blurry",
        "low quality",
        "distorted",
        "deformed"
    ])


def generate_prompts_with_references(
    dataset_path: str,
    selected_diagnoses: List[str],
    num_prompts: int,
    distribution: Optional[Dict[str, float]] = None,
    reference_percentage: float = 1.0
) -> Tuple[List[str], List[Image.Image], List[Image.Image]]:
    """
    Generate prompts with matching HAM10000 reference images.
    
    Args:
        dataset_path: Path to HAM10000 dataset
        selected_diagnoses: List of diagnosis codes to use
        num_prompts: Number of prompts to generate
        distribution: Optional distribution dictionary
        reference_percentage: Percentage of HAM10000 images to use as references (0.0-1.0)
                             e.g., 0.1 = 10% (1000 images), 1.0 = 100% (all 10000 images)
        
    Returns:
        Tuple of (prompts list, reference_images list for prompts, full_reference_pool)
        - prompts: List of prompts (one per image to generate)
        - reference_images: List of reference images paired with prompts (for initial pairing)
        - full_reference_pool: Full pool of reference images to randomly select from
    """
    from synthetic_data_gen.utils.ham10000_loader import HAM10000Loader
    
    loader = HAM10000Loader(dataset_path)
    metadata = load_ham10000_metadata(dataset_path)
    
    if metadata is None:
        raise ValueError(f"Could not load HAM10000 metadata from {dataset_path}")
    
    # Filter by selected diagnoses
    if selected_diagnoses:
        metadata = metadata[metadata['dx'].isin(selected_diagnoses)]
    
    if len(metadata) == 0:
        raise ValueError(f"No images found for diagnoses: {selected_diagnoses}")
    
    # Calculate how many reference images to load based on percentage
    total_available = len(metadata)
    num_references_to_load = int(total_available * reference_percentage)
    num_references_to_load = max(1, min(num_references_to_load, total_available))  # At least 1, at most all
    
    logger.info(f"Loading {num_references_to_load} reference images ({reference_percentage*100:.1f}% of {total_available} available)")
    
    # Load all reference images first (up to the percentage)
    all_reference_images = []
    all_reference_metadata = []
    
    # Sample the requested percentage of images
    sampled_metadata = metadata.sample(n=num_references_to_load, replace=False)
    
    for _, row in sampled_metadata.iterrows():
        image_id = row.get('image_id', '')
        if image_id:
            image_path = Path(dataset_path) / "images" / f"{image_id}.jpg"
            if image_path.exists():
                try:
                    ref_img = Image.open(image_path).convert('RGB')
                    all_reference_images.append(ref_img)
                    all_reference_metadata.append(row)
                except Exception as e:
                    logger.warning(f"Could not load reference image {image_path}: {e}")
                    continue
    
    logger.info(f"Successfully loaded {len(all_reference_images)} reference images")
    
    if len(all_reference_images) == 0:
        raise ValueError("No reference images could be loaded from HAM10000")
    
    # Normalize distribution if provided
    if distribution:
        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v / total for k, v in distribution.items()}
        else:
            distribution = None
    
    # Use equal distribution if not provided
    if not distribution:
        proportion = 1.0 / len(selected_diagnoses)
        distribution = {dx: proportion for dx in selected_diagnoses}
    
    prompts = []
    reference_images = []
    
    # Calculate how many of each diagnosis for prompts
    diagnosis_counts = {}
    for dx in selected_diagnoses:
        count = int(num_prompts * distribution.get(dx, 0))
        diagnosis_counts[dx] = count
    
    # Adjust for rounding
    total_assigned = sum(diagnosis_counts.values())
    if total_assigned < num_prompts:
        remaining = num_prompts - total_assigned
        most_common = max(distribution.items(), key=lambda x: x[1])[0]
        diagnosis_counts[most_common] += remaining
    
    # Generate prompts and match with reference images
    for dx, count in diagnosis_counts.items():
        # Get reference images for this diagnosis
        dx_references = [
            (img, meta) for img, meta in zip(all_reference_images, all_reference_metadata)
            if meta.get('dx') == dx
        ]
        
        if len(dx_references) == 0:
            # If no references for this diagnosis, use any available
            dx_references = list(zip(all_reference_images, all_reference_metadata))
        
        # Generate prompts and randomly select from available references
        for i in range(count):
            # Create prompt
            # Use metadata from a random reference if available
            if dx_references:
                _, ref_meta = random.choice(dx_references)
                localization = ref_meta.get('localization', None)
            else:
                localization = None
            
            prompt = create_improved_prompt_with_reference(
                diagnosis=dx,
                localization=localization
            )
            prompts.append(prompt)
            
            # Randomly select a reference image from the pool
            if dx_references:
                ref_img, _ = random.choice(dx_references)
                reference_images.append(ref_img.copy())
            else:
                # Fallback: use any reference
                ref_img = random.choice(all_reference_images)
                reference_images.append(ref_img.copy())
    
    # Shuffle to mix diagnoses
    combined = list(zip(prompts, reference_images))
    random.shuffle(combined)
    prompts, reference_images = zip(*combined) if combined else ([], [])
    
    logger.info(f"Generated {len(prompts)} prompts. Loaded {len(all_reference_images)} reference images in pool ({reference_percentage*100:.1f}% of available). Will randomly select from pool during generation.")
    return list(prompts), list(reference_images), all_reference_images

