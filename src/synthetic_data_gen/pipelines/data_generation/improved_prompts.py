"""Improved prompt generation based on best practices and user examples."""

from typing import List, Dict, Any, Optional
import random
import logging

logger = logging.getLogger(__name__)

# HAM10000 diagnosis codes to full names and descriptions
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

# Common localizations for realistic prompts
COMMON_LOCALIZATIONS = [
    'back', 'trunk', 'upper extremity', 'lower extremity', 
    'abdomen', 'chest', 'face', 'neck'
]


def create_improved_prompt(
    diagnosis: str,
    localization: Optional[str] = None,
    use_ham10000_style: bool = True
) -> str:
    """
    Create an improved prompt based on best practices and user example.
    
    Args:
        diagnosis: Diagnosis code (e.g., 'mel', 'nv', 'bkl')
        localization: Body location (optional)
        use_ham10000_style: Use ISIC-style formatting
        
    Returns:
        High-quality prompt string
    """
    dx_info = DX_MAPPING.get(diagnosis, {'name': diagnosis, 'short': diagnosis, 'features': ''})
    lesion_name = dx_info['name']
    lesion_short = dx_info['short']
    
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
    # Simple estimation: ~1 token ≈ 4-5 characters for English text
    # 77 tokens ≈ 300-350 characters
    char_limit = 300  # Conservative estimate for 77 tokens
    
    if len(prompt) > char_limit:
        truncated = prompt[:char_limit]
        last_comma = truncated.rfind(',')
        if last_comma > char_limit * 0.8:
            truncated = truncated[:last_comma]
        logger.warning(f"Prompt truncated from {len(prompt)} to {len(truncated)} characters")
        prompt = truncated
    
    return prompt


def generate_prompts_with_distribution(
    selected_diagnoses: List[str],
    num_prompts: int,
    distribution: Optional[Dict[str, float]] = None,
    use_localization: bool = True
) -> List[str]:
    """
    Generate prompts with controlled distribution.
    
    Args:
        selected_diagnoses: List of diagnosis codes to use
        num_prompts: Total number of prompts to generate
        distribution: Dictionary mapping diagnosis to proportion (0.0-1.0)
                     If None, uses equal distribution
        use_localization: Whether to include localization in prompts
        
    Returns:
        List of prompts
    """
    if not selected_diagnoses:
        raise ValueError("At least one diagnosis must be selected")
    
    # Normalize distribution if provided
    if distribution:
        # Filter to only selected diagnoses
        distribution = {k: v for k, v in distribution.items() if k in selected_diagnoses}
        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v / total for k, v in distribution.items()}
        else:
            distribution = None
    
    # Use equal distribution if not provided
    if not distribution:
        proportion = 1.0 / len(selected_diagnoses)
        distribution = {dx: proportion for dx in selected_diagnoses}
    
    # Generate prompts according to distribution
    prompts = []
    diagnosis_counts = {}
    
    for dx in selected_diagnoses:
        count = int(num_prompts * distribution.get(dx, 0))
        diagnosis_counts[dx] = count
    
    # Adjust for rounding errors
    total_assigned = sum(diagnosis_counts.values())
    if total_assigned < num_prompts:
        remaining = num_prompts - total_assigned
        # Distribute remaining to most common diagnosis
        most_common = max(distribution.items(), key=lambda x: x[1])[0]
        diagnosis_counts[most_common] += remaining
    
    # Generate prompts
    for dx, count in diagnosis_counts.items():
        for _ in range(count):
            localization = None
            if use_localization:
                localization = random.choice(COMMON_LOCALIZATIONS)
            
            prompt = create_improved_prompt(diagnosis=dx, localization=localization)
            prompts.append(prompt)
    
    # Shuffle to mix diagnoses
    random.shuffle(prompts)
    
    logger.info(f"Generated {len(prompts)} prompts with distribution: {diagnosis_counts}")
    return prompts


def generate_single_diagnosis_prompts(
    diagnosis: str,
    num_prompts: int,
    use_localization: bool = True
) -> List[str]:
    """
    Generate prompts for a single diagnosis type.
    
    Args:
        diagnosis: Diagnosis code
        num_prompts: Number of prompts to generate
        use_localization: Whether to vary localization
        
    Returns:
        List of prompts
    """
    prompts = []
    
    for i in range(num_prompts):
        localization = None
        if use_localization:
            localization = random.choice(COMMON_LOCALIZATIONS)
        
        prompt = create_improved_prompt(diagnosis=diagnosis, localization=localization)
        prompts.append(prompt)
    
    return prompts

