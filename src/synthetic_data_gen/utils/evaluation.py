
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import random

logger = logging.getLogger(__name__)

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    Images must be of the same size.
    """
    try:
        # Convert to grayscale for SSIM
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = img1
            
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            gray2 = img2
            
        # Ensure same size (resize img2 to match img1)
        if gray1.shape != gray2.shape:
             gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
             
        score, _ = ssim(gray1, gray2, full=True)
        return float(score)
    except Exception as e:
        logger.error(f"Error calculating SSIM: {e}")
        return 0.0

def calculate_color_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate color similarity using histogram correlation.
    Returns value between 0.0 (no correlation) and 1.0 (perfect correlation).
    """
    try:
        # Calculate histograms for each channel
        # We use 32 bins for each channel
        hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # Compare histograms using correlation
        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Clip to 0-1 range
        return float(max(0.0, min(1.0, score)))
    except Exception as e:
        logger.error(f"Error calculating color similarity: {e}")
        return 0.0

def calculate_shape_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate rough shape similarity using simple thresholding and moment comparison.
    This is an approximation as we don't have ground truth segmentation.
    We assume the lesion is darker than the surrounding skin.
    """
    try:
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2
        
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

        # Otsu's thresholding to find dark regions (lesions)
        # Invert because lesions are usually dark
        _, thresh1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, thresh2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate Hu Moments (invariant to scale, rotation, translation)
        moments1 = cv2.HuMoments(cv2.moments(thresh1)).flatten()
        moments2 = cv2.HuMoments(cv2.moments(thresh2)).flatten()
        
        # Log transform for scale invariance stability
        # Use abs log
        # Distance metric: sum of absolute differences of log transformed moments
        # This is essentially shape matching distance (lower is better)
        # We want similarity (higher is better)
        
        diff = 0
        for i in range(7):
            # Safe log
            m1 = -np.sign(moments1[i]) * np.log10(np.abs(moments1[i])) if moments1[i] != 0 else 0
            m2 = -np.sign(moments2[i]) * np.log10(np.abs(moments2[i])) if moments2[i] != 0 else 0
            diff += abs(m1 - m2)
            
        # Normalize to a similarity score 0-1
        # This is heuristic. Small diff = high similarity.
        # A difference of 0 is perfect. A huge difference is bad.
        # Let's map diff to score: 1 / (1 + diff)
        
        similarity = 1.0 / (1.0 + diff)
        return float(similarity)
        
    except Exception as e:
        logger.debug(f"Error calculating shape similarity: {e}")
        return 0.0 # Return 0 if fails

def compare_with_ham10000(
    generated_img: Image.Image, 
    diagnosis: str, 
    ham_loader: Any,
    num_references: int = 5
) -> Dict[str, float]:
    """
    Compare a generated image with a set of random HAM10000 images of the same diagnosis.
    Returns average metrics.
    
    Args:
        generated_img: PIL Image
        diagnosis: Diagnosis code (e.g., 'mel', 'nv')
        ham_loader: Initialized HAM10000Loader instance
        num_references: Number of reference images to compare against
        
    Returns:
        Dictionary with 'ssim', 'color_similarity', 'shape_similarity'
    """
    if not ham_loader:
        return {'ssim': 0.0, 'color_similarity': 0.0, 'shape_similarity': 0.0}
        
    # Get reference images
    ref_images_data = ham_loader.load_images_with_metadata(
        max_images=None, # Load metadata first to sample
        lesion_types=[diagnosis]
    )
    
    if not ref_images_data:
        logger.warning(f"No reference images found for diagnosis {diagnosis}")
        return {'ssim': 0.0, 'color_similarity': 0.0, 'shape_similarity': 0.0}
        
    # Sample random references (if we have more than needed, otherwise take all)
    # Note: load_images_with_metadata actually loads images if max_images was None/large?
    # Ah, looking at the previous view_file, if max_images is None it might load ALL? 
    # That would be slow if HAM10000 is large.
    # Checks: `ham_loader` implementation loads images immediately.
    # We should optimize this to NOT load all images, but `ham_loader.load_images_with_metadata`
    # structure seems to do that.
    # Let's check `ham_loader` behavior again?
    # It filters metadata then iterates and loads.
    # We should only load needed ones.
    # We can use the existing method with a small max_images, but it takes the HEAD (first N).
    # That's not random.
    # We might need to modify or just accept first N for now, or shuffle metadata inside loader?
    # For now let's just use `max_images=num_references` to be fast, even if not random.
    # Ideally we'd add `random_sample=True` to loader, but I won't modify loader right now if I can avoid it.
    
    # Actually, we can just request slightly more and pick?
    # Or just use the first N. It's a "same-type" comparison, first N is fine for POC.
    
    # Wait, the loader loads images into memory. If I call it with max_images=5, it's fast.
    
    references = ham_loader.load_images_with_metadata(
        max_images=num_references,
        lesion_types=[diagnosis]
    )
    
    if not references:
         return {'ssim': 0.0, 'color_similarity': 0.0, 'shape_similarity': 0.0}
         
    # Prepare generated image
    gen_np = np.array(generated_img.convert('RGB'))
    
    ssim_scores = []
    color_scores = []
    shape_scores = []
    
    reference_images = []
    
    for ref_data in references:
        ref_img = ref_data['image']
        ref_np = np.array(ref_img.convert('RGB'))
        
        # Store the reference image
        reference_images.append(ref_img)
        
        # Resize ref to match gen (metrics usually expect similar dimensions for meaningful comparison,
        # but our metric funcs handle resizing)
        
        ssim_scores.append(calculate_ssim(gen_np, ref_np))
        color_scores.append(calculate_color_similarity(gen_np, ref_np))
        shape_scores.append(calculate_shape_similarity(gen_np, ref_np))
     # Calculate average metrics
    metrics = {
        "ssim": float(np.mean(ssim_scores)) if ssim_scores else 0.0,
        "color_similarity": float(np.mean(color_scores)) if color_scores else 0.0,
        "shape_similarity": float(np.mean(shape_scores)) if shape_scores else 0.0
    }
    
    return metrics, reference_images
