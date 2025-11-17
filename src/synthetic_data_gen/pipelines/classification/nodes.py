"""Classification nodes for skin lesion classification."""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any
import pandas as pd
import logging
from tqdm import tqdm

from synthetic_data_gen.utils.model_loaders import ENHANCEModelLoader
from synthetic_data_gen.utils.image_processors import preprocess_batch

logger = logging.getLogger(__name__)


def load_classification_model(model_name: str, num_classes: int = 2, 
                             weights_path: str = None) -> torch.nn.Module:
    """
    Load a classification model for inference.
    
    Args:
        model_name: Name of the model ('vgg16', 'inception_v3', or 'resnet50')
        num_classes: Number of classes
        weights_path: Path to model weights (optional)
        
    Returns:
        Loaded model in evaluation mode
    """
    loader = ENHANCEModelLoader(num_classes=num_classes)
    model = loader.load_model(model_name, pretrained=True, weights_path=weights_path)
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    logger.info(f"Loaded {model_name} model on {device}")
    return model


def classify_images(
    generated_images: List[Dict[str, Any]],
    model_name: str,
    batch_size: int = 32,
    num_classes: int = 2,
    weights_path: str = None,
    class_names: List[str] = None
) -> pd.DataFrame:
    """
    Classify generated images using a pre-trained model.
    
    Args:
        generated_images: List of generated image dictionaries
        model_name: Name of the classification model
        batch_size: Batch size for inference
        num_classes: Number of classes
        weights_path: Path to model weights (optional)
        class_names: Names of classes (e.g., ['benign', 'malignant'])
        
    Returns:
        DataFrame with classification results
    """
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    logger.info(f"Classifying {len(generated_images)} images using {model_name}")
    
    # Load model
    model = load_classification_model(model_name, num_classes, weights_path)
    device = next(model.parameters()).device
    
    # Extract images
    images = [item["image"] for item in generated_images]
    
    # Process in batches
    results = []
    
    for i in tqdm(range(0, len(images), batch_size), desc="Classifying images"):
        batch_images = images[i:i + batch_size]
        batch_metadata = generated_images[i:i + batch_size]
        
        try:
            # Preprocess batch
            batch_tensor = preprocess_batch(batch_images, image_size=224)
            batch_tensor = batch_tensor.to(device)
            
            # Run inference
            with torch.no_grad():
                outputs = model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                confidence_scores = torch.max(probabilities, dim=1)[0]
            
            # Store results
            for j, (pred, conf, prob) in enumerate(zip(
                predictions.cpu().numpy(),
                confidence_scores.cpu().numpy(),
                probabilities.cpu().numpy()
            )):
                metadata = batch_metadata[j]
                result = {
                    "image_index": i + j,
                    "predicted_class": int(pred),
                    "predicted_class_name": class_names[int(pred)],
                    "confidence": float(conf),
                    "generation_model": metadata.get("model", "unknown"),
                    "prompt": metadata.get("prompt", ""),
                }
                
                # Add probabilities for all classes
                for k, class_name in enumerate(class_names):
                    result[f"prob_{class_name}"] = float(prob[k])
                
                results.append(result)
                
        except Exception as e:
            logger.error(f"Error classifying batch starting at index {i}: {e}")
            # Add error entry
            for j in range(len(batch_images)):
                results.append({
                    "image_index": i + j,
                    "predicted_class": -1,
                    "predicted_class_name": "error",
                    "confidence": 0.0,
                    "generation_model": batch_metadata[j].get("model", "unknown"),
                    "prompt": batch_metadata[j].get("prompt", ""),
                    "error": str(e)
                })
    
    df_results = pd.DataFrame(results)
    logger.info(f"Classification complete. Results shape: {df_results.shape}")
    
    return df_results


def aggregate_classification_results(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Aggregate classification results for summary statistics.
    
    Args:
        results_df: DataFrame with classification results
        
    Returns:
        Dictionary with aggregated statistics
    """
    if results_df.empty:
        return {}
    
    summary = {
        "total_images": len(results_df),
        "successful_classifications": len(results_df[results_df["predicted_class"] >= 0]),
        "failed_classifications": len(results_df[results_df["predicted_class"] < 0]),
    }
    
    if summary["successful_classifications"] > 0:
        valid_results = results_df[results_df["predicted_class"] >= 0]
        summary["average_confidence"] = valid_results["confidence"].mean()
        summary["min_confidence"] = valid_results["confidence"].min()
        summary["max_confidence"] = valid_results["confidence"].max()
        
        # Class distribution
        if "predicted_class_name" in valid_results.columns:
            class_counts = valid_results["predicted_class_name"].value_counts().to_dict()
            summary["class_distribution"] = class_counts
        
        # Generation model distribution
        if "generation_model" in valid_results.columns:
            model_counts = valid_results["generation_model"].value_counts().to_dict()
            summary["generation_model_distribution"] = model_counts
    
    return summary

