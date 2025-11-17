"""Model storage utilities for saving and loading retrained models."""

import torch
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ModelStorage:
    """Utility class for storing and loading retrained models with metadata."""
    
    def __init__(self, base_path: str = "data/06_models"):
        """
        Initialize model storage.
        
        Args:
            base_path: Base directory for storing models
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model: torch.nn.Module, model_name: str, 
                  metrics: Dict[str, Any], training_params: Dict[str, Any],
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a retrained model with metadata.
        
        Args:
            model: PyTorch model to save
            model_name: Name of the model (e.g., 'resnet50')
            metrics: Training/evaluation metrics
            training_params: Training parameters used
            metadata: Additional metadata (optional)
            
        Returns:
            Path to saved model file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename with metrics summary
        metrics_str = "_".join([f"{k}_{v:.4f}" for k, v in metrics.items() 
                               if isinstance(v, (int, float))][:3])
        filename = f"{model_name}_retrained_{timestamp}_{metrics_str}.pth"
        model_path = self.base_path / filename
        
        # Save model state dict
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata_file = model_path.with_suffix('.json')
        metadata_data = {
            "model_name": model_name,
            "timestamp": timestamp,
            "metrics": metrics,
            "training_params": training_params,
            **metadata
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_data, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_file}")
        
        return str(model_path)
    
    def load_model_weights(self, model_path: str, 
                          model: torch.nn.Module) -> torch.nn.Module:
        """
        Load model weights from file.
        
        Args:
            model_path: Path to model weights file
            model: Model instance to load weights into
            
        Returns:
            Model with loaded weights
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        logger.info(f"Loaded model weights from {model_path}")
        
        return model
    
    def load_metadata(self, model_path: str) -> Dict[str, Any]:
        """
        Load metadata for a saved model.
        
        Args:
            model_path: Path to model weights file
            
        Returns:
            Metadata dictionary
        """
        metadata_path = Path(model_path).with_suffix('.json')
        if not metadata_path.exists():
            logger.warning(f"Metadata file not found: {metadata_path}")
            return {}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def list_models(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all saved models with their metadata.
        
        Args:
            model_name: Filter by model name (optional)
            
        Returns:
            List of model info dictionaries
        """
        models_info = []
        
        for model_file in self.base_path.glob("*.pth"):
            if model_name and model_name not in model_file.name:
                continue
            
            metadata = self.load_metadata(str(model_file))
            models_info.append({
                "path": str(model_file),
                "name": model_file.name,
                **metadata
            })
        
        return models_info

