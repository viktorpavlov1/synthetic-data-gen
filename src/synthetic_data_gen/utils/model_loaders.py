"""Model loaders for ENHANCE classification models."""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ENHANCEModelLoader:
    """Loader for ENHANCE classification models (VGG-16, Inception v3, ResNet50)."""
    
    def __init__(self, num_classes: int = 2):
        """
        Initialize the model loader.
        
        Args:
            num_classes: Number of classification classes (default: 2 for binary)
        """
        self.num_classes = num_classes
    
    def load_vgg16(self, pretrained: bool = True, weights_path: Optional[str] = None) -> nn.Module:
        """
        Load VGG-16 model adapted for skin lesion classification.
        
        Args:
            pretrained: Whether to use ImageNet pretrained weights
            weights_path: Path to custom weights file (optional)
            
        Returns:
            VGG-16 model with modified classifier head
        """
        model = models.vgg16(pretrained=pretrained)
        
        # Modify classifier for binary/multi-class classification
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, self.num_classes)
        
        if weights_path:
            try:
                state_dict = torch.load(weights_path, map_location='cpu')
                model.load_state_dict(state_dict)
                logger.info(f"Loaded weights from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load weights from {weights_path}: {e}")
        
        return model
    
    def load_inception_v3(self, pretrained: bool = True, weights_path: Optional[str] = None) -> nn.Module:
        """
        Load Inception v3 model adapted for skin lesion classification.
        
        Args:
            pretrained: Whether to use ImageNet pretrained weights
            weights_path: Path to custom weights file (optional)
            
        Returns:
            Inception v3 model with modified classifier head
        """
        model = models.inception_v3(pretrained=pretrained, aux_logits=False)
        
        # Modify classifier for binary/multi-class classification
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)
        
        if weights_path:
            try:
                state_dict = torch.load(weights_path, map_location='cpu')
                model.load_state_dict(state_dict)
                logger.info(f"Loaded weights from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load weights from {weights_path}: {e}")
        
        return model
    
    def load_resnet50(self, pretrained: bool = True, weights_path: Optional[str] = None) -> nn.Module:
        """
        Load ResNet50 model adapted for skin lesion classification.
        
        Args:
            pretrained: Whether to use ImageNet pretrained weights
            weights_path: Path to custom weights file (optional)
            
        Returns:
            ResNet50 model with modified classifier head
        """
        model = models.resnet50(pretrained=pretrained)
        
        # Modify classifier for binary/multi-class classification
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)
        
        if weights_path:
            try:
                state_dict = torch.load(weights_path, map_location='cpu')
                model.load_state_dict(state_dict)
                logger.info(f"Loaded weights from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load weights from {weights_path}: {e}")
        
        return model
    
    def load_model(self, model_name: str, pretrained: bool = True, 
                   weights_path: Optional[str] = None) -> nn.Module:
        """
        Load a model by name.
        
        Args:
            model_name: Name of the model ('vgg16', 'inception_v3', or 'resnet50')
            pretrained: Whether to use ImageNet pretrained weights
            weights_path: Path to custom weights file (optional)
            
        Returns:
            Loaded model
            
        Raises:
            ValueError: If model_name is not recognized
        """
        model_name = model_name.lower()
        
        if model_name == 'vgg16':
            return self.load_vgg16(pretrained=pretrained, weights_path=weights_path)
        elif model_name == 'inception_v3':
            return self.load_inception_v3(pretrained=pretrained, weights_path=weights_path)
        elif model_name == 'resnet50':
            return self.load_resnet50(pretrained=pretrained, weights_path=weights_path)
        else:
            raise ValueError(f"Unknown model name: {model_name}. "
                           f"Supported models: vgg16, inception_v3, resnet50")

