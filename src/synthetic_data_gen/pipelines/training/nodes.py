"""Training nodes for fine-tuning classification models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from tqdm import tqdm
import json

from synthetic_data_gen.utils.model_loaders import ENHANCEModelLoader
from synthetic_data_gen.utils.image_processors import get_preprocessing_transform
from synthetic_data_gen.utils.model_storage import ModelStorage

logger = logging.getLogger(__name__)


class SyntheticImageDataset(Dataset):
    """Dataset for synthetic images with optional labels."""
    
    def __init__(self, images: List[Dict[str, Any]], transform=None, 
                 assign_labels: bool = True):
        """
        Initialize dataset.
        
        Args:
            images: List of image dictionaries
            transform: Image transformation pipeline
            assign_labels: Whether to assign pseudo-labels based on prompts
        """
        self.images = images
        self.transform = transform
        self.assign_labels = assign_labels
        
        # Simple label assignment based on prompt keywords
        if assign_labels:
            self.labels = self._assign_labels()
        else:
            self.labels = [0] * len(images)
    
    def _assign_labels(self) -> List[int]:
        """Assign pseudo-labels based on prompt content."""
        labels = []
        for item in self.images:
            prompt = item.get("prompt", "").lower()
            # Simple heuristic: "benign" -> 0, "melanoma" or "malignant" -> 1
            if "benign" in prompt or "normal" in prompt:
                labels.append(0)
            elif "melanoma" in prompt or "malignant" in prompt or "cancer" in prompt:
                labels.append(1)
            else:
                labels.append(0)  # Default to benign
        return labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        item = self.images[idx]
        image = item["image"]
        
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image)).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


def prepare_training_data(
    generated_images: List[Dict[str, Any]],
    original_data_path: Optional[str] = None,
    use_synthetic: bool = True,
    synthetic_ratio: float = 0.3
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Prepare training, validation, and test datasets.
    
    Args:
        generated_images: List of generated synthetic images
        original_data_path: Path to original ENHANCE dataset (optional)
        use_synthetic: Whether to include synthetic data
        synthetic_ratio: Ratio of synthetic to real data
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    logger.info("Preparing training data...")
    
    # For now, we'll use only synthetic data
    # In a full implementation, you would load original ENHANCE data here
    all_images = generated_images if use_synthetic else []
    
    if original_data_path:
        logger.info(f"Loading original data from {original_data_path}")
        # TODO: Implement loading of original ENHANCE dataset
        # This would involve loading images and labels from the ENHANCE dataset
    
    if not all_images:
        raise ValueError("No images available for training")
    
    # Create dataset
    transform = get_preprocessing_transform(image_size=224, normalize=True)
    dataset = SyntheticImageDataset(all_images, transform=transform)
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Dataset split - Train: {len(train_dataset)}, "
                f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def train_model(
    model_name: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    num_classes: int = 2,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_epochs: int = 10,
    device: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train a classification model.
    
    Args:
        model_name: Name of the model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        num_classes: Number of classes
        batch_size: Batch size
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Training {model_name} on {device}")
    
    # Load model
    loader = ENHANCEModelLoader(num_classes=num_classes)
    model = loader.load_model(model_name, pretrained=True)
    model = model.to(device)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        scheduler.step()
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Load best model
    model.load_state_dict(best_model_state)
    logger.info(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, history


def evaluate_model(
    model: nn.Module,
    test_dataset: Dataset,
    batch_size: int = 32,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test dataset.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        batch_size: Batch size
        device: Device to use
        
    Returns:
        Dictionary with evaluation metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Evaluating model on test set...")
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    
    # Calculate additional metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds).tolist()
    
    metrics = {
        "test_loss": test_loss,
        "test_accuracy": accuracy,
        "test_precision": precision * 100,
        "test_recall": recall * 100,
        "test_f1": f1 * 100,
        "confusion_matrix": cm
    }
    
    logger.info(f"Test Accuracy: {accuracy:.2f}%")
    logger.info(f"Test Precision: {precision*100:.2f}%")
    logger.info(f"Test Recall: {recall*100:.2f}%")
    logger.info(f"Test F1: {f1*100:.2f}%")
    
    return metrics


def save_trained_model(
    model: nn.Module,
    model_name: str,
    evaluation_metrics: Dict[str, Any],
    training_history: Dict[str, Any],
    training_params: Dict[str, Any],
    base_path: str = "data/06_models"
) -> str:
    """
    Save a trained model with metadata.
    
    Args:
        model: Trained model
        model_name: Name of the model
        evaluation_metrics: Evaluation metrics
        training_history: Training history
        training_params: Training parameters
        base_path: Base path for model storage
        
    Returns:
        Path to saved model file
    """
    storage = ModelStorage(base_path=base_path)
    
    # Prepare metrics for filename
    metrics_summary = {
        "test_accuracy": evaluation_metrics.get("test_accuracy", 0.0),
        "test_f1": evaluation_metrics.get("test_f1", 0.0),
        "test_precision": evaluation_metrics.get("test_precision", 0.0),
    }
    
    # Prepare metadata
    metadata = {
        "training_history": training_history,
        "evaluation_metrics": evaluation_metrics,
    }
    
    model_path = storage.save_model(
        model=model,
        model_name=model_name,
        metrics=metrics_summary,
        training_params=training_params,
        metadata=metadata
    )
    
    logger.info(f"Saved trained model to {model_path}")
    return model_path

