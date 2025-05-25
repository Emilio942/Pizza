#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for the pizza detector model
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

from src.constants import IMAGE_MEAN, IMAGE_STD, INPUT_SIZE


def load_model(model_path: str, device: Optional[torch.device] = None) -> torch.nn.Module:
    """
    Load a pizza detection model from a path
    
    Args:
        model_path: Path to the model file
        device: Device to load the model on (cpu or cuda)
        
    Returns:
        The loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load model
        if Path(model_path).exists():
            model = torch.jit.load(model_path, map_location=device)
            model.eval()
            return model
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def preprocess_image(image_path: Union[str, Path], img_size: int = INPUT_SIZE) -> torch.Tensor:
    """
    Preprocess an image for inference
    
    Args:
        image_path: Path to the image file
        img_size: Size to resize the image to
        
    Returns:
        Preprocessed image tensor
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Define transformation
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ])
    
    # Transform image
    img_tensor = transform(img)
    
    return img_tensor


def get_prediction(model: torch.nn.Module, img_input: Union[str, torch.Tensor], 
                  img_size: int = INPUT_SIZE, class_names: List[str] = None) -> Tuple[str, float]:
    """
    Get prediction for an image
    
    Args:
        model: The model to use for prediction
        img_input: Path to the image or preprocessed image tensor
        img_size: Size to resize the image to if img_input is a path
        class_names: List of class names
        
    Returns:
        Tuple of (predicted class name, probability)
    """
    # Set device
    device = next(model.parameters()).device
    
    # Preprocess image if necessary
    if isinstance(img_input, (str, Path)):
        img_tensor = preprocess_image(img_input, img_size)
    else:
        img_tensor = img_input
    
    # Add batch dimension and move to device
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    
    # Convert tensors to Python values
    predicted_idx = predicted.item()
    confidence_val = confidence.item()
    
    # Get class name
    if class_names is not None:
        predicted_class = class_names[predicted_idx]
    else:
        predicted_class = str(predicted_idx)
    
    return predicted_class, confidence_val


def evaluate_model(model, data_loader, device=None, class_names=None):
    """
    Evaluate model accuracy and metrics on a dataset
    
    Args:
        model: The PyTorch model to evaluate
        data_loader: DataLoader containing the evaluation dataset
        device: Device to run evaluation on (cpu or cuda)
        class_names: List of class names (optional)
        
    Returns:
        Dictionary with evaluation metrics
    """
    import torch
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    model.to(device)
    
    # Initialize variables
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = 100. * correct / total
    
    # Calculate precision, recall, F1-score
    if len(np.unique(all_labels)) > 1:  # Make sure there's more than one class
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
    else:
        precision, recall, f1 = 0, 0, 0
        cm = np.zeros((1, 1))
    
    # Create evaluation report
    report = {
        'accuracy': accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'confusion_matrix': cm.tolist(),
        'num_samples': total
    }
    
    print(f'Evaluation Results:')
    print(f'  Accuracy: {accuracy:.2f}%')
    print(f'  Precision: {precision * 100:.2f}%')
    print(f'  Recall: {recall * 100:.2f}%')
    print(f'  F1 Score: {f1 * 100:.2f}%')
    
    return report


class RP2040Config:
    """Simulation of RP2040 configuration settings"""
    # RP2040 Hardware-Spezifikationen
    RP2040_FLASH_SIZE_KB = 2048  # 2MB Flash
    RP2040_RAM_SIZE_KB = 264     # 264KB RAM
    RP2040_CLOCK_SPEED_MHZ = 133 # 133MHz Dual-Core Arm Cortex M0+
    
    # OV2640 Kamera-Spezifikationen
    CAMERA_WIDTH = 320
    CAMERA_HEIGHT = 240
    CAMERA_FPS = 7  # Durchschnittliche FPS für Batteriebetrieb
    
    # Batterieparameter (CR123A)
    BATTERY_CAPACITY_MAH = 1500   # Typische CR123A Kapazität
    ACTIVE_CURRENT_MA = 180       # Durchschnittlicher Stromverbrauch im aktiven Zustand
    SLEEP_CURRENT_MA = 0.5        # Stromverbrauch im Schlafmodus
    
    # Datensatz-Konfiguration
    DATA_DIR = 'augmented_pizza'
    MODEL_DIR = 'models_optimized'
    TEMP_DIR = 'temp_preprocessing'
    
    # Modellparameter
    IMG_SIZE = 48       # Kleine Bildgröße für Mikrocontroller
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 0.002
    EARLY_STOPPING_PATIENCE = 10
    
    # Speicheroptimierungen
    MAX_MODEL_SIZE_KB = 180       # Maximale Modellgröße (Flash)
    MAX_RUNTIME_RAM_KB = 100      # Maximaler RAM-Verbrauch während Inferenz
    QUANTIZATION_BITS = 8         # Int8-Quantisierung
    
    # Trainingsgerät
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self, data_dir=None):
        if data_dir:
            self.DATA_DIR = data_dir
