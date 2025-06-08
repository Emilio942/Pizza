#!/usr/bin/env python3
"""
Pizza Verifier - Quality assessment for pizza recognition results.

This module implements a neural network-based verifier that predicts the quality
of pizza recognition results for adaptive inference strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass

from ..constants import CLASS_NAMES
from ..pizza_detector import MicroPizzaNet, MicroPizzaNetV2, MicroPizzaNetWithSE

logger = logging.getLogger(__name__)


@dataclass
class VerifierData:
    """Data structure for pizza verifier input."""
    pizza_image_path: str
    model_prediction: str
    ground_truth_class: str
    confidence_score: float
    quality_score: float
    
    # Additional features
    model_variant: Optional[str] = None
    processing_intensity: Optional[float] = None
    temporal_consistency: Optional[float] = None
    energy_cost: Optional[float] = None
    latency: Optional[float] = None


class PizzaVerifierNet(nn.Module):
    """
    Neural network for predicting pizza recognition quality.
    
    This network takes various features about a pizza recognition result
    and predicts a quality score from 0.0 to 1.0.
    """
    
    def __init__(
        self,
        input_dim: int = 32,
        hidden_dims: List[int] = [64, 32, 16],
        dropout_rate: float = 0.2,
        num_classes: int = 6
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Feature encoding layers
        self.class_embedding = nn.Embedding(num_classes, 8)
        self.model_embedding = nn.Embedding(3, 4)  # 3 model variants
        
        # Main network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Quality score in [0, 1]
        
        self.network = nn.Sequential(*layers)
        
        # Food safety classifier (additional task)
        self.safety_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # Safe/Unsafe
            nn.Softmax(dim=-1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the verifier network.
        
        Args:
            features: Input feature tensor [batch_size, input_dim]
            
        Returns:
            Tuple of (quality_scores, safety_predictions)
        """
        # Extract hidden features
        hidden = features
        for layer in self.network[:-2]:  # All except final linear + sigmoid
            hidden = layer(hidden)
        
        # Quality prediction
        quality_logits = self.network[-2](hidden)  # Final linear layer
        quality_scores = torch.sigmoid(quality_logits)
        
        # Safety prediction
        safety_logits = self.safety_head(hidden)
        
        return quality_scores, safety_logits
    
    def predict_quality(self, features: torch.Tensor) -> torch.Tensor:
        """Predict quality scores only."""
        with torch.no_grad():
            quality_scores, _ = self.forward(features)
            return quality_scores


class PizzaVerifier:
    """
    Complete pizza recognition quality verifier system.
    
    This class handles feature extraction, model inference, and quality prediction
    for pizza recognition results.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cpu',
        num_classes: int = 6
    ):
        self.device = device
        self.num_classes = num_classes
        
        # Initialize verifier network
        self.verifier_net = PizzaVerifierNet(num_classes=num_classes).to(device)
        self.verifier_net.eval()  # Set to evaluation mode to avoid BatchNorm issues
        
        # Load pre-trained weights if available
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.warning(f"Verifier model not found at {model_path}, using random initialization")
        
        # Feature statistics for normalization
        self.feature_stats = {
            'mean': torch.zeros(32),
            'std': torch.ones(32)
        }
        
        # Class name to index mapping
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        
        logger.info(f"Pizza verifier initialized on {device}")
    
    def extract_features(self, data: Union[VerifierData, Dict[str, Any]]) -> torch.Tensor:
        """
        Extract features from verifier input data.
        
        Args:
            data: Verifier input data
            
        Returns:
            Feature tensor ready for network input
        """
        if isinstance(data, dict):
            data = VerifierData(**data)
        
        features = []
        
        # Basic recognition features
        pred_idx = self.class_to_idx.get(data.model_prediction, 0)
        gt_idx = self.class_to_idx.get(data.ground_truth_class, 0)
        
        features.extend([
            pred_idx / self.num_classes,  # Normalized prediction class
            gt_idx / self.num_classes,    # Normalized ground truth class
            data.confidence_score,        # Model confidence
            float(pred_idx == gt_idx),    # Prediction correctness
        ])
        
        # Class-specific features
        class_one_hot = [0.0] * self.num_classes
        class_one_hot[pred_idx] = 1.0
        features.extend(class_one_hot)
        
        # Model variant features
        model_variant_features = [0.0, 0.0, 0.0]  # [MicroPizzaNet, V2, SE]
        if data.model_variant:
            if 'v2' in data.model_variant.lower():
                model_variant_features[1] = 1.0
            elif 'se' in data.model_variant.lower():
                model_variant_features[2] = 1.0
            else:
                model_variant_features[0] = 1.0
        features.extend(model_variant_features)
        
        # Processing intensity
        features.append(data.processing_intensity or 1.0)
        
        # Temporal consistency
        features.append(data.temporal_consistency or 0.5)
        
        # Performance metrics
        features.extend([
            (data.energy_cost or 10.0) / 20.0,  # Normalized energy cost
            (data.latency or 50.0) / 100.0,     # Normalized latency
        ])
        
        # Food safety critical features
        is_food_safety_critical = float(
            'burnt' in data.model_prediction.lower() or 
            'raw' in data.model_prediction.lower() or
            pred_idx != gt_idx  # Any misclassification is potentially unsafe
        )
        features.append(is_food_safety_critical)
        
        # Cross-class confusion patterns
        confusion_features = self._extract_confusion_features(pred_idx, gt_idx)
        features.extend(confusion_features)
        
        # Pad or truncate to input_dim
        while len(features) < 32:
            features.append(0.0)
        features = features[:32]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def _extract_confusion_features(self, pred_idx: int, gt_idx: int) -> List[float]:
        """Extract features related to common confusion patterns."""
        features = []
        
        # Common confusion pairs for pizza recognition
        confusion_pairs = [
            (0, 1),  # basic vs burnt
            (2, 3),  # combined vs mixed
            (4, 5),  # progression vs segment
        ]
        
        for pair in confusion_pairs:
            # Check if this is a known confusion pattern
            is_confusion = float(
                (pred_idx == pair[0] and gt_idx == pair[1]) or
                (pred_idx == pair[1] and gt_idx == pair[0])
            )
            features.append(is_confusion)
        
        return features
    
    def predict_quality(self, data: Union[VerifierData, Dict[str, Any]]) -> float:
        """
        Predict quality score for pizza recognition result.
        
        Args:
            data: Recognition result data
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        features = self.extract_features(data).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            quality_score = self.verifier_net.predict_quality(features)
            return quality_score.item()
    
    def predict_batch(self, data_list: List[Union[VerifierData, Dict[str, Any]]]) -> List[float]:
        """Predict quality scores for a batch of recognition results."""
        if not data_list:
            return []
        
        # Extract features for all data points
        features = torch.stack([self.extract_features(data) for data in data_list])
        
        with torch.no_grad():
            quality_scores = self.verifier_net.predict_quality(features)
            return quality_scores.squeeze().tolist()
    
    def predict_with_safety(
        self, 
        data: Union[VerifierData, Dict[str, Any]]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Predict both quality and food safety assessment.
        
        Returns:
            Tuple of (quality_score, safety_prediction)
        """
        features = self.extract_features(data).unsqueeze(0)
        
        with torch.no_grad():
            quality_scores, safety_logits = self.verifier_net(features)
            safety_probs = F.softmax(safety_logits, dim=-1)
            
            safety_prediction = {
                'safe_prob': safety_probs[0, 0].item(),
                'unsafe_prob': safety_probs[0, 1].item(),
                'is_safe': safety_probs[0, 0].item() > 0.5
            }
            
            return quality_scores.item(), safety_prediction
    
    def train_step(
        self,
        batch_data: List[VerifierData],
        optimizer: torch.optim.Optimizer,
        quality_weight: float = 1.0,
        safety_weight: float = 0.5
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch_data: Batch of training data
            optimizer: Optimizer for training
            quality_weight: Weight for quality loss
            safety_weight: Weight for safety loss
            
        Returns:
            Dictionary of training metrics
        """
        self.verifier_net.train()
        
        # Extract features and targets
        features = torch.stack([self.extract_features(data) for data in batch_data])
        quality_targets = torch.tensor([data.quality_score for data in batch_data], 
                                     dtype=torch.float32, device=self.device)
        
        # Safety targets (1 if quality < 0.5, 0 otherwise)
        safety_targets = torch.tensor([int(data.quality_score < 0.5) for data in batch_data],
                                    dtype=torch.long, device=self.device)
        
        # Forward pass
        quality_pred, safety_logits = self.verifier_net(features)
        
        # Compute losses
        quality_loss = F.mse_loss(quality_pred.squeeze(), quality_targets)
        safety_loss = F.cross_entropy(safety_logits, safety_targets)
        
        # Combined loss
        total_loss = quality_weight * quality_loss + safety_weight * safety_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'quality_loss': quality_loss.item(),
            'safety_loss': safety_loss.item(),
            'quality_mae': F.l1_loss(quality_pred.squeeze(), quality_targets).item()
        }
    
    def evaluate(self, test_data: List[VerifierData]) -> Dict[str, float]:
        """Evaluate verifier on test data."""
        self.verifier_net.eval()
        
        if not test_data:
            return {}
        
        with torch.no_grad():
            features = torch.stack([self.extract_features(data) for data in test_data])
            quality_targets = torch.tensor([data.quality_score for data in test_data],
                                         dtype=torch.float32, device=self.device)
            
            quality_pred, safety_logits = self.verifier_net(features)
            
            # Quality metrics
            quality_mse = F.mse_loss(quality_pred.squeeze(), quality_targets).item()
            quality_mae = F.l1_loss(quality_pred.squeeze(), quality_targets).item()
            
            # Correlation
            pred_np = quality_pred.squeeze().cpu().numpy()
            target_np = quality_targets.cpu().numpy()
            correlation = np.corrcoef(pred_np, target_np)[0, 1]
            
            # Safety metrics
            safety_targets = torch.tensor([int(data.quality_score < 0.5) for data in test_data],
                                        dtype=torch.long, device=self.device)
            safety_pred = torch.argmax(safety_logits, dim=-1)
            safety_accuracy = (safety_pred == safety_targets).float().mean().item()
            
            return {
                'quality_mse': quality_mse,
                'quality_mae': quality_mae,
                'quality_correlation': correlation if not np.isnan(correlation) else 0.0,
                'safety_accuracy': safety_accuracy
            }
    
    def save_model(self, path: str):
        """Save the verifier model."""
        save_data = {
            'model_state_dict': self.verifier_net.state_dict(),
            'feature_stats': self.feature_stats,
            'class_to_idx': self.class_to_idx,
            'num_classes': self.num_classes
        }
        torch.save(save_data, path)
        logger.info(f"Verifier model saved to {path}")
    
    def load_model(self, path: str):
        """Load a saved verifier model."""
        save_data = torch.load(path, map_location=self.device)
        self.verifier_net.load_state_dict(save_data['model_state_dict'])
        self.verifier_net.eval()  # Ensure eval mode after loading
        self.feature_stats = save_data.get('feature_stats', self.feature_stats)
        self.class_to_idx = save_data.get('class_to_idx', self.class_to_idx)
        self.num_classes = save_data.get('num_classes', self.num_classes)
        logger.info(f"Verifier model loaded from {path}")
    
    def update_feature_stats(self, training_data: List[VerifierData]):
        """Update feature normalization statistics from training data."""
        if not training_data:
            return
        
        features = torch.stack([self.extract_features(data) for data in training_data])
        self.feature_stats['mean'] = features.mean(dim=0)
        self.feature_stats['std'] = features.std(dim=0) + 1e-8  # Avoid division by zero
        
        logger.info("Feature normalization statistics updated")


def load_verifier_data(json_path: str) -> List[VerifierData]:
    """
    Load verifier data from JSON file.
    
    Args:
        json_path: Path to JSON file containing verifier data
        
    Returns:
        List of VerifierData objects
    """
    data_list = []
    
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        # Handle both single objects and lists
        if isinstance(json_data, list):
            items = json_data
        else:
            items = [json_data]
        
        for item in items:
            data = VerifierData(
                pizza_image_path=item.get('pizza_image_path', ''),
                model_prediction=item.get('model_prediction', 'basic'),
                ground_truth_class=item.get('ground_truth_class', 'basic'),
                confidence_score=float(item.get('confidence_score', 0.5)),
                quality_score=float(item.get('quality_score', 0.5)),
                model_variant=item.get('model_variant'),
                processing_intensity=item.get('processing_intensity'),
                temporal_consistency=item.get('temporal_consistency'),
                energy_cost=item.get('energy_cost'),
                latency=item.get('latency')
            )
            data_list.append(data)
        
        logger.info(f"Loaded {len(data_list)} verifier data items from {json_path}")
        
    except Exception as e:
        logger.error(f"Error loading verifier data from {json_path}: {e}")
    
    return data_list


def save_verifier_data(data_list: List[VerifierData], json_path: str):
    """Save verifier data to JSON file."""
    json_data = []
    
    for data in data_list:
        item = {
            'pizza_image_path': data.pizza_image_path,
            'model_prediction': data.model_prediction,
            'ground_truth_class': data.ground_truth_class,
            'confidence_score': data.confidence_score,
            'quality_score': data.quality_score
        }
        
        # Add optional fields if present
        if data.model_variant:
            item['model_variant'] = data.model_variant
        if data.processing_intensity is not None:
            item['processing_intensity'] = data.processing_intensity
        if data.temporal_consistency is not None:
            item['temporal_consistency'] = data.temporal_consistency
        if data.energy_cost is not None:
            item['energy_cost'] = data.energy_cost
        if data.latency is not None:
            item['latency'] = data.latency
        
        json_data.append(item)
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    logger.info(f"Saved {len(data_list)} verifier data items to {json_path}")
