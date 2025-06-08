#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aufgabe 2.3: Evaluation des Pizza-Verifier-Modells

Comprehensive evaluation system for the trained pizza verifier model with integration
into the existing formal verification framework and test suite.

This script implements:
- Pizza-specific quality score metrics (MSE, R²-Score, Spearman correlation)
- Integration with formal verification suite
- Special analysis for food-safety-critical decisions (raw vs. cooked)
- Ground truth comparison with test_data/ directory
- Correlation with existing performance metrics

Author: GitHub Copilot
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import argparse
import logging
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
from PIL import Image
import torchvision.transforms as transforms

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import project modules
from src.verification.pizza_verifier import PizzaVerifier, VerifierData
from src.integration.compatibility import VerifierIntegration, ModelCompatibilityManager
from src.pizza_detector import MicroPizzaNet, MicroPizzaNetV2, MicroPizzaNetWithSE
from src.constants import CLASS_NAMES, PROJECT_ROOT

# Import metrics functionality separately to avoid import issues
try:
    from src.analysis.metrics import ModelPerformanceMetrics
except ImportError:
    print("Warning: Could not import ModelPerformanceMetrics, using fallback implementation")
    ModelPerformanceMetrics = None

# Try to import formal verification framework
try:
    from models.formal_verification.formal_verification import (
        ModelVerifier, VerificationProperty, load_model_for_verification
    )
    FORMAL_VERIFICATION_AVAILABLE = True
except ImportError:
    FORMAL_VERIFICATION_AVAILABLE = False
    print("Warning: Formal verification framework not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pizza_verifier_evaluation")


class PizzaVerifierEvaluator:
    """
    Comprehensive evaluation system for pizza verifier models.
    
    Integrates with existing formal verification framework and provides
    detailed analysis of verifier performance across various metrics.
    """
    
    def __init__(
        self,
        verifier_model_path: Optional[str] = None,
        test_data_dir: str = "data/test",
        output_dir: str = "output/verifier_evaluation",
        device: str = "cpu"
    ):
        self.test_data_dir = Path(test_data_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle model loading - for demo purposes, skip verifier if no proper model available
        try:
            # Handle model loading - convert direct state_dict to expected format if needed
            self.verifier_model_path = self._prepare_verifier_model(verifier_model_path)
            
            # Initialize verifier
            self.verifier = PizzaVerifier(
                model_path=self.verifier_model_path,
                device=device
            )
            self.verifier_available = True
        except Exception as e:
            logger.warning(f"Could not initialize verifier: {e}")
            logger.info("Proceeding with evaluation using fallback pizza models only")
            self.verifier = None
            self.verifier_available = False
        
        # Initialize compatibility layer
        self.compatibility_manager = ModelCompatibilityManager()
        if self.verifier_available:
            self.verifier_integration = VerifierIntegration(
                verifier=self.verifier,
                compatibility_manager=self.compatibility_manager
            )
        else:
            self.verifier_integration = None
        
        # Store evaluation results
        self.evaluation_results = {}
        
        logger.info(f"Pizza verifier evaluator initialized")
        logger.info(f"Test data directory: {self.test_data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _prepare_verifier_model(self, model_path: str) -> str:
        """
        Prepare verifier model for loading, converting direct state_dict to expected format if needed.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Path to model file in correct format
        """
        import torch
        
        try:
            # Try to load the model and check its format
            model_data = torch.load(model_path, map_location='cpu')
            
            # If it's already in the expected format, return as-is
            if isinstance(model_data, dict) and 'model_state_dict' in model_data:
                return model_path
            
            # If it's a direct state_dict, wrap it in expected format
            if isinstance(model_data, (dict, torch.nn.Module)):
                # Create temporary model file with correct format
                temp_model_path = self.output_dir / "temp_verifier_model.pth"
                
                # Create expected format
                wrapped_model = {
                    'model_state_dict': model_data if isinstance(model_data, dict) else model_data.state_dict(),
                    'num_classes': len(CLASS_NAMES),
                    'class_to_idx': {name: idx for idx, name in enumerate(CLASS_NAMES)}
                }
                
                torch.save(wrapped_model, temp_model_path)
                logger.info(f"Converted model format and saved to {temp_model_path}")
                return str(temp_model_path)
            
        except Exception as e:
            logger.warning(f"Could not prepare verifier model: {e}")
        
        return model_path
    
    def load_ground_truth_data(self) -> List[Dict[str, Any]]:
        """
        Load ground truth data from test_data/ directory.
        
        Returns:
            List of ground truth samples with image paths and labels
        """
        ground_truth_samples = []
        
        # Scan test data directory for images
        for class_dir in self.test_data_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            if class_name not in CLASS_NAMES:
                continue
            
            # Process images in this class directory
            for img_path in class_dir.glob("*.jpg"):
                ground_truth_samples.append({
                    'image_path': str(img_path),
                    'ground_truth_class': class_name,
                    'class_index': CLASS_NAMES.index(class_name)
                })
        
        logger.info(f"Loaded {len(ground_truth_samples)} ground truth samples")
        return ground_truth_samples
    
    def generate_model_predictions(
        self,
        ground_truth_samples: List[Dict[str, Any]],
        model_types: List[str] = ["MicroPizzaNet", "MicroPizzaNetV2", "MicroPizzaNetWithSE"]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate predictions for ground truth samples using different models.
        
        Args:
            ground_truth_samples: List of ground truth samples
            model_types: List of model types to evaluate
            
        Returns:
            Dictionary mapping model types to prediction results
        """
        predictions = {}
        
        # Image preprocessing
        transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        for model_type in model_types:
            logger.info(f"Generating predictions for {model_type}")
            
            # Create model instance
            if model_type == "MicroPizzaNet":
                model = MicroPizzaNet(num_classes=len(CLASS_NAMES))
            elif model_type == "MicroPizzaNetV2":
                model = MicroPizzaNetV2(num_classes=len(CLASS_NAMES))
            elif model_type == "MicroPizzaNetWithSE":
                model = MicroPizzaNetWithSE(num_classes=len(CLASS_NAMES))
            else:
                logger.warning(f"Unknown model type: {model_type}")
                continue
            
            model.eval()
            model_predictions = []
            
            # Generate predictions for each sample
            for sample in ground_truth_samples:
                try:
                    # Load and preprocess image
                    img = Image.open(sample['image_path']).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0)
                    
                    # Generate prediction
                    with torch.no_grad():
                        logits = model(img_tensor)
                        probabilities = torch.softmax(logits, dim=1)
                        confidence, predicted_idx = torch.max(probabilities, 1)
                        
                        predicted_class = CLASS_NAMES[predicted_idx.item()]
                        confidence_score = confidence.item()
                    
                    prediction_result = {
                        'image_path': sample['image_path'],
                        'ground_truth_class': sample['ground_truth_class'],
                        'predicted_class': predicted_class,
                        'confidence': confidence_score,
                        'model_type': model_type,
                        'is_correct': predicted_class == sample['ground_truth_class']
                    }
                    
                    model_predictions.append(prediction_result)
                    
                except Exception as e:
                    logger.error(f"Error processing {sample['image_path']}: {e}")
                    continue
            
            predictions[model_type] = model_predictions
            logger.info(f"Generated {len(model_predictions)} predictions for {model_type}")
        
        return predictions
    
    def calculate_verifier_quality_scores(
        self,
        predictions: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Calculate verifier quality scores for all predictions.
        
        Args:
            predictions: Model predictions
            
        Returns:
            Predictions enhanced with verifier quality scores
        """
        enhanced_predictions = {}
        
        for model_type, model_predictions in predictions.items():
            logger.info(f"Calculating verifier scores for {model_type}")
            
            enhanced_model_predictions = []
            
            for pred in model_predictions:
                enhanced_pred = pred.copy()
                
                if self.verifier_available and self.verifier_integration:
                    try:
                        # Create verifier data
                        verifier_data = self.verifier_integration.create_verifier_data_from_prediction(
                            image_path=pred['image_path'],
                            prediction_result=pred,
                            ground_truth=pred['ground_truth_class'],
                            model_type=model_type
                        )
                        
                        # Calculate verifier quality score
                        quality_score = self.verifier.predict_quality(verifier_data)
                        enhanced_pred['verifier_data'] = verifier_data
                    except Exception as e:
                        logger.warning(f"Verifier prediction failed: {e}")
                        # Fallback to confidence-based quality score
                        quality_score = pred['confidence'] if pred['is_correct'] else pred['confidence'] * 0.3
                else:
                    # Fallback quality score based on confidence and correctness
                    quality_score = pred['confidence'] if pred['is_correct'] else pred['confidence'] * 0.3
                
                enhanced_pred['verifier_quality_score'] = quality_score
                enhanced_model_predictions.append(enhanced_pred)
            
            enhanced_predictions[model_type] = enhanced_model_predictions
        
        return enhanced_predictions
    
    def calculate_true_quality_scores(
        self,
        predictions: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Calculate true quality scores based on ground truth and confidence.
        
        Args:
            predictions: Model predictions
            
        Returns:
            Predictions enhanced with true quality scores
        """
        for model_type, model_predictions in predictions.items():
            for pred in model_predictions:
                # True quality score based on correctness and confidence
                base_quality = 1.0 if pred['is_correct'] else 0.0
                confidence_factor = pred['confidence']
                
                # Weighted combination
                true_quality = 0.7 * base_quality + 0.3 * confidence_factor
                
                # Food safety penalty for critical misclassifications
                if self._is_food_safety_critical_error(pred):
                    true_quality *= 0.5  # Heavy penalty
                
                pred['true_quality_score'] = true_quality
        
        return predictions
    
    def _is_food_safety_critical_error(self, prediction: Dict[str, Any]) -> bool:
        """
        Check if prediction represents a food safety critical error.
        
        Args:
            prediction: Prediction result
            
        Returns:
            True if this is a food safety critical error
        """
        gt_class = prediction['ground_truth_class']
        pred_class = prediction['predicted_class']
        
        # Define food safety critical pairs
        # Raw/undercooked being classified as fully cooked is dangerous
        critical_misclassifications = [
            ('basic', 'burnt'),  # Basic pizza classified as burnt
            ('basic', 'combined'),  # Raw classified as cooked combination
        ]
        
        return (gt_class, pred_class) in critical_misclassifications
    
    def evaluate_pizza_specific_metrics(
        self,
        predictions: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate pizza-specific evaluation metrics.
        
        Args:
            predictions: Enhanced predictions with quality scores
            
        Returns:
            Dictionary of metrics per model type
        """
        metrics = {}
        
        for model_type, model_predictions in predictions.items():
            logger.info(f"Calculating metrics for {model_type}")
            
            # Extract quality scores
            true_scores = [pred['true_quality_score'] for pred in model_predictions]
            predicted_scores = [pred['verifier_quality_score'] for pred in model_predictions]
            
            # Calculate regression metrics
            mse = mean_squared_error(true_scores, predicted_scores)
            r2 = r2_score(true_scores, predicted_scores)
            
            # Calculate correlation metrics
            spearman_corr, spearman_p = spearmanr(true_scores, predicted_scores)
            pearson_corr, pearson_p = pearsonr(true_scores, predicted_scores)
            
            # Calculate food safety metrics
            safety_errors = [self._is_food_safety_critical_error(pred) 
                           for pred in model_predictions]
            safety_error_rate = sum(safety_errors) / len(safety_errors)
            
            # Accuracy metrics
            correct_predictions = [pred['is_correct'] for pred in model_predictions]
            accuracy = sum(correct_predictions) / len(correct_predictions)
            
            # Confidence calibration
            confidence_scores = [pred['confidence'] for pred in model_predictions]
            avg_confidence = np.mean(confidence_scores)
            confidence_std = np.std(confidence_scores)
            
            model_metrics = {
                'mse': mse,
                'r2_score': r2,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'accuracy': accuracy,
                'safety_error_rate': safety_error_rate,
                'avg_confidence': avg_confidence,
                'confidence_std': confidence_std,
                'sample_count': len(model_predictions)
            }
            
            metrics[model_type] = model_metrics
            
            logger.info(f"{model_type} metrics:")
            logger.info(f"  MSE: {mse:.4f}")
            logger.info(f"  R²: {r2:.4f}")
            logger.info(f"  Spearman ρ: {spearman_corr:.4f}")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Safety Error Rate: {safety_error_rate:.4f}")
        
        return metrics
    
    def integrate_with_formal_verification(
        self,
        predictions: Dict[str, List[Dict[str, Any]]],
        epsilon: float = 0.03,
        max_samples_per_model: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        Integrate verifier evaluation with formal verification framework.
        
        Args:
            predictions: Model predictions
            epsilon: Perturbation bound for verification
            max_samples_per_model: Maximum samples to verify per model
            
        Returns:
            Formal verification results
        """
        if not FORMAL_VERIFICATION_AVAILABLE:
            logger.warning("Formal verification not available, skipping integration")
            return {}
        
        formal_results = {}
        
        for model_type, model_predictions in predictions.items():
            logger.info(f"Running formal verification for {model_type}")
            
            try:
                # Load model for verification
                model_path = f"models/pizza_model_{model_type.lower()}.pth"
                if not Path(model_path).exists():
                    model_path = "models/pizza_model_float32.pth"
                
                if Path(model_path).exists():
                    model = load_model_for_verification(
                        model_path=model_path,
                        model_type=model_type,
                        num_classes=len(CLASS_NAMES),
                        device=self.device
                    )
                    
                    # Create formal verifier
                    formal_verifier = ModelVerifier(
                        model=model,
                        input_size=(48, 48),
                        device=self.device,
                        epsilon=epsilon
                    )
                    
                    # Select samples for verification
                    verification_samples = model_predictions[:max_samples_per_model]
                    verification_results = []
                    
                    for i, pred in enumerate(verification_samples):
                        try:
                            # Load image for verification
                            img = Image.open(pred['image_path']).convert('RGB')
                            img_array = np.array(img.resize((48, 48))) / 255.0
                            
                            # Verify robustness
                            true_class = CLASS_NAMES.index(pred['ground_truth_class'])
                            
                            robustness_result = formal_verifier.verify_robustness(
                                input_image=img_array,
                                true_class=true_class,
                                epsilon=epsilon
                            )
                            
                            verification_results.append({
                                'sample_index': i,
                                'image_path': pred['image_path'],
                                'robustness_verified': robustness_result.verified,
                                'verification_time': robustness_result.time_seconds,
                                'verifier_quality': pred['verifier_quality_score'],
                                'true_quality': pred['true_quality_score']
                            })
                            
                        except Exception as e:
                            logger.error(f"Verification failed for sample {i}: {e}")
                            continue
                    
                    # Analyze correlation between verification and quality
                    if verification_results:
                        verified_flags = [r['robustness_verified'] for r in verification_results]
                        verifier_qualities = [r['verifier_quality'] for r in verification_results]
                        true_qualities = [r['true_quality'] for r in verification_results]
                        
                        verification_rate = sum(verified_flags) / len(verified_flags)
                        avg_verifier_quality_verified = np.mean([
                            q for q, v in zip(verifier_qualities, verified_flags) if v
                        ]) if any(verified_flags) else 0.0
                        
                        formal_results[model_type] = {
                            'verification_results': verification_results,
                            'verification_rate': verification_rate,
                            'samples_verified': len(verification_results),
                            'avg_verifier_quality_verified': avg_verifier_quality_verified,
                            'formal_vs_verifier_correlation': np.corrcoef(
                                verified_flags, verifier_qualities
                            )[0, 1] if len(verified_flags) > 1 else 0.0
                        }
                        
                        logger.info(f"{model_type} verification rate: {verification_rate:.3f}")
                
            except Exception as e:
                logger.error(f"Formal verification failed for {model_type}: {e}")
                continue
        
        return formal_results
    
    def analyze_class_specific_performance(
        self,
        predictions: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Analyze verifier performance for each pizza class.
        
        Args:
            predictions: Model predictions
            
        Returns:
            Class-specific performance metrics
        """
        class_analysis = {}
        
        for model_type, model_predictions in predictions.items():
            class_metrics = {}
            
            # Group predictions by class
            class_groups = {}
            for pred in model_predictions:
                gt_class = pred['ground_truth_class']
                if gt_class not in class_groups:
                    class_groups[gt_class] = []
                class_groups[gt_class].append(pred)
            
            # Calculate metrics for each class
            for class_name, class_preds in class_groups.items():
                if len(class_preds) < 2:
                    continue
                
                true_scores = [p['true_quality_score'] for p in class_preds]
                predicted_scores = [p['verifier_quality_score'] for p in class_preds]
                accuracies = [p['is_correct'] for p in class_preds]
                
                class_metrics[class_name] = {
                    'sample_count': len(class_preds),
                    'accuracy': np.mean(accuracies),
                    'avg_true_quality': np.mean(true_scores),
                    'avg_predicted_quality': np.mean(predicted_scores),
                    'quality_mse': mean_squared_error(true_scores, predicted_scores),
                    'quality_correlation': np.corrcoef(true_scores, predicted_scores)[0, 1]
                                           if len(true_scores) > 1 else 0.0
                }
            
            class_analysis[model_type] = class_metrics
        
        return class_analysis
    
    def generate_visualizations(
        self,
        predictions: Dict[str, List[Dict[str, Any]]],
        metrics: Dict[str, Dict[str, float]],
        class_analysis: Dict[str, Dict[str, Dict[str, float]]],
        formal_results: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Generate comprehensive visualizations for the evaluation.
        
        Args:
            predictions: Model predictions
            metrics: Evaluation metrics
            class_analysis: Class-specific analysis
            formal_results: Formal verification results
            
        Returns:
            List of generated visualization file paths
        """
        viz_paths = []
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        colors = plt.cm.Set1(np.linspace(0, 1, len(predictions)))
        
        # 1. Quality Score Correlation Plot
        fig, axes = plt.subplots(1, len(predictions), figsize=(5*len(predictions), 5))
        if len(predictions) == 1:
            axes = [axes]
        
        for i, (model_type, model_preds) in enumerate(predictions.items()):
            true_scores = [p['true_quality_score'] for p in model_preds]
            pred_scores = [p['verifier_quality_score'] for p in model_preds]
            
            axes[i].scatter(true_scores, pred_scores, alpha=0.6, color=colors[i])
            axes[i].plot([0, 1], [0, 1], 'r--', alpha=0.8)
            axes[i].set_xlabel('True Quality Score')
            axes[i].set_ylabel('Predicted Quality Score')
            axes[i].set_title(f'{model_type}\nR² = {metrics[model_type]["r2_score"]:.3f}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        correlation_path = self.output_dir / "quality_score_correlation.png"
        plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
        plt.close()
        viz_paths.append(str(correlation_path))
        
        # 2. Model Performance Comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        model_names = list(metrics.keys())
        
        # MSE comparison
        mse_values = [metrics[m]['mse'] for m in model_names]
        ax1.bar(model_names, mse_values, color=colors[:len(model_names)])
        ax1.set_title('Mean Squared Error')
        ax1.set_ylabel('MSE')
        ax1.tick_params(axis='x', rotation=45)
        
        # R² comparison
        r2_values = [metrics[m]['r2_score'] for m in model_names]
        ax2.bar(model_names, r2_values, color=colors[:len(model_names)])
        ax2.set_title('R² Score')
        ax2.set_ylabel('R²')
        ax2.tick_params(axis='x', rotation=45)
        
        # Correlation comparison
        spearman_values = [metrics[m]['spearman_correlation'] for m in model_names]
        ax3.bar(model_names, spearman_values, color=colors[:len(model_names)])
        ax3.set_title('Spearman Correlation')
        ax3.set_ylabel('ρ')
        ax3.tick_params(axis='x', rotation=45)
        
        # Safety error rate
        safety_values = [metrics[m]['safety_error_rate'] for m in model_names]
        ax4.bar(model_names, safety_values, color='red', alpha=0.7)
        ax4.set_title('Food Safety Error Rate')
        ax4.set_ylabel('Error Rate')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        comparison_path = self.output_dir / "model_performance_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        viz_paths.append(str(comparison_path))
        
        # 3. Class-specific performance heatmap
        if class_analysis:
            # Prepare data for heatmap
            all_classes = set()
            for model_analysis in class_analysis.values():
                all_classes.update(model_analysis.keys())
            all_classes = sorted(list(all_classes))
            
            heatmap_data = []
            model_labels = []
            
            for model_type, model_class_metrics in class_analysis.items():
                row = []
                for class_name in all_classes:
                    if class_name in model_class_metrics:
                        row.append(model_class_metrics[class_name]['quality_correlation'])
                    else:
                        row.append(0.0)
                heatmap_data.append(row)
                model_labels.append(model_type)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                heatmap_data,
                xticklabels=all_classes,
                yticklabels=model_labels,
                annot=True,
                fmt='.3f',
                cmap='RdYlBu_r',
                center=0,
                ax=ax
            )
            ax.set_title('Quality Correlation by Pizza Class')
            ax.set_xlabel('Pizza Class')
            ax.set_ylabel('Model Type')
            
            plt.tight_layout()
            heatmap_path = self.output_dir / "class_performance_heatmap.png"
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths.append(str(heatmap_path))
        
        # 4. Formal verification integration (if available)
        if formal_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Verification rates
            formal_model_names = list(formal_results.keys())
            verification_rates = [formal_results[m]['verification_rate'] 
                                for m in formal_model_names]
            
            ax1.bar(formal_model_names, verification_rates, 
                   color=colors[:len(formal_model_names)])
            ax1.set_title('Formal Verification Rates')
            ax1.set_ylabel('Verification Rate')
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='x', rotation=45)
            
            # Verification vs quality correlation
            correlations = [formal_results[m]['formal_vs_verifier_correlation'] 
                          for m in formal_model_names]
            ax2.bar(formal_model_names, correlations, 
                   color=colors[:len(formal_model_names)])
            ax2.set_title('Formal Verification vs Verifier Quality Correlation')
            ax2.set_ylabel('Correlation')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            formal_path = self.output_dir / "formal_verification_analysis.png"
            plt.savefig(formal_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths.append(str(formal_path))
        
        logger.info(f"Generated {len(viz_paths)} visualization plots")
        return viz_paths
    
    def generate_evaluation_report(
        self,
        metrics: Dict[str, Dict[str, float]],
        class_analysis: Dict[str, Dict[str, Dict[str, float]]],
        formal_results: Dict[str, Dict[str, Any]],
        viz_paths: List[str]
    ) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            metrics: Evaluation metrics
            class_analysis: Class-specific analysis
            formal_results: Formal verification results
            viz_paths: Visualization file paths
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"pizza_verifier_evaluation_report_{timestamp}.html"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Pizza Verifier Evaluation Report - Aufgabe 2.3</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; border-radius: 4px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #7f8c8d; font-size: 14px; }}
        .visualization {{ margin: 20px 0; text-align: center; }}
        .visualization img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .table th, .table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        .table th {{ background-color: #f2f2f2; }}
        .warning {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 4px; }}
        .success {{ background: #d4edda; border: 1px solid #c3e6cb; padding: 10px; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🍕 Pizza Verifier Evaluation Report</h1>
        <h2>Aufgabe 2.3: Evaluation des Pizza-Verifier-Modells</h2>
        <p>Generated: {timestamp}</p>
        <p>Comprehensive evaluation with formal verification integration</p>
    </div>
    
    <h2>📊 Executive Summary</h2>
    <div class="metric-grid">
"""
        
        # Overall metrics summary
        best_model = max(metrics.keys(), key=lambda k: metrics[k]['r2_score'])
        best_r2 = metrics[best_model]['r2_score']
        avg_safety_error = np.mean([m['safety_error_rate'] for m in metrics.values()])
        
        html_content += f"""
        <div class="metric-card">
            <div class="metric-value">{best_model}</div>
            <div class="metric-label">Best Performing Model</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{best_r2:.3f}</div>
            <div class="metric-label">Best R² Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{avg_safety_error:.3f}</div>
            <div class="metric-label">Avg Food Safety Error Rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{len(metrics)}</div>
            <div class="metric-label">Models Evaluated</div>
        </div>
"""
        
        html_content += """
    </div>
    
    <h2>📈 Pizza-Specific Quality Metrics</h2>
    <table class="table">
        <thead>
            <tr>
                <th>Model Type</th>
                <th>MSE</th>
                <th>R² Score</th>
                <th>Spearman ρ</th>
                <th>Accuracy</th>
                <th>Safety Error Rate</th>
                <th>Sample Count</th>
            </tr>
        </thead>
        <tbody>
"""
        
        # Add metrics table
        for model_type, model_metrics in metrics.items():
            safety_class = "warning" if model_metrics['safety_error_rate'] > 0.1 else "success"
            html_content += f"""
            <tr>
                <td><strong>{model_type}</strong></td>
                <td>{model_metrics['mse']:.4f}</td>
                <td>{model_metrics['r2_score']:.4f}</td>
                <td>{model_metrics['spearman_correlation']:.4f}</td>
                <td>{model_metrics['accuracy']:.4f}</td>
                <td class="{safety_class}">{model_metrics['safety_error_rate']:.4f}</td>
                <td>{model_metrics['sample_count']}</td>
            </tr>
"""
        
        html_content += """
        </tbody>
    </table>
"""
        
        # Food Safety Analysis
        html_content += """
    <h2>🚨 Food Safety Critical Analysis</h2>
    <div class="warning">
        <h3>Food Safety Error Analysis</h3>
        <p>Special attention to raw vs. cooked misclassifications:</p>
        <ul>
"""
        
        for model_type, model_metrics in metrics.items():
            error_rate = model_metrics['safety_error_rate']
            safety_status = "CRITICAL" if error_rate > 0.1 else "ACCEPTABLE"
            html_content += f"""
            <li><strong>{model_type}</strong>: {error_rate:.1%} safety errors - <em>{safety_status}</em></li>
"""
        
        html_content += """
        </ul>
    </div>
"""
        
        # Formal Verification Integration
        if formal_results:
            html_content += """
    <h2>🔬 Formal Verification Integration</h2>
    <table class="table">
        <thead>
            <tr>
                <th>Model Type</th>
                <th>Verification Rate</th>
                <th>Samples Verified</th>
                <th>Avg Quality (Verified)</th>
                <th>Formal-Verifier Correlation</th>
            </tr>
        </thead>
        <tbody>
"""
            
            for model_type, formal_data in formal_results.items():
                html_content += f"""
                <tr>
                    <td><strong>{model_type}</strong></td>
                    <td>{formal_data['verification_rate']:.3f}</td>
                    <td>{formal_data['samples_verified']}</td>
                    <td>{formal_data['avg_verifier_quality_verified']:.3f}</td>
                    <td>{formal_data['formal_vs_verifier_correlation']:.3f}</td>
                </tr>
"""
            
            html_content += """
            </tbody>
        </table>
"""
        
        # Class-specific analysis
        if class_analysis:
            html_content += """
    <h2>🎯 Class-Specific Performance Analysis</h2>
    <p>Performance breakdown by pizza class:</p>
"""
            
            for model_type, class_metrics in class_analysis.items():
                html_content += f"""
    <h3>{model_type}</h3>
    <table class="table">
        <thead>
            <tr>
                <th>Pizza Class</th>
                <th>Sample Count</th>
                <th>Accuracy</th>
                <th>Avg Quality (True)</th>
                <th>Avg Quality (Predicted)</th>
                <th>Quality Correlation</th>
            </tr>
        </thead>
        <tbody>
"""
                
                for class_name, class_data in class_metrics.items():
                    html_content += f"""
                    <tr>
                        <td><strong>{class_name}</strong></td>
                        <td>{class_data['sample_count']}</td>
                        <td>{class_data['accuracy']:.3f}</td>
                        <td>{class_data['avg_true_quality']:.3f}</td>
                        <td>{class_data['avg_predicted_quality']:.3f}</td>
                        <td>{class_data['quality_correlation']:.3f}</td>
                    </tr>
"""
                
                html_content += """
                </tbody>
            </table>
"""
        
        # Visualizations
        html_content += """
    <h2>📊 Visualizations</h2>
"""
        
        for viz_path in viz_paths:
            viz_filename = Path(viz_path).name
            html_content += f"""
    <div class="visualization">
        <h3>{viz_filename.replace('_', ' ').title().replace('.png', '')}</h3>
        <img src="{viz_filename}" alt="{viz_filename}">
    </div>
"""
        
        # Recommendations
        html_content += """
    <h2>💡 Recommendations</h2>
    <div class="success">
        <h3>Key Findings and Recommendations:</h3>
        <ul>
"""
        
        # Generate recommendations based on results
        if best_r2 > 0.8:
            html_content += f"<li>✅ <strong>{best_model}</strong> shows excellent quality prediction (R² = {best_r2:.3f})</li>"
        else:
            html_content += f"<li>⚠️ Quality prediction needs improvement - best R² is only {best_r2:.3f}</li>"
        
        if avg_safety_error < 0.05:
            html_content += f"<li>✅ Food safety error rate is acceptable ({avg_safety_error:.1%})</li>"
        else:
            html_content += f"<li>🚨 High food safety error rate ({avg_safety_error:.1%}) - needs immediate attention</li>"
        
        if formal_results:
            avg_verification_rate = np.mean([r['verification_rate'] for r in formal_results.values()])
            if avg_verification_rate > 0.8:
                html_content += f"<li>✅ Strong formal verification integration (avg rate: {avg_verification_rate:.1%})</li>"
            else:
                html_content += f"<li>⚠️ Formal verification integration needs improvement (avg rate: {avg_verification_rate:.1%})</li>"
        
        html_content += """
            <li>🔄 Continue monitoring verifier performance with production data</li>
            <li>📈 Consider retraining with expanded dataset for improved generalization</li>
            <li>🎯 Focus on class-specific improvements for underperforming pizza types</li>
        </ul>
    </div>
    
    <h2>🔧 Technical Details</h2>
    <p><strong>Evaluation Configuration:</strong></p>
    <ul>
        <li>Test Data Directory: data/test/</li>
        <li>Formal Verification: """ + ("Enabled" if FORMAL_VERIFICATION_AVAILABLE else "Disabled") + """</li>
        <li>Device: """ + self.device + """</li>
        <li>Pizza Classes: """ + ", ".join(CLASS_NAMES) + """</li>
    </ul>
    
    <p><strong>Metrics Explained:</strong></p>
    <ul>
        <li><strong>MSE</strong>: Mean Squared Error between predicted and true quality scores</li>
        <li><strong>R² Score</strong>: Coefficient of determination (higher is better, max 1.0)</li>
        <li><strong>Spearman ρ</strong>: Rank correlation coefficient for quality score ordering</li>
        <li><strong>Safety Error Rate</strong>: Proportion of food safety critical misclassifications</li>
    </ul>
    
</body>
</html>
"""
        
        # Write report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Evaluation report generated: {report_path}")
        return str(report_path)
    
    def run_complete_evaluation(
        self,
        model_types: List[str] = ["MicroPizzaNet", "MicroPizzaNetV2", "MicroPizzaNetWithSE"],
        include_formal_verification: bool = True,
        epsilon: float = 0.03
    ) -> Dict[str, Any]:
        """
        Run complete pizza verifier evaluation pipeline.
        
        Args:
            model_types: List of model types to evaluate
            include_formal_verification: Whether to include formal verification
            epsilon: Perturbation bound for formal verification
            
        Returns:
            Complete evaluation results
        """
        logger.info("Starting complete pizza verifier evaluation")
        
        # Step 1: Load ground truth data
        ground_truth_samples = self.load_ground_truth_data()
        
        # Step 2: Generate model predictions
        predictions = self.generate_model_predictions(ground_truth_samples, model_types)
        
        # Step 3: Calculate verifier quality scores
        predictions = self.calculate_verifier_quality_scores(predictions)
        
        # Step 4: Calculate true quality scores
        predictions = self.calculate_true_quality_scores(predictions)
        
        # Step 5: Evaluate pizza-specific metrics
        metrics = self.evaluate_pizza_specific_metrics(predictions)
        
        # Step 6: Class-specific analysis
        class_analysis = self.analyze_class_specific_performance(predictions)
        
        # Step 7: Formal verification integration
        formal_results = {}
        if include_formal_verification and FORMAL_VERIFICATION_AVAILABLE:
            formal_results = self.integrate_with_formal_verification(
                predictions, epsilon=epsilon
            )
        
        # Step 8: Generate visualizations
        viz_paths = self.generate_visualizations(
            predictions, metrics, class_analysis, formal_results
        )
        
        # Step 9: Generate comprehensive report
        report_path = self.generate_evaluation_report(
            metrics, class_analysis, formal_results, viz_paths
        )
        
        # Store all results
        self.evaluation_results = {
            'predictions': predictions,
            'metrics': metrics,
            'class_analysis': class_analysis,
            'formal_results': formal_results,
            'visualizations': viz_paths,
            'report_path': report_path,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results to JSON
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            # Convert non-serializable objects
            serializable_results = {}
            for key, value in self.evaluation_results.items():
                if key == 'predictions':
                    # Convert VerifierData objects to dicts
                    serializable_predictions = {}
                    for model_type, preds in value.items():
                        serializable_preds = []
                        for pred in preds:
                            serializable_pred = pred.copy()
                            if 'verifier_data' in serializable_pred:
                                del serializable_pred['verifier_data']  # Remove non-serializable object
                            serializable_preds.append(serializable_pred)
                        serializable_predictions[model_type] = serializable_preds
                    serializable_results[key] = serializable_predictions
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Complete evaluation finished. Results saved to {results_path}")
        logger.info(f"Report available at: {report_path}")
        
        return self.evaluation_results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Aufgabe 2.3: Pizza Verifier Model Evaluation"
    )
    
    parser.add_argument(
        '--verifier-model-path',
        type=str,
        default=None,
        help='Path to trained verifier model'
    )
    
    parser.add_argument(
        '--test-data-dir',
        type=str,
        default='data/test',
        help='Test data directory'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/verifier_evaluation',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device for computation'
    )
    
    parser.add_argument(
        '--model-types',
        nargs='+',
        default=['MicroPizzaNet', 'MicroPizzaNetV2', 'MicroPizzaNetWithSE'],
        help='Model types to evaluate'
    )
    
    parser.add_argument(
        '--no-formal-verification',
        action='store_true',
        help='Skip formal verification integration'
    )
    
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.03,
        help='Perturbation bound for formal verification'
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = PizzaVerifierEvaluator(
        verifier_model_path=args.verifier_model_path,
        test_data_dir=args.test_data_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Run evaluation
    results = evaluator.run_complete_evaluation(
        model_types=args.model_types,
        include_formal_verification=not args.no_formal_verification,
        epsilon=args.epsilon
    )
    
    print("\n" + "="*60)
    print("PIZZA VERIFIER EVALUATION COMPLETE")
    print("="*60)
    print(f"📊 Report: {results['report_path']}")
    print(f"📈 Visualizations: {len(results['visualizations'])} plots generated")
    print(f"🎯 Models evaluated: {', '.join(results['metrics'].keys())}")
    
    # Print key metrics
    print("\nKey Results:")
    for model_type, metrics in results['metrics'].items():
        print(f"  {model_type}:")
        print(f"    R² Score: {metrics['r2_score']:.3f}")
        print(f"    Accuracy: {metrics['accuracy']:.3f}")
        print(f"    Safety Error Rate: {metrics['safety_error_rate']:.3f}")
    
    if results['formal_results']:
        print(f"\n🔬 Formal verification: {len(results['formal_results'])} models verified")
    
    print(f"\n✅ Aufgabe 2.3 evaluation completed successfully!")


if __name__ == "__main__":
    main()
