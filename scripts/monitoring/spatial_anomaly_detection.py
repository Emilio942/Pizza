#!/usr/bin/env python3
"""
Spatial Feature Anomaly Detection System
Part of SPATIAL-4.3: Monitoring und Logging erweitern

This module implements anomaly detection for spatial features, including
statistical outliers, pattern deviations, and model confidence anomalies.
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import threading
from collections import deque, defaultdict
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

try:
    from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock classes for when prometheus is not available
    class Counter:
        def __init__(self, *args, **kwargs):
            pass
        def inc(self, *args, **kwargs):
            pass
        def labels(self, *args, **kwargs):
            return self
    
    class Gauge:
        def __init__(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
        def labels(self, *args, **kwargs):
            return self
    
    class Histogram:
        def __init__(self, *args, **kwargs):
            pass
        def time(self):
            return self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def observe(self, *args, **kwargs):
            pass
        def labels(self, *args, **kwargs):
            return self
    
    class CollectorRegistry:
        def __init__(self):
            pass
    
    REGISTRY = CollectorRegistry()


@dataclass
class SpatialAnomalyResult:
    """Result of spatial anomaly detection"""
    image_id: str
    timestamp: str
    anomaly_type: str
    severity: str  # low, medium, high, critical
    confidence: float
    spatial_location: Optional[Tuple[int, int, int, int]]  # x, y, w, h
    feature_deviation: float
    expected_range: Tuple[float, float]
    actual_value: float
    description: str
    recommendation: str
    metadata: Dict[str, Any]


@dataclass
class FeatureStatistics:
    """Statistical baseline for features"""
    mean: float
    std: float
    min_val: float
    max_val: float
    percentile_95: float
    percentile_5: float
    sample_count: int
    last_updated: str


class SpatialAnomalyDetector:
    """Anomaly detection for spatial features"""
    
    def __init__(self, 
                 sensitivity: float = 0.1,
                 window_size: int = 1000,
                 enable_prometheus: bool = True):
        
        self.sensitivity = sensitivity
        self.window_size = window_size
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        
        # Feature baselines and statistics
        self.feature_baselines = {}
        self.feature_history = defaultdict(lambda: deque(maxlen=window_size))
        self.anomaly_history = deque(maxlen=5000)
        
        # Anomaly detection models
        self.isolation_forest = IsolationForest(
            contamination=sensitivity,
            random_state=42,
            n_estimators=100
        )
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.scaler = StandardScaler()
        
        # Model training state
        self.models_trained = False
        self.training_data = []
        self.min_training_samples = 100
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Prometheus metrics
        if self.enable_prometheus:
            self.setup_prometheus_metrics()
        
        # Anomaly detection rules
        self.setup_detection_rules()
        
        print("🔍 Spatial anomaly detector initialized")
        print(f"Sensitivity: {sensitivity}, Window size: {window_size}")
    
    def setup_prometheus_metrics(self):
        """Setup Prometheus metrics for anomaly detection"""
        self.anomaly_counter = Counter(
            'spatial_anomalies_detected_total',
            'Total number of spatial anomalies detected',
            ['anomaly_type', 'severity']
        )
        
        self.anomaly_confidence = Histogram(
            'spatial_anomaly_confidence',
            'Confidence scores for detected anomalies',
            ['anomaly_type'],
            buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
        )
        
        self.feature_deviation = Gauge(
            'spatial_feature_deviation',
            'Current deviation from baseline for spatial features',
            ['feature_type', 'image_id']
        )
        
        self.baseline_health = Gauge(
            'spatial_baseline_health_score',
            'Health score of feature baselines (0-1)',
            ['feature_type']
        )
    
    def setup_detection_rules(self):
        """Setup anomaly detection rules"""
        self.detection_rules = {
            'statistical_outlier': {
                'threshold': 3.0,  # Standard deviations
                'min_samples': 50,
                'description': 'Feature value is statistical outlier'
            },
            'distribution_shift': {
                'threshold': 0.1,  # KL divergence threshold
                'window': 100,
                'description': 'Feature distribution has shifted'
            },
            'spatial_inconsistency': {
                'threshold': 0.3,  # Spatial consistency score
                'description': 'Spatial features are inconsistent'
            },
            'confidence_anomaly': {
                'threshold': 0.5,  # Confidence threshold
                'description': 'Model confidence is anomalously low'
            },
            'preprocessing_error': {
                'nan_tolerance': 0.01,  # Percentage of NaN values
                'description': 'Preprocessing artifacts detected'
            },
            'feature_correlation': {
                'threshold': 0.2,  # Correlation threshold
                'description': 'Feature correlations are anomalous'
            }
        }
    
    def update_baselines(self, features: Dict[str, np.ndarray], image_id: str):
        """Update feature baselines with new data"""
        with self.lock:
            for feature_name, feature_values in features.items():
                # Flatten feature values
                values = feature_values.flatten() if hasattr(feature_values, 'flatten') else np.array(feature_values).flatten()
                
                # Remove NaN and infinite values
                values = values[np.isfinite(values)]
                
                if len(values) == 0:
                    continue
                
                # Update history
                self.feature_history[feature_name].extend(values)
                
                # Calculate statistics
                all_values = np.array(list(self.feature_history[feature_name]))
                
                if len(all_values) >= 10:  # Minimum samples for statistics
                    self.feature_baselines[feature_name] = FeatureStatistics(
                        mean=np.mean(all_values),
                        std=np.std(all_values),
                        min_val=np.min(all_values),
                        max_val=np.max(all_values),
                        percentile_95=np.percentile(all_values, 95),
                        percentile_5=np.percentile(all_values, 5),
                        sample_count=len(all_values),
                        last_updated=datetime.now(timezone.utc).isoformat()
                    )
                
                # Update training data for ML models
                if len(self.training_data) < self.window_size:
                    self.training_data.extend(values.tolist())
                
                # Retrain models if enough data
                if len(self.training_data) >= self.min_training_samples and not self.models_trained:
                    self._train_anomaly_models()
    
    def _train_anomaly_models(self):
        """Train anomaly detection models"""
        try:
            training_array = np.array(self.training_data).reshape(-1, 1)
            
            # Scale data
            scaled_data = self.scaler.fit_transform(training_array)
            
            # Train isolation forest
            self.isolation_forest.fit(scaled_data)
            
            self.models_trained = True
            print(f"✅ Anomaly detection models trained on {len(self.training_data)} samples")
            
        except Exception as e:
            print(f"❌ Error training anomaly models: {e}")
    
    def detect_anomalies(self, 
                        features: Dict[str, np.ndarray], 
                        image_id: str,
                        confidence_score: Optional[float] = None,
                        spatial_image: Optional[np.ndarray] = None) -> List[SpatialAnomalyResult]:
        """Detect anomalies in spatial features"""
        
        anomalies = []
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Update baselines first
        self.update_baselines(features, image_id)
        
        # 1. Statistical outlier detection
        statistical_anomalies = self._detect_statistical_outliers(features, image_id, timestamp)
        anomalies.extend(statistical_anomalies)
        
        # 2. Distribution shift detection
        distribution_anomalies = self._detect_distribution_shifts(features, image_id, timestamp)
        anomalies.extend(distribution_anomalies)
        
        # 3. Spatial consistency check
        if spatial_image is not None:
            spatial_anomalies = self._detect_spatial_inconsistencies(
                features, spatial_image, image_id, timestamp
            )
            anomalies.extend(spatial_anomalies)
        
        # 4. Confidence-based anomalies
        if confidence_score is not None:
            confidence_anomalies = self._detect_confidence_anomalies(
                confidence_score, image_id, timestamp
            )
            anomalies.extend(confidence_anomalies)
        
        # 5. Preprocessing error detection
        preprocessing_anomalies = self._detect_preprocessing_errors(features, image_id, timestamp)
        anomalies.extend(preprocessing_anomalies)
        
        # 6. Feature correlation anomalies
        correlation_anomalies = self._detect_correlation_anomalies(features, image_id, timestamp)
        anomalies.extend(correlation_anomalies)
        
        # 7. Machine learning based detection
        if self.models_trained:
            ml_anomalies = self._detect_ml_anomalies(features, image_id, timestamp)
            anomalies.extend(ml_anomalies)
        
        # Store anomalies
        with self.lock:
            self.anomaly_history.extend(anomalies)
        
        # Update Prometheus metrics
        if self.enable_prometheus:
            for anomaly in anomalies:
                self.anomaly_counter.labels(
                    anomaly_type=anomaly.anomaly_type,
                    severity=anomaly.severity
                ).inc()
                
                self.anomaly_confidence.labels(
                    anomaly_type=anomaly.anomaly_type
                ).observe(anomaly.confidence)
        
        return anomalies
    
    def _detect_statistical_outliers(self, features: Dict[str, np.ndarray], 
                                   image_id: str, timestamp: str) -> List[SpatialAnomalyResult]:
        """Detect statistical outliers in features"""
        anomalies = []
        rule = self.detection_rules['statistical_outlier']
        
        for feature_name, feature_values in features.items():
            if feature_name not in self.feature_baselines:
                continue
            
            baseline = self.feature_baselines[feature_name]
            
            if baseline.sample_count < rule['min_samples']:
                continue
            
            # Calculate z-scores
            values = feature_values.flatten() if hasattr(feature_values, 'flatten') else np.array(feature_values).flatten()
            values = values[np.isfinite(values)]
            
            if len(values) == 0:
                continue
            
            z_scores = np.abs((values - baseline.mean) / (baseline.std + 1e-8))
            outlier_mask = z_scores > rule['threshold']
            
            if np.any(outlier_mask):
                max_deviation = np.max(z_scores)
                outlier_count = np.sum(outlier_mask)
                
                severity = self._calculate_severity(max_deviation, [3.0, 4.0, 5.0])
                confidence = min(max_deviation / 10.0, 1.0)
                
                anomaly = SpatialAnomalyResult(
                    image_id=image_id,
                    timestamp=timestamp,
                    anomaly_type='statistical_outlier',
                    severity=severity,
                    confidence=confidence,
                    spatial_location=None,
                    feature_deviation=max_deviation,
                    expected_range=(baseline.percentile_5, baseline.percentile_95),
                    actual_value=np.max(values[outlier_mask]),
                    description=f"Statistical outlier in {feature_name}: {outlier_count} values exceed {rule['threshold']} std",
                    recommendation=f"Review {feature_name} extraction or check input data quality",
                    metadata={
                        'feature_name': feature_name,
                        'outlier_count': int(outlier_count),
                        'max_z_score': float(max_deviation),
                        'baseline_mean': float(baseline.mean),
                        'baseline_std': float(baseline.std)
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_distribution_shifts(self, features: Dict[str, np.ndarray], 
                                  image_id: str, timestamp: str) -> List[SpatialAnomalyResult]:
        """Detect distribution shifts in features"""
        anomalies = []
        rule = self.detection_rules['distribution_shift']
        
        for feature_name, feature_values in features.items():
            if feature_name not in self.feature_history:
                continue
            
            history = list(self.feature_history[feature_name])
            if len(history) < rule['window'] * 2:
                continue
            
            # Compare recent vs historical distribution
            recent_data = history[-rule['window']:]
            historical_data = history[:-rule['window']]
            
            # KL divergence between distributions
            try:
                kl_div = self._calculate_kl_divergence(recent_data, historical_data)
                
                if kl_div > rule['threshold']:
                    severity = self._calculate_severity(kl_div, [0.1, 0.2, 0.5])
                    confidence = min(kl_div / 1.0, 1.0)
                    
                    anomaly = SpatialAnomalyResult(
                        image_id=image_id,
                        timestamp=timestamp,
                        anomaly_type='distribution_shift',
                        severity=severity,
                        confidence=confidence,
                        spatial_location=None,
                        feature_deviation=kl_div,
                        expected_range=(0.0, rule['threshold']),
                        actual_value=kl_div,
                        description=f"Distribution shift detected in {feature_name}: KL divergence = {kl_div:.3f}",
                        recommendation=f"Check for systematic changes in {feature_name} or data drift",
                        metadata={
                            'feature_name': feature_name,
                            'kl_divergence': float(kl_div),
                            'window_size': rule['window']
                        }
                    )
                    anomalies.append(anomaly)
                    
            except Exception as e:
                # Skip if KL divergence calculation fails
                pass
        
        return anomalies
    
    def _detect_spatial_inconsistencies(self, features: Dict[str, np.ndarray], 
                                      spatial_image: np.ndarray,
                                      image_id: str, timestamp: str) -> List[SpatialAnomalyResult]:
        """Detect spatial inconsistencies in features"""
        anomalies = []
        rule = self.detection_rules['spatial_inconsistency']
        
        try:
            # Check for spatial consistency
            if 'spatial_features' in features:
                spatial_features = features['spatial_features']
                
                # Calculate spatial consistency score
                consistency_score = self._calculate_spatial_consistency(spatial_features, spatial_image)
                
                if consistency_score < rule['threshold']:
                    severity = self._calculate_severity(
                        1 - consistency_score, [0.3, 0.5, 0.7]
                    )
                    confidence = 1 - consistency_score
                    
                    # Identify problematic spatial region
                    location = self._identify_problematic_region(spatial_features, spatial_image)
                    
                    anomaly = SpatialAnomalyResult(
                        image_id=image_id,
                        timestamp=timestamp,
                        anomaly_type='spatial_inconsistency',
                        severity=severity,
                        confidence=confidence,
                        spatial_location=location,
                        feature_deviation=1 - consistency_score,
                        expected_range=(rule['threshold'], 1.0),
                        actual_value=consistency_score,
                        description=f"Spatial inconsistency detected: consistency score = {consistency_score:.3f}",
                        recommendation="Review spatial preprocessing or check for distorted input",
                        metadata={
                            'consistency_score': float(consistency_score),
                            'spatial_shape': spatial_features.shape if hasattr(spatial_features, 'shape') else None
                        }
                    )
                    anomalies.append(anomaly)
        
        except Exception as e:
            # Create anomaly for spatial processing error
            anomaly = SpatialAnomalyResult(
                image_id=image_id,
                timestamp=timestamp,
                anomaly_type='spatial_processing_error',
                severity='medium',
                confidence=0.8,
                spatial_location=None,
                feature_deviation=1.0,
                expected_range=(0.0, 0.1),
                actual_value=1.0,
                description=f"Error in spatial consistency check: {str(e)[:100]}",
                recommendation="Check spatial feature extraction pipeline",
                metadata={'error': str(e)}
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_confidence_anomalies(self, confidence_score: float, 
                                   image_id: str, timestamp: str) -> List[SpatialAnomalyResult]:
        """Detect confidence-based anomalies"""
        anomalies = []
        rule = self.detection_rules['confidence_anomaly']
        
        if confidence_score < rule['threshold']:
            severity = self._calculate_severity(
                1 - confidence_score, [0.2, 0.4, 0.6]
            )
            anomaly_confidence = 1 - confidence_score
            
            anomaly = SpatialAnomalyResult(
                image_id=image_id,
                timestamp=timestamp,
                anomaly_type='confidence_anomaly',
                severity=severity,
                confidence=anomaly_confidence,
                spatial_location=None,
                feature_deviation=1 - confidence_score,
                expected_range=(rule['threshold'], 1.0),
                actual_value=confidence_score,
                description=f"Low model confidence detected: {confidence_score:.3f}",
                recommendation="Review input quality or consider model uncertainty",
                metadata={
                    'model_confidence': float(confidence_score),
                    'threshold': rule['threshold']
                }
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_preprocessing_errors(self, features: Dict[str, np.ndarray], 
                                   image_id: str, timestamp: str) -> List[SpatialAnomalyResult]:
        """Detect preprocessing errors in features"""
        anomalies = []
        rule = self.detection_rules['preprocessing_error']
        
        for feature_name, feature_values in features.items():
            values = feature_values.flatten() if hasattr(feature_values, 'flatten') else np.array(feature_values).flatten()
            
            # Check for NaN values
            nan_ratio = np.sum(np.isnan(values)) / len(values) if len(values) > 0 else 0
            
            if nan_ratio > rule['nan_tolerance']:
                severity = self._calculate_severity(nan_ratio, [0.01, 0.05, 0.1])
                confidence = min(nan_ratio * 10, 1.0)
                
                anomaly = SpatialAnomalyResult(
                    image_id=image_id,
                    timestamp=timestamp,
                    anomaly_type='preprocessing_error',
                    severity=severity,
                    confidence=confidence,
                    spatial_location=None,
                    feature_deviation=nan_ratio,
                    expected_range=(0.0, rule['nan_tolerance']),
                    actual_value=nan_ratio,
                    description=f"Preprocessing error in {feature_name}: {nan_ratio:.1%} NaN values",
                    recommendation=f"Check preprocessing pipeline for {feature_name}",
                    metadata={
                        'feature_name': feature_name,
                        'nan_ratio': float(nan_ratio),
                        'total_values': len(values),
                        'nan_count': int(np.sum(np.isnan(values)))
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_correlation_anomalies(self, features: Dict[str, np.ndarray], 
                                    image_id: str, timestamp: str) -> List[SpatialAnomalyResult]:
        """Detect feature correlation anomalies"""
        anomalies = []
        rule = self.detection_rules['feature_correlation']
        
        # Need at least 2 features for correlation
        if len(features) < 2:
            return anomalies
        
        try:
            # Create feature matrix
            feature_vectors = []
            feature_names = []
            
            for name, values in features.items():
                flattened = values.flatten() if hasattr(values, 'flatten') else np.array(values).flatten()
                if len(flattened) > 0 and np.all(np.isfinite(flattened)):
                    feature_vectors.append(np.mean(flattened))  # Use mean as representative value
                    feature_names.append(name)
            
            if len(feature_vectors) >= 2:
                # Calculate correlation matrix
                corr_matrix = np.corrcoef(feature_vectors)
                
                # Check for unexpectedly high correlations
                for i in range(len(feature_names)):
                    for j in range(i + 1, len(feature_names)):
                        correlation = abs(corr_matrix[i, j])
                        
                        if correlation < rule['threshold']:  # Unexpectedly low correlation
                            severity = 'low'
                            confidence = 1 - correlation
                            
                            anomaly = SpatialAnomalyResult(
                                image_id=image_id,
                                timestamp=timestamp,
                                anomaly_type='feature_correlation',
                                severity=severity,
                                confidence=confidence,
                                spatial_location=None,
                                feature_deviation=rule['threshold'] - correlation,
                                expected_range=(rule['threshold'], 1.0),
                                actual_value=correlation,
                                description=f"Low correlation between {feature_names[i]} and {feature_names[j]}: {correlation:.3f}",
                                recommendation="Check feature extraction consistency",
                                metadata={
                                    'feature_1': feature_names[i],
                                    'feature_2': feature_names[j],
                                    'correlation': float(correlation)
                                }
                            )
                            anomalies.append(anomaly)
        
        except Exception as e:
            # Skip correlation analysis if it fails
            pass
        
        return anomalies
    
    def _detect_ml_anomalies(self, features: Dict[str, np.ndarray], 
                           image_id: str, timestamp: str) -> List[SpatialAnomalyResult]:
        """Detect anomalies using trained ML models"""
        anomalies = []
        
        try:
            # Prepare feature vector
            feature_vector = []
            for values in features.values():
                flattened = values.flatten() if hasattr(values, 'flatten') else np.array(values).flatten()
                if len(flattened) > 0:
                    feature_vector.extend([
                        np.mean(flattened),
                        np.std(flattened),
                        np.min(flattened),
                        np.max(flattened)
                    ])
            
            if len(feature_vector) == 0:
                return anomalies
            
            # Scale features
            feature_array = np.array(feature_vector).reshape(1, -1)
            scaled_features = self.scaler.transform(feature_array)
            
            # Isolation Forest prediction
            isolation_score = self.isolation_forest.decision_function(scaled_features)[0]
            is_anomaly = self.isolation_forest.predict(scaled_features)[0] == -1
            
            if is_anomaly:
                # Convert score to confidence (higher magnitude = higher confidence)
                confidence = min(abs(isolation_score) / 2.0, 1.0)
                severity = self._calculate_severity(confidence, [0.3, 0.6, 0.8])
                
                anomaly = SpatialAnomalyResult(
                    image_id=image_id,
                    timestamp=timestamp,
                    anomaly_type='ml_anomaly',
                    severity=severity,
                    confidence=confidence,
                    spatial_location=None,
                    feature_deviation=abs(isolation_score),
                    expected_range=(-0.5, 0.5),
                    actual_value=isolation_score,
                    description=f"ML-based anomaly detected: isolation score = {isolation_score:.3f}",
                    recommendation="Review feature patterns or investigate unusual input characteristics",
                    metadata={
                        'isolation_score': float(isolation_score),
                        'feature_dimension': len(feature_vector)
                    }
                )
                anomalies.append(anomaly)
        
        except Exception as e:
            # Skip ML detection if it fails
            pass
        
        return anomalies
    
    def _calculate_kl_divergence(self, data1: List[float], data2: List[float]) -> float:
        """Calculate KL divergence between two distributions"""
        # Create histograms
        combined_data = data1 + data2
        bins = np.histogram_bin_edges(combined_data, bins=20)
        
        hist1, _ = np.histogram(data1, bins=bins, density=True)
        hist2, _ = np.histogram(data2, bins=bins, density=True)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        hist1 = hist1 + epsilon
        hist2 = hist2 + epsilon
        
        # Normalize
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        
        # Calculate KL divergence
        kl_div = np.sum(hist1 * np.log(hist1 / hist2))
        return float(kl_div)
    
    def _calculate_spatial_consistency(self, spatial_features: np.ndarray, 
                                     spatial_image: np.ndarray) -> float:
        """Calculate spatial consistency score"""
        try:
            # Simple consistency metric based on gradient correlation
            if len(spatial_features.shape) >= 2:
                # Calculate gradients
                grad_x = np.gradient(spatial_features, axis=-1)
                grad_y = np.gradient(spatial_features, axis=-2)
                feature_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                
                # Compare with image gradients
                if len(spatial_image.shape) >= 2:
                    img_grad_x = np.gradient(spatial_image, axis=-1)
                    img_grad_y = np.gradient(spatial_image, axis=-2)
                    img_magnitude = np.sqrt(img_grad_x**2 + img_grad_y**2)
                    
                    # Calculate correlation
                    corr_matrix = np.corrcoef(feature_magnitude.flatten(), img_magnitude.flatten())
                    consistency = abs(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
                    
                    return consistency
            
            return 0.5  # Default moderate consistency
            
        except Exception:
            return 0.0  # Low consistency if calculation fails
    
    def _identify_problematic_region(self, spatial_features: np.ndarray, 
                                   spatial_image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Identify spatial region with problems"""
        try:
            if len(spatial_features.shape) >= 2 and len(spatial_image.shape) >= 2:
                # Calculate local inconsistency map
                height, width = spatial_features.shape[-2:]
                
                # Simple approach: find region with highest variance
                if height > 20 and width > 20:
                    variance_map = np.var(spatial_features, axis=0) if len(spatial_features.shape) > 2 else spatial_features
                    
                    # Find location of maximum variance
                    max_loc = np.unravel_index(np.argmax(variance_map), variance_map.shape)
                    
                    # Define region around maximum
                    y, x = max_loc
                    region_size = 50
                    
                    x1 = max(0, x - region_size // 2)
                    y1 = max(0, y - region_size // 2)
                    x2 = min(width, x + region_size // 2)
                    y2 = min(height, y + region_size // 2)
                    
                    return (x1, y1, x2 - x1, y2 - y1)
            
            return None
            
        except Exception:
            return None
    
    def _calculate_severity(self, value: float, thresholds: List[float]) -> str:
        """Calculate severity based on value and thresholds"""
        if value <= thresholds[0]:
            return 'low'
        elif value <= thresholds[1]:
            return 'medium'
        elif value <= thresholds[2]:
            return 'high'
        else:
            return 'critical'
    
    def get_anomaly_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of detected anomalies"""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (hours * 3600)
        
        recent_anomalies = [
            a for a in self.anomaly_history
            if datetime.fromisoformat(a.timestamp.replace('Z', '+00:00')).timestamp() > cutoff_time
        ]
        
        if not recent_anomalies:
            return {'total_anomalies': 0}
        
        # Group by type and severity
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        
        for anomaly in recent_anomalies:
            by_type[anomaly.anomaly_type] += 1
            by_severity[anomaly.severity] += 1
        
        # Calculate statistics
        confidences = [a.confidence for a in recent_anomalies]
        deviations = [a.feature_deviation for a in recent_anomalies]
        
        return {
            'total_anomalies': len(recent_anomalies),
            'by_type': dict(by_type),
            'by_severity': dict(by_severity),
            'avg_confidence': np.mean(confidences),
            'avg_deviation': np.mean(deviations),
            'max_deviation': np.max(deviations),
            'models_trained': self.models_trained,
            'training_samples': len(self.training_data)
        }
    
    def export_anomalies(self, filepath: str, hours: int = 24):
        """Export anomaly data to file"""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (hours * 3600)
        
        recent_anomalies = [
            asdict(a) for a in self.anomaly_history
            if datetime.fromisoformat(a.timestamp.replace('Z', '+00:00')).timestamp() > cutoff_time
        ]
        
        export_data = {
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'hours_covered': hours,
            'total_anomalies': len(recent_anomalies),
            'anomalies': recent_anomalies,
            'summary': self.get_anomaly_summary(hours),
            'detection_rules': self.detection_rules,
            'feature_baselines': {
                name: asdict(baseline) for name, baseline in self.feature_baselines.items()
            }
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

    def detect_feature_count_anomaly(self, actual_count: int, expected_count: int, test_case_name: str) -> Optional[Dict]:
        """Detect anomalies in feature count"""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Calculate deviation ratio
        if expected_count == 0:
            deviation_ratio = float(actual_count) if actual_count > 0 else 0
        else:
            deviation_ratio = abs(actual_count - expected_count) / expected_count
        
        # Threshold for significant deviation (30%)
        threshold = 0.3
        
        if deviation_ratio > threshold:
            severity = self._calculate_severity(deviation_ratio, [0.3, 0.5, 0.8])
            confidence = min(deviation_ratio / 2.0, 1.0)
            
            anomaly_result = SpatialAnomalyResult(
                image_id=test_case_name,
                timestamp=timestamp,
                anomaly_type='feature_count_anomaly',
                severity=severity,
                confidence=confidence,
                spatial_location=None,
                feature_deviation=deviation_ratio,
                expected_range=(expected_count * 0.7, expected_count * 1.3),
                actual_value=actual_count,
                description=f"Feature count anomaly: expected {expected_count}, got {actual_count}",
                recommendation="Check feature extraction pipeline or input data quality",
                metadata={
                    'expected_count': expected_count,
                    'actual_count': actual_count,
                    'deviation_ratio': deviation_ratio,
                    'test_case': test_case_name
                }
            )
            
            # Store in history
            with self.lock:
                self.anomaly_history.append(anomaly_result)
            
            # Update Prometheus metrics
            if self.enable_prometheus:
                self.anomaly_counter.labels(
                    anomaly_type='feature_count_anomaly',
                    severity=severity
                ).inc()
            
            return asdict(anomaly_result)
        
        return None

    def detect_spatial_distribution_anomaly(self, locations: List[Tuple[float, float]], test_case_name: str) -> Optional[Dict]:
        """Detect anomalies in spatial distribution of features"""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        if len(locations) < 3:
            return None
        
        # Calculate spatial distribution metrics
        try:
            import numpy as np
            locations_array = np.array(locations)
            
            # Calculate center of mass
            center_x = np.mean(locations_array[:, 0])
            center_y = np.mean(locations_array[:, 1])
            
            # Calculate distances from center
            distances = np.sqrt((locations_array[:, 0] - center_x)**2 + (locations_array[:, 1] - center_y)**2)
            
            # Calculate distribution uniformity (coefficient of variation)
            if np.mean(distances) > 0:
                uniformity_score = np.std(distances) / np.mean(distances)
            else:
                uniformity_score = 0
            
            # Check for clustering (high uniformity score indicates clustering)
            clustering_threshold = 0.8
            
            if uniformity_score > clustering_threshold:
                severity = self._calculate_severity(uniformity_score, [0.8, 1.2, 1.6])
                confidence = min(uniformity_score / 2.0, 1.0)
                
                anomaly_result = SpatialAnomalyResult(
                    image_id=test_case_name,
                    timestamp=timestamp,
                    anomaly_type='spatial_distribution_anomaly',
                    severity=severity,
                    confidence=confidence,
                    spatial_location=(int(center_x), int(center_y), 50, 50),  # Region around center
                    feature_deviation=uniformity_score,
                    expected_range=(0.0, clustering_threshold),
                    actual_value=uniformity_score,
                    description=f"Spatial clustering detected: uniformity score {uniformity_score:.3f}",
                    recommendation="Check for feature extraction clustering or review spatial preprocessing",
                    metadata={
                        'locations_count': len(locations),
                        'center_of_mass': [center_x, center_y],
                        'uniformity_score': uniformity_score,
                        'mean_distance': np.mean(distances),
                        'test_case': test_case_name
                    }
                )
                
                # Store in history
                with self.lock:
                    self.anomaly_history.append(anomaly_result)
                
                # Update Prometheus metrics
                if self.enable_prometheus:
                    self.anomaly_counter.labels(
                        anomaly_type='spatial_distribution_anomaly',
                        severity=severity
                    ).inc()
                
                return asdict(anomaly_result)
            
        except Exception as e:
            # Return error anomaly if calculation fails
            anomaly_result = SpatialAnomalyResult(
                image_id=test_case_name,
                timestamp=timestamp,
                anomaly_type='spatial_processing_error',
                severity='medium',
                confidence=0.8,
                spatial_location=None,
                feature_deviation=1.0,
                expected_range=(0.0, 0.1),
                actual_value=1.0,
                description=f"Error in spatial distribution analysis: {str(e)[:100]}",
                recommendation="Check spatial analysis pipeline",
                metadata={'error': str(e), 'test_case': test_case_name}
            )
            
            with self.lock:
                self.anomaly_history.append(anomaly_result)
            
            return asdict(anomaly_result)
        
        return None

    def detect_color_distribution_anomaly(self, color_dist: Dict[str, float], test_case_name: str) -> Optional[Dict]:
        """Detect anomalies in color distribution"""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        if not color_dist:
            return None
        
        # Calculate color distribution entropy
        try:
            import numpy as np
            values = list(color_dist.values())
            total = sum(values)
            
            if total == 0:
                return None
            
            # Normalize to probabilities
            probabilities = [v / total for v in values]
            
            # Calculate entropy
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            max_entropy = np.log2(len(probabilities))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            # Check for extremely low entropy (indicates color concentration)
            low_entropy_threshold = 0.3
            # Check for extremely high entropy (indicates color dispersion)
            high_entropy_threshold = 0.9
            
            anomaly_detected = False
            anomaly_type = 'color_distribution_anomaly'
            description = ""
            
            if normalized_entropy < low_entropy_threshold:
                anomaly_detected = True
                description = f"Low color diversity detected: entropy {normalized_entropy:.3f}"
                deviation = low_entropy_threshold - normalized_entropy
            elif normalized_entropy > high_entropy_threshold:
                anomaly_detected = True
                description = f"High color dispersion detected: entropy {normalized_entropy:.3f}"
                deviation = normalized_entropy - high_entropy_threshold
            
            if anomaly_detected:
                severity = self._calculate_severity(deviation, [0.1, 0.2, 0.4])
                confidence = min(deviation / 0.5, 1.0)
                
                anomaly_result = SpatialAnomalyResult(
                    image_id=test_case_name,
                    timestamp=timestamp,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    confidence=confidence,
                    spatial_location=None,
                    feature_deviation=deviation,
                    expected_range=(low_entropy_threshold, high_entropy_threshold),
                    actual_value=normalized_entropy,
                    description=description,
                    recommendation="Review color preprocessing or check input image quality",
                    metadata={
                        'color_distribution': color_dist,
                        'entropy': entropy,
                        'normalized_entropy': normalized_entropy,
                        'color_count': len(color_dist),
                        'test_case': test_case_name
                    }
                )
                
                # Store in history
                with self.lock:
                    self.anomaly_history.append(anomaly_result)
                
                # Update Prometheus metrics
                if self.enable_prometheus:
                    self.anomaly_counter.labels(
                        anomaly_type=anomaly_type,
                        severity=severity
                    ).inc()
                
                return asdict(anomaly_result)
            
        except Exception as e:
            # Return error anomaly if calculation fails
            anomaly_result = SpatialAnomalyResult(
                image_id=test_case_name,
                timestamp=timestamp,
                anomaly_type='color_processing_error',
                severity='medium',
                confidence=0.8,
                spatial_location=None,
                feature_deviation=1.0,
                expected_range=(0.0, 0.1),
                actual_value=1.0,
                description=f"Error in color distribution analysis: {str(e)[:100]}",
                recommendation="Check color analysis pipeline",
                metadata={'error': str(e), 'test_case': test_case_name}
            )
            
            with self.lock:
                self.anomaly_history.append(anomaly_result)
            
            return asdict(anomaly_result)
        
        return None


# Global detector instance
_anomaly_detector_instance = None

def get_spatial_anomaly_detector(**kwargs) -> SpatialAnomalyDetector:
    """Get or create spatial anomaly detector singleton"""
    global _anomaly_detector_instance
    if _anomaly_detector_instance is None:
        _anomaly_detector_instance = SpatialAnomalyDetector(**kwargs)
    return _anomaly_detector_instance


if __name__ == "__main__":
    # Demo usage
    detector = get_spatial_anomaly_detector(sensitivity=0.15)
    
    print("🔍 Running spatial anomaly detection demo...")
    
    # Generate demo features
    for i in range(20):
        # Normal features
        normal_features = {
            'visual_features': np.random.normal(0.5, 0.1, (512,)),
            'spatial_features': np.random.normal(0.3, 0.05, (256,)),
            'depth_features': np.random.normal(0.4, 0.08, (128,))
        }
        
        # Inject some anomalies
        if i == 15:  # Statistical outlier
            normal_features['visual_features'] += 2.0
        elif i == 17:  # NaN values
            normal_features['spatial_features'][:10] = np.nan
        
        # Detect anomalies
        anomalies = detector.detect_anomalies(
            features=normal_features,
            image_id=f"demo_pizza_{i:03d}",
            confidence_score=0.9 if i != 16 else 0.3,  # Low confidence at i=16
            spatial_image=np.random.random((224, 224))
        )
        
        if anomalies:
            print(f"🚨 Anomalies detected in image {i}:")
            for anomaly in anomalies:
                print(f"  - {anomaly.anomaly_type} ({anomaly.severity}): {anomaly.description}")
    
    # Get summary
    summary = detector.get_anomaly_summary()
    print("\n📊 Anomaly Summary:")
    print(json.dumps(summary, indent=2))
    
    # Export data
    detector.export_anomalies("output/spatial_anomalies.json")
    print("✅ Anomaly data exported to output/spatial_anomalies.json")
