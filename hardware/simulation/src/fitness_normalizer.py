"""
Fitness Normalization Utilities
Implements different fitness normalization strategies for PG-ES training
"""

import numpy as np
from typing import List, Dict, Optional, Union
from enum import Enum
import warnings

class NormalizationMethod(Enum):
    """Available fitness normalization methods"""
    STANDARD = "standard"
    RANK_BASED = "rank_based"
    MIN_MAX = "min_max"
    ADAPTIVE = "adaptive"

class FitnessNormalizer:
    """
    Fitness normalization for PG-ES optimization
    Implements methods described in docs/fitness_normalization.md
    """
    
    def __init__(self, method: Union[str, NormalizationMethod] = "standard"):
        if isinstance(method, str):
            method = NormalizationMethod(method)
        self.method = method
        
        # Adaptive normalization state
        self.running_mean = 0.0
        self.running_var = 1.0
        self.update_count = 0
        self.alpha = 0.99  # Exponential moving average factor
        
    def normalize(self, fitness_values: np.ndarray) -> np.ndarray:
        """
        Normalize fitness values according to the selected method
        
        Args:
            fitness_values: Array of fitness values to normalize
            
        Returns:
            Normalized fitness values
        """
        if len(fitness_values) == 0:
            return fitness_values
            
        if self.method == NormalizationMethod.STANDARD:
            return self._standard_normalize(fitness_values)
        elif self.method == NormalizationMethod.RANK_BASED:
            return self._rank_based_normalize(fitness_values)
        elif self.method == NormalizationMethod.MIN_MAX:
            return self._min_max_normalize(fitness_values)
        elif self.method == NormalizationMethod.ADAPTIVE:
            return self._adaptive_normalize(fitness_values)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def _standard_normalize(self, fitness_values: np.ndarray) -> np.ndarray:
        """Standard z-score normalization"""
        mean = np.mean(fitness_values)
        std = np.std(fitness_values)
        
        if std < 1e-8:
            warnings.warn("Standard deviation too small, returning original values")
            return fitness_values
            
        return (fitness_values - mean) / std
    
    def _rank_based_normalize(self, fitness_values: np.ndarray) -> np.ndarray:
        """Rank-based normalization (higher rank = higher fitness)"""
        ranks = np.argsort(np.argsort(fitness_values))
        n = len(fitness_values)
        
        # Convert ranks to normalized values [-1, 1]
        normalized_ranks = 2.0 * ranks / (n - 1) - 1.0
        return normalized_ranks
    
    def _min_max_normalize(self, fitness_values: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1]"""
        min_val = np.min(fitness_values)
        max_val = np.max(fitness_values)
        
        if abs(max_val - min_val) < 1e-8:
            warnings.warn("Range too small, returning original values")
            return fitness_values
            
        return (fitness_values - min_val) / (max_val - min_val)
    
    def _adaptive_normalize(self, fitness_values: np.ndarray) -> np.ndarray:
        """Adaptive normalization using running statistics"""
        current_mean = np.mean(fitness_values)
        current_var = np.var(fitness_values)
        
        # Update running statistics
        if self.update_count == 0:
            self.running_mean = current_mean
            self.running_var = current_var
        else:
            self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * current_mean
            self.running_var = self.alpha * self.running_var + (1 - self.alpha) * current_var
        
        self.update_count += 1
        
        # Normalize using running statistics
        running_std = np.sqrt(self.running_var)
        if running_std < 1e-8:
            warnings.warn("Running standard deviation too small, using current batch")
            return self._standard_normalize(fitness_values)
            
        return (fitness_values - self.running_mean) / running_std
    
    def get_stats(self) -> Dict:
        """Get normalization statistics"""
        return {
            'method': self.method.value,
            'running_mean': self.running_mean,
            'running_var': self.running_var,
            'running_std': np.sqrt(self.running_var),
            'update_count': self.update_count
        }
    
    def reset(self):
        """Reset adaptive normalization state"""
        self.running_mean = 0.0
        self.running_var = 1.0
        self.update_count = 0

def compute_fitness_ranking(fitness_values: np.ndarray) -> np.ndarray:
    """Compute fitness ranking for ES selection"""
    return np.argsort(np.argsort(-fitness_values))  # Descending order ranks

def compute_fitness_weights(fitness_values: np.ndarray, 
                          selection_pressure: float = 1.0) -> np.ndarray:
    """Compute selection weights based on fitness"""
    normalized_fitness = FitnessNormalizer(NormalizationMethod.STANDARD).normalize(fitness_values)
    weights = np.exp(selection_pressure * normalized_fitness)
    return weights / np.sum(weights)

def detect_fitness_stagnation(fitness_history: List[float], 
                            window_size: int = 50,
                            threshold: float = 1e-6) -> bool:
    """Detect if fitness has stagnated"""
    if len(fitness_history) < window_size:
        return False
    
    recent_fitness = fitness_history[-window_size:]
    fitness_std = np.std(recent_fitness)
    
    return fitness_std < threshold
