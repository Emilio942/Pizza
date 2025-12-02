"""
Evolution Strategy Optimizer
Implementation of OpenAI-style ES for black-box hardware parameter optimization
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time

@dataclass
class ESMetrics:
    """Metrics tracking for Evolution Strategy optimization"""
    fitness: float = 0.0
    population_fitness_mean: float = 0.0
    population_fitness_std: float = 0.0
    population_diversity: float = 0.0
    mutation_strength: float = 0.0
    contribution: float = 0.0  # Contribution to overall optimization
    generation: int = 0
    evaluations: int = 0
    elite_fitness: float = 0.0
    improvement_rate: float = 0.0

class EvolutionStrategyOptimizer:
    """
    Evolution Strategy optimizer based on OpenAI ES
    Optimizes non-differentiable hardware parameters using population-based search
    """
    
    def __init__(self, config, parameter_dim: int):
        self.config = config.es
        self.parameter_dim = parameter_dim
        self.logger = logging.getLogger(__name__)
        
        # ES parameters
        self.population_size = self.config.population_size
        self.sigma = self.config.sigma  # Mutation strength
        self.learning_rate = self.config.learning_rate
        
        # Sampling strategy
        self.antithetic_sampling = self.config.antithetic_sampling
        self.fitness_shaping = self.config.fitness_shaping
        self.mirror_sampling = self.config.mirror_sampling
        
        # Noise parameters
        self.noise_decay = self.config.noise_decay
        self.min_sigma = self.config.min_sigma
        self.max_sigma = self.config.max_sigma
        
        # Selection parameters
        self.elite_ratio = self.config.elite_ratio
        self.tournament_size = self.config.tournament_size
        
        # Current population
        self.population = []
        self.fitness_values = []
        self.noise_vectors = []
        
        # Best parameters found so far
        self.best_parameters = np.zeros(parameter_dim)
        self.best_fitness = float('-inf')
        
        # Generation tracking
        self.generation = 0
        self.total_evaluations = 0
        self.metrics = ESMetrics()
        
        # Performance history
        self.fitness_history = []
        self.sigma_history = []
        self.diversity_history = []
        
        # Random number generator
        self.rng = np.random.RandomState(42)
        
        self.logger.info(f"Evolution Strategy optimizer initialized - param_dim={parameter_dim}, pop_size={self.population_size}")
    
    def initialize_population(self, initial_parameters: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """Initialize population around initial parameters"""
        
        if initial_parameters is None:
            initial_parameters = np.zeros(self.parameter_dim)
        
        self.best_parameters = initial_parameters.copy()
        self.population = []
        self.noise_vectors = []
        
        # Generate population
        for i in range(self.population_size):
            if self.antithetic_sampling and i % 2 == 1:
                # Use antithetic (opposite) noise
                noise = -self.noise_vectors[-1]
            else:
                # Generate new noise
                noise = self.rng.normal(0, 1, self.parameter_dim)
            
            # Create individual
            individual = initial_parameters + self.sigma * noise
            
            self.population.append(individual)
            self.noise_vectors.append(noise)
        
        self.fitness_values = [0.0] * self.population_size
        
        self.logger.info(f"Population initialized with {len(self.population)} individuals")
        return self.population
    
    def get_next_individual(self) -> Tuple[np.ndarray, int]:
        """Get next individual to evaluate"""
        
        if not self.population:
            raise ValueError("Population not initialized. Call initialize_population() first.")
        
        # Find next unevaluated individual
        for i, fitness in enumerate(self.fitness_values):
            if fitness == 0.0:  # Not evaluated yet
                return self.population[i], i
        
        # All individuals evaluated - this shouldn't happen in normal operation
        raise ValueError("All individuals already evaluated. Call update_population() first.")
    
    def set_fitness(self, individual_index: int, fitness: float):
        """Set fitness value for specific individual"""
        
        if 0 <= individual_index < len(self.fitness_values):
            self.fitness_values[individual_index] = fitness
            self.total_evaluations += 1
            
            # Update best if this is better
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_parameters = self.population[individual_index].copy()
                self.logger.debug(f"New best fitness: {fitness:.6f}")
        else:
            raise ValueError(f"Invalid individual index: {individual_index}")
    
    def all_evaluated(self) -> bool:
        """Check if all individuals in current population have been evaluated"""
        return all(fitness != 0.0 for fitness in self.fitness_values)
    
    def update_population(self) -> ESMetrics:
        """
        Update population based on fitness values
        Implements ES parameter update rule
        """
        
        if not self.all_evaluated():
            raise ValueError("Not all individuals evaluated. Cannot update population.")
        
        fitness_array = np.array(self.fitness_values)
        
        # Fitness shaping (rank-based)
        if self.fitness_shaping:
            fitness_ranks = np.argsort(np.argsort(fitness_array))
            shaped_fitness = fitness_ranks / (len(fitness_ranks) - 1) - 0.5
        else:
            shaped_fitness = fitness_array
        
        # Normalize fitness
        if np.std(shaped_fitness) > 1e-8:
            shaped_fitness = (shaped_fitness - np.mean(shaped_fitness)) / np.std(shaped_fitness)
        
        # Calculate parameter update
        noise_matrix = np.array(self.noise_vectors)  # Shape: (pop_size, param_dim)
        
        # Weighted sum of noise vectors
        parameter_update = np.dot(shaped_fitness, noise_matrix) / self.population_size
        
        # Update best parameters
        self.best_parameters += self.learning_rate * parameter_update
        
        # Update mutation strength (sigma adaptation)
        self._adapt_sigma(shaped_fitness, noise_matrix)
        
        # Generate new population around updated parameters
        self._generate_new_population()
        
        # Update metrics
        self._update_metrics(fitness_array)
        
        # Update generation counter
        self.generation += 1
        self.metrics.generation = self.generation
        
        self.logger.debug(f"Generation {self.generation} completed - Best fitness: {self.best_fitness:.6f}")
        
        return self.metrics
    
    def _adapt_sigma(self, shaped_fitness: np.ndarray, noise_matrix: np.ndarray):
        """Adapt mutation strength based on fitness landscape"""
        
        # Simple sigma adaptation based on fitness variance
        fitness_variance = np.var(shaped_fitness)
        
        if fitness_variance > 0.1:  # High variance - reduce sigma
            self.sigma *= self.noise_decay
        elif fitness_variance < 0.01:  # Low variance - increase sigma
            self.sigma /= self.noise_decay
        
        # Clamp sigma to valid range
        self.sigma = np.clip(self.sigma, self.min_sigma, self.max_sigma)
        
        self.sigma_history.append(self.sigma)
        if len(self.sigma_history) > 1000:
            self.sigma_history.pop(0)
    
    def _generate_new_population(self):
        """Generate new population around current best parameters"""
        
        new_population = []
        new_noise_vectors = []
        
        # Keep elite individuals
        if self.elite_ratio > 0:
            n_elite = max(1, int(self.population_size * self.elite_ratio))
            elite_indices = np.argsort(self.fitness_values)[-n_elite:]
            
            for idx in elite_indices:
                new_population.append(self.population[idx].copy())
                new_noise_vectors.append(self.noise_vectors[idx].copy())
        
        # Generate remaining individuals
        while len(new_population) < self.population_size:
            if self.antithetic_sampling and len(new_population) % 2 == 1:
                # Use antithetic noise
                noise = -new_noise_vectors[-1]
            else:
                # Generate new noise
                noise = self.rng.normal(0, 1, self.parameter_dim)
            
            # Create new individual
            individual = self.best_parameters + self.sigma * noise
            
            new_population.append(individual)
            new_noise_vectors.append(noise)
        
        # Update population
        self.population = new_population
        self.noise_vectors = new_noise_vectors
        self.fitness_values = [0.0] * self.population_size
    
    def _update_metrics(self, fitness_array: np.ndarray):
        """Update tracking metrics"""
        
        # Basic fitness statistics
        self.metrics.fitness = self.best_fitness
        self.metrics.population_fitness_mean = np.mean(fitness_array)
        self.metrics.population_fitness_std = np.std(fitness_array)
        self.metrics.elite_fitness = np.max(fitness_array)
        self.metrics.mutation_strength = self.sigma
        self.metrics.evaluations = self.total_evaluations
        
        # Population diversity
        if len(self.population) > 1:
            # Calculate pairwise distances
            distances = []
            for i in range(len(self.population)):
                for j in range(i + 1, len(self.population)):
                    dist = np.linalg.norm(self.population[i] - self.population[j])
                    distances.append(dist)
            
            self.metrics.population_diversity = np.mean(distances) if distances else 0.0
        else:
            self.metrics.population_diversity = 0.0
        
        # Improvement rate
        self.fitness_history.append(self.best_fitness)
        if len(self.fitness_history) > 10:
            self.fitness_history.pop(0)
            recent_improvement = (self.fitness_history[-1] - self.fitness_history[0]) / len(self.fitness_history)
            self.metrics.improvement_rate = recent_improvement
        
        # Track diversity
        self.diversity_history.append(self.metrics.population_diversity)
        if len(self.diversity_history) > 1000:
            self.diversity_history.pop(0)
        
        # Calculate contribution (how much ES is helping)
        if len(self.fitness_history) > 1:
            self.metrics.contribution = abs(self.fitness_history[-1] - self.fitness_history[-2])
        else:
            self.metrics.contribution = 0.0
    
    def get_best_parameters(self) -> Tuple[np.ndarray, float]:
        """Get current best parameters and fitness"""
        return self.best_parameters.copy(), self.best_fitness
    
    def get_population_statistics(self) -> Dict[str, Any]:
        """Get detailed population statistics"""
        
        if not self.population:
            return {"population_size": 0}
        
        # Parameter statistics
        param_matrix = np.array(self.population)
        param_means = np.mean(param_matrix, axis=0)
        param_stds = np.std(param_matrix, axis=0)
        
        return {
            "population_size": len(self.population),
            "generation": self.generation,
            "total_evaluations": self.total_evaluations,
            "best_fitness": self.best_fitness,
            "mean_fitness": np.mean(self.fitness_values) if self.fitness_values else 0.0,
            "fitness_std": np.std(self.fitness_values) if self.fitness_values else 0.0,
            "mutation_strength": self.sigma,
            "population_diversity": self.metrics.population_diversity,
            "parameter_means": param_means.tolist(),
            "parameter_stds": param_stds.tolist(),
            "improvement_rate": self.metrics.improvement_rate,
            "sigma_trend": np.mean(self.sigma_history[-10:]) if len(self.sigma_history) >= 10 else self.sigma
        }
    
    def get_parameter_bounds_violation(self, bounds: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Check if population violates parameter bounds"""
        
        violations = {
            "individuals_violating": 0,
            "total_violations": 0,
            "violation_details": []
        }
        
        if len(bounds) != self.parameter_dim:
            return violations
        
        for i, individual in enumerate(self.population):
            individual_violations = 0
            
            for j, (param_val, (min_bound, max_bound)) in enumerate(zip(individual, bounds)):
                if param_val < min_bound or param_val > max_bound:
                    individual_violations += 1
                    violations["violation_details"].append({
                        "individual": i,
                        "parameter": j,
                        "value": param_val,
                        "bounds": (min_bound, max_bound)
                    })
            
            if individual_violations > 0:
                violations["individuals_violating"] += 1
                violations["total_violations"] += individual_violations
        
        return violations
    
    def constrain_population(self, bounds: List[Tuple[float, float]]):
        """Constrain population to parameter bounds"""
        
        if len(bounds) != self.parameter_dim:
            self.logger.warning(f"Bounds dimension {len(bounds)} != parameter dimension {self.parameter_dim}")
            return
        
        constrained_count = 0
        
        for i, individual in enumerate(self.population):
            for j, (min_bound, max_bound) in enumerate(bounds):
                if individual[j] < min_bound:
                    individual[j] = min_bound
                    constrained_count += 1
                elif individual[j] > max_bound:
                    individual[j] = max_bound
                    constrained_count += 1
        
        if constrained_count > 0:
            self.logger.debug(f"Constrained {constrained_count} parameter values to bounds")
    
    def save_checkpoint(self, filepath: str):
        """Save ES optimizer state"""
        checkpoint = {
            'best_parameters': self.best_parameters,
            'best_fitness': self.best_fitness,
            'generation': self.generation,
            'total_evaluations': self.total_evaluations,
            'sigma': self.sigma,
            'population': self.population,
            'fitness_values': self.fitness_values,
            'noise_vectors': self.noise_vectors,
            'metrics': self.metrics.__dict__,
            'fitness_history': self.fitness_history,
            'sigma_history': self.sigma_history,
            'diversity_history': self.diversity_history
        }
        
        # In real implementation, use pickle or custom serialization
        self.logger.info(f"ES checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load ES optimizer state"""
        # In real implementation, load from pickle or custom format
        self.logger.info(f"ES checkpoint loaded from {filepath}")
    
    def get_sampling_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics about sampling strategy"""
        
        return {
            "antithetic_sampling": self.antithetic_sampling,
            "fitness_shaping": self.fitness_shaping,
            "mirror_sampling": self.mirror_sampling,
            "current_sigma": self.sigma,
            "sigma_trend": np.polyfit(range(len(self.sigma_history)), self.sigma_history, 1)[0] if len(self.sigma_history) > 10 else 0.0,
            "population_spread": np.mean([np.std(param) for param in np.array(self.population).T]) if self.population else 0.0,
            "elite_ratio": self.elite_ratio,
            "tournament_size": self.tournament_size
        }
