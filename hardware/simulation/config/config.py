"""
PG-ES Sim-to-Real Configuration
Configuration file for hybrid Policy Gradient - Evolution Strategy optimization
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional, Any

@dataclass
class SimulationConfig:
    """Core simulation parameters"""
    
    # Time parameters
    dt: float = 0.1  # seconds, simulation timestep
    sim_duration: float = 3600.0  # seconds, 1 hour simulation
    
    # Hardware parameters
    pcb_mass: float = 0.05  # kg
    pcb_surface_area: float = 0.01  # m²
    thermal_mass: float = 0.05  # kg equivalent
    
    # Physics constants
    specific_heat: float = 900.0  # J/(kg*K)
    heat_transfer_coeff: float = 10.0  # W/(m²*K)
    thermal_expansion: float = 15e-6  # /K
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
@dataclass 
class SafetyLimits:
    """Hard constraints - never violate these"""
    
    max_temperature: float = 85.0  # °C
    min_temperature: float = -40.0  # °C
    max_voltage: float = 3.6  # V
    min_voltage: float = 2.7  # V
    max_current: float = 0.6  # A
    max_humidity: float = 95.0  # % RH
    max_thermal_gradient: float = 30.0  # °C/cm
    max_power_dissipation: float = 2.0  # W
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PGConfig:
    """Policy Gradient optimization parameters"""
    
    learning_rate: float = 1e-3
    discount_factor: float = 0.99
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5
    
    # Network architecture
    hidden_sizes: List[int] = None
    activation: str = 'tanh'
    
    # Training parameters
    batch_size: int = 64
    buffer_size: int = 10000
    n_epochs: int = 10
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [64, 64]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ESConfig:
    """Evolution Strategy parameters"""
    
    population_size: int = 50
    sigma: float = 0.1  # mutation strength
    learning_rate: float = 0.01
    
    # Sampling strategy
    antithetic_sampling: bool = True
    fitness_shaping: bool = True
    mirror_sampling: bool = True
    
    # Noise parameters
    noise_decay: float = 0.999
    min_sigma: float = 0.01
    max_sigma: float = 1.0
    
    # Selection parameters
    elite_ratio: float = 0.1
    tournament_size: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DomainRandomizationConfig:
    """Domain randomization for sim-to-real transfer"""
    
    # Environmental noise (std dev)
    temperature_noise: float = 2.0  # °C
    humidity_noise: float = 5.0  # % RH
    voltage_noise: float = 0.05  # V
    
    # Manufacturing variations (multiplicative factors)
    component_tolerance_range: Tuple[float, float] = (0.95, 1.05)
    aging_factor_range: Tuple[float, float] = (1.0, 1.2)
    
    # Disturbance events
    enable_thermal_events: bool = True
    thermal_event_prob: float = 0.01  # per timestep
    thermal_event_magnitude: float = 20.0  # °C
    
    enable_power_events: bool = True
    power_event_prob: float = 0.005
    power_event_magnitude: float = 0.3  # V
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""
    
    log_dir: str = "./logs"
    log_level: str = "INFO"
    
    # Versioning
    experiment_name: str = "pg_es_sim2real"
    version_format: str = "v{major}.{minor}.{patch}"
    
    # Metrics to track
    track_fitness: bool = True
    track_parameters: bool = True
    track_gradients: bool = True
    track_safety_violations: bool = True
    
    # Visualization
    plot_frequency: int = 100  # every N iterations
    save_plots: bool = True
    
    # Checkpointing
    checkpoint_frequency: int = 1000
    keep_n_checkpoints: int = 5
    
    # Fitness normalization
    fitness_normalization_method: str = "standard"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class HybridConfig:
    """Combined PG-ES configuration"""
    
    # Scheduling
    pg_es_ratio: float = 0.7  # 70% PG, 30% ES
    alternating_mode: bool = False
    warm_start_pg_iterations: int = 1000
    
    # Information sharing
    share_experience: bool = True
    experience_buffer_size: int = 50000
    
    # Adaptation
    adaptive_ratio: bool = True
    ratio_adaptation_rate: float = 0.01
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# Optional SupervisorConfig for safety tooling expectations
@dataclass
class SupervisorConfig:
    alert_rate_threshold: float = 0.05
    freeze_on_emergency: bool = True
    max_consecutive_violations: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# Master configuration
@dataclass
class MasterConfig:
    """Master configuration combining all components"""
    
    simulation: SimulationConfig = None
    safety: SafetyLimits = None
    pg: PGConfig = None
    es: ESConfig = None
    domain_rand: DomainRandomizationConfig = None
    logging: LoggingConfig = None
    hybrid: HybridConfig = None
    
    def __post_init__(self):
        if self.simulation is None:
            self.simulation = SimulationConfig()
        if self.safety is None:
            self.safety = SafetyLimits()
        if self.pg is None:
            self.pg = PGConfig()
        if self.es is None:
            self.es = ESConfig()
        if self.domain_rand is None:
            self.domain_rand = DomainRandomizationConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.hybrid is None:
            self.hybrid = HybridConfig()
        # Provide a minimal supervisor config for compatibility with callers
        if not hasattr(self, 'supervisor'):
            # type: ignore[attr-defined]
            self.supervisor = SupervisorConfig()
    
    def validate(self) -> List[str]:
        """Validate configuration for consistency"""
        errors = []
        
        # Safety checks
        if self.safety.max_temperature <= self.safety.min_temperature:
            errors.append("max_temperature must be > min_temperature")
        
        if self.safety.max_voltage <= self.safety.min_voltage:
            errors.append("max_voltage must be > min_voltage")
        
        # PG-ES compatibility
        if self.hybrid.pg_es_ratio < 0 or self.hybrid.pg_es_ratio > 1:
            errors.append("pg_es_ratio must be between 0 and 1")
        
        # Population size should be even for antithetic sampling
        if self.es.antithetic_sampling and self.es.population_size % 2 != 0:
            errors.append("Population size should be even for antithetic sampling")
        
        return errors

# Default configuration instance
DEFAULT_CONFIG = MasterConfig()

# Utility: attribute + dict-style access
class AttrDict(dict):
    """Dict that also supports attribute access and .get/.to_dict."""
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e
    def __setattr__(self, key, value):
        self[key] = value
    def to_dict(self) -> Dict[str, Any]:
        return dict(self)

class Config:
    """Facade config class expected by tests and runtime.
    - Exposes dataclass configs under .simulation, .safety, .pg, .es, .domain_rand, .logging
    - Provides .hybrid as a dict-like with attribute access and .get
    - Includes a minimal .supervisor with to_dict()
    - Implements .copy() for tests that clone configs
    """
    def __init__(self):
        # Core sections
        self.simulation = SimulationConfig()
        self.safety = SafetyLimits()
        self.pg = PGConfig()
        self.es = ESConfig()
        self.domain_rand = DomainRandomizationConfig()
        self.logging = LoggingConfig()
        # Hybrid section supporting both dict and attribute access
        self.hybrid = AttrDict({
            'pg_es_ratio': 0.7,
            'alternating_mode': False,
            'warm_start_pg_iterations': 1000,
            'share_experience': True,
            'experience_buffer_size': 50000,
            'adaptive_ratio': True,
            # Feature flags used by tests and trainer
            'enable_temperature_optimization': True,
            'enable_enhanced_safety': True,
            # Placeholders for nested configs
            'temperature_config': {},
            'safety_config': {}
        })
        # Minimal supervisor section expected by trainer
        self.supervisor = SupervisorConfig()

    def copy(self) -> 'Config':
        # Shallow copy is fine for dataclasses; copy dict for hybrid
        new = Config()
        new.simulation = SimulationConfig(**self.simulation.to_dict())
        new.safety = SafetyLimits(**self.safety.to_dict())
        new.pg = PGConfig(**self.pg.to_dict())
        new.es = ESConfig(**self.es.to_dict())
        new.domain_rand = DomainRandomizationConfig(**self.domain_rand.to_dict())
        new.logging = LoggingConfig(**self.logging.to_dict())
        new.hybrid = AttrDict(self.hybrid.to_dict())
        new.supervisor = SupervisorConfig(**self.supervisor.to_dict())
        return new

# Configuration factory functions
def create_debug_config() -> MasterConfig:
    """Create configuration for debugging (smaller, faster)"""
    config = MasterConfig()
    config.simulation.sim_duration = 60.0  # 1 minute
    config.pg.batch_size = 16
    config.es.population_size = 10
    config.logging.plot_frequency = 10
    return config

def create_production_config() -> MasterConfig:
    """Create configuration for production runs"""
    config = MasterConfig()
    config.simulation.sim_duration = 3600.0  # 1 hour
    config.pg.batch_size = 128
    config.es.population_size = 100
    config.logging.plot_frequency = 500
    return config

def load_config(config_path: str) -> MasterConfig:
    """Load configuration from file"""
    # TODO: Implement JSON/YAML loading
    return DEFAULT_CONFIG

def save_config(config: MasterConfig, config_path: str) -> None:
    """Save configuration to file"""
    # TODO: Implement JSON/YAML saving
    pass
