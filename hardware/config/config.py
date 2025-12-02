"""
PG-ES Sim-to-Real Configuration (Top-level alias for tests)
Provides the Config API expected by tests via `from config import Config` and `from config.config import Config`.
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional, Any

@dataclass
class SimulationConfig:
    # Time parameters
    dt: float = 0.1
    sim_duration: float = 3600.0
    # Hardware parameters
    pcb_mass: float = 0.05
    pcb_surface_area: float = 0.01
    thermal_mass: float = 0.05
    # Physics constants
    specific_heat: float = 900.0
    heat_transfer_coeff: float = 10.0
    thermal_expansion: float = 15e-6
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass 
class SafetyLimits:
    max_temperature: float = 85.0
    min_temperature: float = -40.0
    max_voltage: float = 3.6
    min_voltage: float = 2.7
    max_current: float = 0.6
    max_humidity: float = 95.0
    max_thermal_gradient: float = 30.0
    max_power_dissipation: float = 2.0
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PGConfig:
    learning_rate: float = 1e-3
    discount_factor: float = 0.99
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: List[int] = None
    activation: str = 'tanh'
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
    population_size: int = 50
    sigma: float = 0.1
    learning_rate: float = 0.01
    antithetic_sampling: bool = True
    fitness_shaping: bool = True
    mirror_sampling: bool = True
    noise_decay: float = 0.999
    min_sigma: float = 0.01
    max_sigma: float = 1.0
    elite_ratio: float = 0.1
    tournament_size: int = 5
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DomainRandomizationConfig:
    temperature_noise: float = 2.0
    humidity_noise: float = 5.0
    voltage_noise: float = 0.05
    component_tolerance_range: Tuple[float, float] = (0.95, 1.05)
    aging_factor_range: Tuple[float, float] = (1.0, 1.2)
    enable_thermal_events: bool = True
    thermal_event_prob: float = 0.01
    thermal_event_magnitude: float = 20.0
    enable_power_events: bool = True
    power_event_prob: float = 0.005
    power_event_magnitude: float = 0.3
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class LoggingConfig:
    log_dir: str = "./logs"
    log_level: str = "INFO"
    experiment_name: str = "pg_es_sim2real"
    version_format: str = "v{major}.{minor}.{patch}"
    track_fitness: bool = True
    track_parameters: bool = True
    track_gradients: bool = True
    track_safety_violations: bool = True
    plot_frequency: int = 100
    save_plots: bool = True
    checkpoint_frequency: int = 1000
    keep_n_checkpoints: int = 5
    fitness_normalization_method: str = "standard"
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class HybridConfig:
    pg_es_ratio: float = 0.7
    alternating_mode: bool = False
    warm_start_pg_iterations: int = 1000
    share_experience: bool = True
    experience_buffer_size: int = 50000
    adaptive_ratio: bool = True
    ratio_adaptation_rate: float = 0.01
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SupervisorConfig:
    alert_rate_threshold: float = 0.05
    freeze_on_emergency: bool = True
    max_consecutive_violations: int = 3
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class MasterConfig:
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
        if not hasattr(self, 'supervisor'):
            self.supervisor = SupervisorConfig()  # type: ignore[attr-defined]
    def validate(self) -> List[str]:
        errors = []
        if self.safety.max_temperature <= self.safety.min_temperature:
            errors.append("max_temperature must be > min_temperature")
        if self.safety.max_voltage <= self.safety.min_voltage:
            errors.append("max_voltage must be > min_voltage")
        if self.hybrid.pg_es_ratio < 0 or self.hybrid.pg_es_ratio > 1:
            errors.append("pg_es_ratio must be between 0 and 1")
        if self.es.antithetic_sampling and self.es.population_size % 2 != 0:
            errors.append("Population size should be even for antithetic sampling")
        return errors

DEFAULT_CONFIG = MasterConfig()

class AttrDict(dict):
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
    def __init__(self):
        self.simulation = SimulationConfig()
        self.safety = SafetyLimits()
        self.pg = PGConfig()
        self.es = ESConfig()
        self.domain_rand = DomainRandomizationConfig()
        self.logging = LoggingConfig()
        self.hybrid = AttrDict({
            'pg_es_ratio': 0.7,
            'alternating_mode': False,
            'warm_start_pg_iterations': 1000,
            'share_experience': True,
            'experience_buffer_size': 50000,
            'adaptive_ratio': True,
            'enable_temperature_optimization': True,
            'enable_enhanced_safety': True,
            'temperature_config': {},
            'safety_config': {}
        })
        self.supervisor = SupervisorConfig()
    def copy(self) -> 'Config':
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
