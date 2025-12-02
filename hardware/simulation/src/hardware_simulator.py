"""
Hardware Physics Simulation Engine
Simulates thermal, electrical, and mechanical behavior of PCV hardware
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import MasterConfig, SafetyLimits

@dataclass
class SimulationState:
    """Current state of the hardware simulation"""
    
    # Physical state
    temperature: float = 25.0  # °C
    humidity: float = 50.0     # % RH
    voltage: float = 3.3       # V
    current: float = 0.02      # A
    
    # Derived state
    power_dissipation: float = 0.0    # W
    thermal_gradient: float = 0.0     # °C/cm
    degradation_factor: float = 1.0   # 0-1, component health
    
    # Time
    time: float = 0.0          # simulation time in seconds
    dt: float = 0.1            # timestep
    
    # Safety status
    safety_violations: List[str] = None
    is_safe: bool = True
    
    def __post_init__(self):
        if self.safety_violations is None:
            self.safety_violations = []

class HardwareSimulator:
    """Main hardware physics simulation engine"""
    
    def __init__(self, config: MasterConfig):
        self.config = config
        self.state = SimulationState()
        self.logger = logging.getLogger(__name__)
        
        # Physics parameters
        self.thermal_mass = config.simulation.thermal_mass
        self.heat_transfer_coeff = config.simulation.heat_transfer_coeff
        self.surface_area = config.simulation.pcb_surface_area
        self.specific_heat = config.simulation.specific_heat
        
        # Environmental conditions
        self.ambient_temp = 25.0  # °C
        self.ambient_humidity = 50.0  # % RH
        
        # Component degradation tracking
        self.fatigue_cycles = 0
        self.thermal_cycles = 0
        self.humidity_exposure_time = 0.0
        
        # Random number generator for domain randomization
        self.rng = np.random.RandomState(42)
        
        self.logger.info("Hardware simulator initialized")
    
    def reset_state(self) -> SimulationState:
        """Reset simulation to initial conditions"""
        self.state = SimulationState()
        self.fatigue_cycles = 0
        self.thermal_cycles = 0
        self.humidity_exposure_time = 0.0
        
        # Apply domain randomization to initial conditions
        if hasattr(self.config, 'domain_rand'):
            self._apply_domain_randomization()
        
        return self.state
    
    def step(self, control_inputs: Dict[str, float]) -> Tuple[SimulationState, float, bool, Dict]:
        """
        Execute one simulation timestep
        
        Args:
            control_inputs: Dictionary with control commands
                - 'target_power': desired power consumption (W)
                - 'cooling_effort': cooling fan PWM (0-1)
                - 'voltage_regulation': voltage setpoint (V)
        
        Returns:
            state: Updated simulation state
            reward: Fitness/reward value
            done: Whether simulation should terminate
            info: Additional debugging information
        """
        
        # Extract control inputs (accept dict or list/tuple as used by tests)
        if isinstance(control_inputs, dict):
            target_power = control_inputs.get('target_power', 0.1)
            cooling_effort = control_inputs.get('cooling_effort', 0.0)
            voltage_setpoint = control_inputs.get('voltage_regulation', 3.3)
        else:
            # Fallback: interpret as sequence [target_power, cooling_effort, voltage_norm]
            # Tests provide a normalized voltage (value ~ 0..1.5) relative to 5V base
            try:
                tp = float(control_inputs[0]) if len(control_inputs) > 0 else 0.1
                ce = float(control_inputs[1]) if len(control_inputs) > 1 else 0.0
                vn = float(control_inputs[2]) if len(control_inputs) > 2 else (3.3 / 5.0)
            except Exception:
                tp, ce, vn = 0.1, 0.0, (3.3 / 5.0)
            target_power = tp
            cooling_effort = ce
            voltage_setpoint = vn * 5.0  # denormalize to volts
        
        # Update physics
        self._update_thermal_dynamics(target_power, cooling_effort)
        self._update_electrical_dynamics(voltage_setpoint)
        self._update_humidity_dynamics()
        self._update_degradation()
        
        # Apply environmental disturbances
        self._apply_environmental_disturbances()
        
        # Check safety constraints
        self._check_safety_constraints()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Update time
        self.state.time += self.state.dt
        
        # Check termination conditions
        done = self._check_termination_conditions()
        
        # Gather debug info
        info = {
            # Return both the list and count for compatibility with tests
            'safety_violations': list(self.state.safety_violations),
            'safety_violation_count': len(self.state.safety_violations),
            'degradation_factor': self.state.degradation_factor,
            'thermal_cycles': self.thermal_cycles,
            'fatigue_cycles': self.fatigue_cycles,
            'power_efficiency': self.state.voltage * self.state.current / max(target_power, 0.001)
        }
        
        return self.state, reward, done, info
    
    def _update_thermal_dynamics(self, power_input: float, cooling_effort: float):
        """Update temperature based on heat generation and dissipation"""
        
        # Heat generation from power dissipation
        heat_generated = power_input  # W (assuming all power becomes heat)
        
        # Heat dissipation (natural + forced convection)
        natural_convection = self.heat_transfer_coeff * self.surface_area * \
                           (self.state.temperature - self.ambient_temp)
        
        # Enhanced cooling from fan (if present)
        forced_convection = cooling_effort * 20.0 * self.surface_area * \
                          (self.state.temperature - self.ambient_temp)
        
        total_heat_loss = natural_convection + forced_convection
        
        # Net heat change
        net_heat = heat_generated - total_heat_loss
        
        # Temperature change (Q = mcΔT)
        temp_change = (net_heat * self.state.dt) / (self.thermal_mass * self.specific_heat)
        
        self.state.temperature += temp_change
        self.state.power_dissipation = power_input
        
        # Update thermal gradient (simplified)
        self.state.thermal_gradient = abs(temp_change) / self.state.dt * 10  # rough estimate
        
        # Track thermal cycling
        if abs(temp_change) > 1.0:  # significant temperature change
            self.thermal_cycles += 1
    
    def _update_electrical_dynamics(self, voltage_setpoint: float):
        """Update electrical parameters"""
        
        # Simple voltage regulation (first-order response)
        voltage_error = voltage_setpoint - self.state.voltage
        self.state.voltage += 0.1 * voltage_error * self.state.dt
        
        # Current depends on temperature and voltage
        # Simplified model: current increases with temperature and voltage
        temp_factor = 1.0 + 0.002 * (self.state.temperature - 25.0)  # +0.2% per °C
        voltage_factor = (self.state.voltage / 3.3) ** 2
        
        base_current = 0.02  # 20mA base current
        self.state.current = base_current * temp_factor * voltage_factor
        
        # Add some noise
        if hasattr(self.config, 'domain_rand'):
            noise = self.rng.normal(0, self.config.domain_rand.voltage_noise)
            self.state.voltage += noise * self.state.dt
    
    def _update_humidity_dynamics(self):
        """Update humidity and its effects"""
        
        # Simple humidity model - tends toward ambient
        humidity_error = self.ambient_humidity - self.state.humidity
        self.state.humidity += 0.01 * humidity_error * self.state.dt
        
        # Add humidity noise
        if hasattr(self.config, 'domain_rand'):
            noise = self.rng.normal(0, self.config.domain_rand.humidity_noise)
            self.state.humidity += noise * self.state.dt
        
        # Track humidity exposure time for corrosion
        if self.state.humidity > 70.0:
            self.humidity_exposure_time += self.state.dt
    
    def _update_degradation(self):
        """Update component degradation based on stress factors"""
        
        # Temperature stress
        temp_stress = max(0, (self.state.temperature - 60.0) / 25.0)  # stress above 60°C
        
        # Humidity stress  
        humidity_stress = max(0, (self.state.humidity - 70.0) / 25.0)  # stress above 70%
        
        # Voltage stress
        voltage_stress = max(0, abs(self.state.voltage - 3.3) / 0.3)  # stress outside ±0.3V
        
        # Combined stress factor
        total_stress = temp_stress + humidity_stress + voltage_stress
        
        # Degradation rate (very slow for realistic simulation)
        degradation_rate = 1e-6 * total_stress * self.state.dt
        
        self.state.degradation_factor = max(0.0, self.state.degradation_factor - degradation_rate)
        
        # Track fatigue cycles
        if total_stress > 0.1:
            self.fatigue_cycles += 1
    
    def _apply_environmental_disturbances(self):
        """Apply random environmental events"""
        
        if not hasattr(self.config, 'domain_rand'):
            return
        
        dr_config = self.config.domain_rand
        
        # Thermal events (sudden temperature changes)
        if (dr_config.enable_thermal_events and 
            self.rng.random() < dr_config.thermal_event_prob):
            
            temp_delta = self.rng.normal(0, dr_config.thermal_event_magnitude)
            self.ambient_temp += temp_delta
            self.logger.debug(f"Thermal event: ambient temp -> {self.ambient_temp:.1f}°C")
        
        # Power supply events
        if (dr_config.enable_power_events and 
            self.rng.random() < dr_config.power_event_prob):
            
            voltage_delta = self.rng.normal(0, dr_config.power_event_magnitude)
            self.state.voltage += voltage_delta
            self.logger.debug(f"Power event: voltage -> {self.state.voltage:.2f}V")
    
    def _apply_domain_randomization(self):
        """Apply initial domain randomization"""
        
        if not hasattr(self.config, 'domain_rand'):
            return
        
        dr_config = self.config.domain_rand
        
        # Randomize initial conditions
        self.state.temperature += self.rng.normal(0, dr_config.temperature_noise)
        self.state.humidity += self.rng.normal(0, dr_config.humidity_noise)
        self.state.voltage += self.rng.normal(0, dr_config.voltage_noise)
        
        # Randomize component values
        tol_range = dr_config.component_tolerance_range
        tolerance = self.rng.uniform(tol_range[0], tol_range[1])
        
        # Apply tolerance to thermal properties
        self.thermal_mass *= tolerance
        self.heat_transfer_coeff *= self.rng.uniform(tol_range[0], tol_range[1])
    
    def _check_safety_constraints(self):
        """Check if current state violates safety limits"""
        
        safety = self.config.safety
        violations = []
        
        if self.state.temperature > safety.max_temperature:
            violations.append(f"Temperature {self.state.temperature:.1f}°C > {safety.max_temperature}°C")
        
        if self.state.temperature < safety.min_temperature:
            violations.append(f"Temperature {self.state.temperature:.1f}°C < {safety.min_temperature}°C")
        
        if self.state.voltage > safety.max_voltage:
            violations.append(f"Voltage {self.state.voltage:.2f}V > {safety.max_voltage}V")
        
        if self.state.voltage < safety.min_voltage:
            violations.append(f"Voltage {self.state.voltage:.2f}V < {safety.min_voltage}V")
        
        if self.state.current > safety.max_current:
            violations.append(f"Current {self.state.current:.3f}A > {safety.max_current}A")
        
        if self.state.humidity > safety.max_humidity:
            violations.append(f"Humidity {self.state.humidity:.1f}% > {safety.max_humidity}%")
        
        if self.state.power_dissipation > safety.max_power_dissipation:
            violations.append(f"Power {self.state.power_dissipation:.2f}W > {safety.max_power_dissipation}W")
        
        if self.state.thermal_gradient > safety.max_thermal_gradient:
            violations.append(f"Thermal gradient {self.state.thermal_gradient:.1f}°C/cm > {safety.max_thermal_gradient}°C/cm")
        
        self.state.safety_violations = violations
        self.state.is_safe = len(violations) == 0
        
        if violations:
            self.logger.warning(f"Safety violations detected: {violations}")
    
    def _calculate_reward(self) -> float:
        """Calculate reward/fitness value for current state"""
        
        reward = 0.0
        
        # Positive rewards for good operation
        # 1. Stay within normal operating ranges
        if 20 <= self.state.temperature <= 60:
            reward += 1.0
        
        if 30 <= self.state.humidity <= 70:
            reward += 1.0
        
        if 3.2 <= self.state.voltage <= 3.4:
            reward += 1.0
        
        # 2. Efficiency bonus
        if self.state.power_dissipation > 0:
            efficiency = (self.state.voltage * self.state.current) / self.state.power_dissipation
            reward += efficiency * 2.0
        
        # 3. Longevity bonus
        reward += self.state.degradation_factor * 2.0
        
        # Negative rewards for problems
        # 1. Safety violations (severe penalty)
        if not self.state.is_safe:
            reward -= 100.0 * len(self.state.safety_violations)
        
        # 2. Thermal stress
        if self.state.temperature > 70:
            reward -= (self.state.temperature - 70) * 0.5
        
        # 3. High thermal gradients
        if self.state.thermal_gradient > 10:
            reward -= (self.state.thermal_gradient - 10) * 0.1
        
        # 4. Degradation penalty
        degradation_penalty = (1.0 - self.state.degradation_factor) * 10.0
        reward -= degradation_penalty
        
        return reward
    
    def _check_termination_conditions(self) -> bool:
        """Check if simulation should terminate"""
        
        # Time limit
        if self.state.time >= self.config.simulation.sim_duration:
            return True
        
        # Critical safety violation
        if (not self.state.is_safe and 
            any('Temperature' in v for v in self.state.safety_violations)):
            self.logger.error("Critical safety violation - terminating simulation")
            return True
        
        # Component failure
        if self.state.degradation_factor <= 0.1:
            self.logger.warning("Component degradation critical - terminating simulation")
            return True
        
        return False
    
    def get_state_vector(self) -> np.ndarray:
        """Get current state as numpy array for ML algorithms"""
        return np.array([
            self.state.temperature,
            self.state.humidity, 
            self.state.voltage,
            self.state.current,
            self.state.power_dissipation,
            self.state.thermal_gradient,
            self.state.degradation_factor,
            self.state.time / self.config.simulation.sim_duration,  # normalized time
            len(self.state.safety_violations),
            self.thermal_cycles / 1000.0,  # normalized
            self.fatigue_cycles / 1000.0,  # normalized
        ])
    
    def get_observation_space_size(self) -> int:
        """Get size of observation space"""
        return len(self.get_state_vector())
    
    def get_action_space_size(self) -> int:
        """Get size of action space"""
        return 3  # target_power, cooling_effort, voltage_regulation
