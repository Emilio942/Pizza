"""
Advanced Temperature Controller for Enhanced Stability
Implements PID control, predictive thermal management, and stability optimization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

@dataclass
class ThermalState:
    """Enhanced thermal state tracking"""
    temperature: float
    target_temperature: float
    temperature_history: List[float]
    gradient: float
    stability_metric: float
    pid_integral: float
    pid_derivative: float
    thermal_mass_estimate: float
    disturbance_estimate: float

class AdvancedTemperatureController:
    """
    Advanced temperature controller with enhanced stability features:
    - Adaptive PID control
    - Predictive thermal management
    - Disturbance rejection
    - Stability optimization
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # PID Parameters (adaptive)
        self.kp = config.get('pid_kp', 2.0)
        self.ki = config.get('pid_ki', 0.5)
        self.kd = config.get('pid_kd', 0.1)
        
        # Adaptive parameters
        self.kp_adaptive_range = (0.5, 5.0)
        self.ki_adaptive_range = (0.1, 2.0)
        self.kd_adaptive_range = (0.01, 0.5)
        
        # Stability metrics
        self.stability_window = config.get('stability_window', 50)
        self.stability_threshold = config.get('stability_threshold', 0.5)  # °C
        
        # Predictive control
        self.prediction_horizon = config.get('prediction_horizon', 10)
        self.thermal_time_constant = config.get('thermal_time_constant', 30.0)
        
        # State tracking
        self.thermal_state = ThermalState(
            temperature=25.0,
            target_temperature=25.0,
            temperature_history=[],
            gradient=0.0,
            stability_metric=0.0,
            pid_integral=0.0,
            pid_derivative=0.0,
            thermal_mass_estimate=1.0,
            disturbance_estimate=0.0
        )
        
        # Performance tracking
        self.control_history = []
        self.stability_history = []
        self.overshoot_count = 0
        self.steady_state_error_history = []
        
    def update_control(self, 
                      current_temp: float, 
                      target_temp: float, 
                      dt: float,
                      power_limit: float = 1.0,
                      environmental_disturbances: Dict = None) -> Dict:
        """
        Advanced temperature control update
        Returns optimized control signals for maximum stability
        """
        
        # Update thermal state
        self._update_thermal_state(current_temp, target_temp, dt)
        
        # Estimate disturbances
        self._estimate_disturbances(environmental_disturbances or {})
        
        # Adaptive PID tuning
        self._adaptive_pid_tuning()
        
        # Calculate control output
        control_output = self._calculate_pid_control(dt)
        
        # Predictive enhancement
        predictive_adjustment = self._predictive_control_adjustment()
        
        # Combine control signals
        total_control = control_output + predictive_adjustment
        
        # Apply power limiting and safety constraints
        safe_control = self._apply_safety_constraints(total_control, power_limit)
        
        # Generate control signals
        control_signals = self._generate_control_signals(safe_control)
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Log control action
        self._log_control_action(control_signals)
        
        return control_signals
    
    def _update_thermal_state(self, current_temp: float, target_temp: float, dt: float):
        """Update thermal state with current measurements"""
        
        # Update basic state
        prev_temp = self.thermal_state.temperature
        self.thermal_state.temperature = current_temp
        self.thermal_state.target_temperature = target_temp
        
        # Update temperature history
        self.thermal_state.temperature_history.append(current_temp)
        if len(self.thermal_state.temperature_history) > self.stability_window * 2:
            self.thermal_state.temperature_history.pop(0)
        
        # Calculate gradient
        if len(self.thermal_state.temperature_history) > 1:
            self.thermal_state.gradient = (current_temp - prev_temp) / dt
        
        # Update stability metric
        self._calculate_stability_metric()
        
        # Estimate thermal mass (adaptive)
        self._estimate_thermal_mass(dt)
    
    def _calculate_stability_metric(self):
        """Calculate temperature stability over recent history"""
        
        if len(self.thermal_state.temperature_history) < self.stability_window:
            self.thermal_state.stability_metric = 0.0
            return
        
        recent_temps = self.thermal_state.temperature_history[-self.stability_window:]
        
        # Calculate stability as inverse of standard deviation
        temp_std = np.std(recent_temps)
        temp_mean = np.mean(recent_temps)
        
        # Normalize by target temperature
        if abs(self.thermal_state.target_temperature) > 0.1:
            normalized_std = temp_std / abs(self.thermal_state.target_temperature)
        else:
            normalized_std = temp_std
        
        # Stability metric (0-1, higher is more stable)
        self.thermal_state.stability_metric = max(0.0, 1.0 - normalized_std * 10.0)
        
        self.stability_history.append(self.thermal_state.stability_metric)
        if len(self.stability_history) > 1000:
            self.stability_history.pop(0)
    
    def _estimate_disturbances(self, environmental_disturbances: Dict):
        """Estimate external thermal disturbances"""
        
        # Extract environmental factors
        ambient_temp_change = environmental_disturbances.get('ambient_temp_delta', 0.0)
        humidity_effect = environmental_disturbances.get('humidity_thermal_effect', 0.0)
        power_supply_variation = environmental_disturbances.get('power_variation', 0.0)
        
        # Combined disturbance estimate
        total_disturbance = ambient_temp_change + humidity_effect + power_supply_variation
        
        # Apply exponential smoothing
        alpha = 0.1
        self.thermal_state.disturbance_estimate = (
            alpha * total_disturbance + 
            (1 - alpha) * self.thermal_state.disturbance_estimate
        )
    
    def _adaptive_pid_tuning(self):
        """Adaptive PID parameter tuning based on system performance"""
        
        # Get recent performance metrics
        if len(self.stability_history) < 10:
            return
        
        recent_stability = np.mean(self.stability_history[-10:])
        error_magnitude = abs(self.thermal_state.temperature - self.thermal_state.target_temperature)
        
        # Stability-based tuning
        if recent_stability < 0.7:  # Low stability
            # Reduce aggressive control
            self.kp = max(self.kp_adaptive_range[0], self.kp * 0.9)
            self.kd = min(self.kd_adaptive_range[1], self.kd * 1.1)
        elif recent_stability > 0.9:  # High stability
            # Allow more aggressive control if needed
            if error_magnitude > 2.0:
                self.kp = min(self.kp_adaptive_range[1], self.kp * 1.05)
        
        # Error-based tuning
        if error_magnitude > 5.0:  # Large error
            self.ki = min(self.ki_adaptive_range[1], self.ki * 1.1)
        elif error_magnitude < 0.5:  # Small error
            self.ki = max(self.ki_adaptive_range[0], self.ki * 0.95)
    
    def _calculate_pid_control(self, dt: float) -> float:
        """Calculate PID control output"""
        
        # Error terms
        error = self.thermal_state.target_temperature - self.thermal_state.temperature
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (with windup protection)
        self.thermal_state.pid_integral += error * dt
        # Anti-windup
        integral_limit = 10.0
        self.thermal_state.pid_integral = np.clip(
            self.thermal_state.pid_integral, -integral_limit, integral_limit
        )
        i_term = self.ki * self.thermal_state.pid_integral
        
        # Derivative term (with filtering)
        error_derivative = -self.thermal_state.gradient  # Negative because we want derivative of error
        # Apply low-pass filter to reduce noise
        alpha_d = 0.8
        self.thermal_state.pid_derivative = (
            alpha_d * self.thermal_state.pid_derivative + 
            (1 - alpha_d) * error_derivative
        )
        d_term = self.kd * self.thermal_state.pid_derivative
        
        # Combined PID output
        pid_output = p_term + i_term + d_term
        
        return pid_output
    
    def _predictive_control_adjustment(self) -> float:
        """Predictive control to anticipate thermal behavior"""
        
        if len(self.thermal_state.temperature_history) < self.prediction_horizon:
            return 0.0
        
        # Simple predictive model: first-order thermal dynamics
        recent_temps = self.thermal_state.temperature_history[-self.prediction_horizon:]
        
        # Estimate temperature trend
        if len(recent_temps) >= 3:
            # Linear trend estimation
            x = np.arange(len(recent_temps))
            coeffs = np.polyfit(x, recent_temps, 1)
            predicted_temp_change = coeffs[0] * self.prediction_horizon
            
            # Predict future error
            predicted_temp = self.thermal_state.temperature + predicted_temp_change
            predicted_error = self.thermal_state.target_temperature - predicted_temp
            
            # Predictive adjustment (proportional to predicted error)
            predictive_gain = 0.3
            predictive_adjustment = predictive_gain * predicted_error
            
            return predictive_adjustment
        
        return 0.0
    
    def _apply_safety_constraints(self, control_signal: float, power_limit: float) -> float:
        """Apply safety constraints to control output"""
        
        # Power limiting
        max_power = power_limit
        min_power = -0.1  # Small negative for cooling
        
        safe_signal = np.clip(control_signal, min_power, max_power)
        
        # Rate limiting (prevent sudden changes)
        if hasattr(self, '_last_control_signal'):
            max_rate = 0.5  # Maximum change per update
            rate_limited_signal = np.clip(
                safe_signal,
                self._last_control_signal - max_rate,
                self._last_control_signal + max_rate
            )
            safe_signal = rate_limited_signal
        
        self._last_control_signal = safe_signal
        
        return safe_signal
    
    def _generate_control_signals(self, control_signal: float) -> Dict:
        """Generate specific control signals for hardware"""
        
        # Convert control signal to specific hardware commands
        control_signals = {
            'target_power': max(0.0, control_signal),  # Heating power
            'cooling_effort': max(0.0, -control_signal * 2.0),  # Cooling (if control is negative)
            'voltage_regulation': 3.3,  # Stable voltage
            'fan_speed': min(1.0, max(0.0, abs(control_signal) * 0.5)),  # Fan based on control effort
        }
        
        # Add stability-based adjustments
        if self.thermal_state.stability_metric < 0.8:
            # Reduce aggressive control when stability is low
            control_signals['target_power'] *= 0.8
            control_signals['cooling_effort'] *= 0.8
        
        return control_signals
    
    def _estimate_thermal_mass(self, dt: float):
        """Estimate thermal mass based on temperature response"""
        
        if len(self.control_history) < 5:
            return
        
        # Simple thermal mass estimation based on temperature response to control
        recent_controls = [entry['control_signal'] for entry in self.control_history[-5:]]
        recent_temp_changes = [entry['temp_change'] for entry in self.control_history[-5:]]
        
        if len(recent_controls) > 0 and len(recent_temp_changes) > 0:
            avg_control = np.mean(recent_controls)
            avg_temp_change = np.mean(recent_temp_changes)
            
            if abs(avg_temp_change) > 0.001:
                # Thermal mass ~ control / temperature_change_rate
                estimated_mass = abs(avg_control) / abs(avg_temp_change) * dt
                
                # Apply exponential smoothing
                alpha = 0.1
                self.thermal_state.thermal_mass_estimate = (
                    alpha * estimated_mass + 
                    (1 - alpha) * self.thermal_state.thermal_mass_estimate
                )
    
    def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        
        error = abs(self.thermal_state.temperature - self.thermal_state.target_temperature)
        self.steady_state_error_history.append(error)
        
        if len(self.steady_state_error_history) > 100:
            self.steady_state_error_history.pop(0)
        
        # Track overshoot
        if len(self.thermal_state.temperature_history) >= 2:
            prev_temp = self.thermal_state.temperature_history[-2]
            current_temp = self.thermal_state.temperature
            target_temp = self.thermal_state.target_temperature
            
            # Check for overshoot
            if ((prev_temp < target_temp < current_temp) or 
                (prev_temp > target_temp > current_temp)):
                self.overshoot_count += 1
    
    def _log_control_action(self, control_signals: Dict):
        """Log control action for analysis"""
        
        control_entry = {
            'timestamp': len(self.control_history),
            'temperature': self.thermal_state.temperature,
            'target_temperature': self.thermal_state.target_temperature,
            'control_signal': control_signals.get('target_power', 0.0),
            'temp_change': self.thermal_state.gradient,
            'stability_metric': self.thermal_state.stability_metric,
            'pid_params': {'kp': self.kp, 'ki': self.ki, 'kd': self.kd},
            'disturbance_estimate': self.thermal_state.disturbance_estimate
        }
        
        self.control_history.append(control_entry)
        
        # Keep history manageable
        if len(self.control_history) > 10000:
            self.control_history.pop(0)
    
    def get_stability_metrics(self) -> Dict:
        """Get comprehensive stability metrics"""
        
        if not self.steady_state_error_history:
            return {"error": "No performance data available"}
        
        recent_errors = self.steady_state_error_history[-50:] if len(self.steady_state_error_history) >= 50 else self.steady_state_error_history
        recent_stability = self.stability_history[-50:] if len(self.stability_history) >= 50 else self.stability_history
        
        metrics = {
            'current_stability': self.thermal_state.stability_metric,
            'average_stability': np.mean(recent_stability) if recent_stability else 0.0,
            'stability_trend': np.mean(recent_stability[-10:]) - np.mean(recent_stability[:10]) if len(recent_stability) >= 20 else 0.0,
            
            'steady_state_error_mean': np.mean(recent_errors),
            'steady_state_error_std': np.std(recent_errors),
            'steady_state_error_max': np.max(recent_errors),
            
            'temperature_stability_percentage': min(100.0, self.thermal_state.stability_metric * 100.0),
            'overshoot_rate': self.overshoot_count / max(1, len(self.control_history)),
            
            'control_effort_mean': np.mean([entry['control_signal'] for entry in self.control_history[-50:]]) if len(self.control_history) >= 50 else 0.0,
            
            'adaptive_pid_params': {
                'kp': self.kp,
                'ki': self.ki, 
                'kd': self.kd
            },
            
            'thermal_mass_estimate': self.thermal_state.thermal_mass_estimate,
            'disturbance_rejection': 1.0 - min(1.0, abs(self.thermal_state.disturbance_estimate) / 10.0)
        }
        
        return metrics
    
    def reset(self):
        """Reset controller state"""
        
        self.thermal_state.pid_integral = 0.0
        self.thermal_state.pid_derivative = 0.0
        self.thermal_state.temperature_history.clear()
        self.control_history.clear()
        self.stability_history.clear()
        self.steady_state_error_history.clear()
        self.overshoot_count = 0
        
        # Reset PID parameters to defaults
        self.kp = self.config.get('pid_kp', 2.0)
        self.ki = self.config.get('pid_ki', 0.5)
        self.kd = self.config.get('pid_kd', 0.1)
