"""
Advanced Temperature Stability Optimizer
Addresses the critical temperature stability gap (64.4% -> 95% target)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

@dataclass
class TemperatureProfile:
    """Temperature profile with enhanced stability metrics"""
    target_temp: float
    current_temp: float
    stability_window: int = 50
    tolerance: float = 0.05
    thermal_mass: float = 1.0
    heat_capacity: float = 1.0
    
class AdvancedTemperatureController:
    """Enhanced temperature control with PID++ and adaptive learning"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Enhanced PID parameters with adaptive tuning
        self.kp_adaptive = config.get('kp_base', 0.8)
        self.ki_adaptive = config.get('ki_base', 0.1)
        self.kd_adaptive = config.get('kd_base', 0.2)
        
        # Stability enhancement parameters
        self.stability_buffer = []
        self.thermal_history = []
        self.disturbance_prediction = []
        
        # Advanced control features
        self.enable_feedforward = config.get('enable_feedforward', True)
        self.enable_disturbance_rejection = config.get('enable_disturbance_rejection', True)
        self.enable_adaptive_tuning = config.get('enable_adaptive_tuning', True)
        
        # Thermal modeling parameters
        self.thermal_model = {
            'ambient_coupling': 0.1,
            'thermal_lag': 0.95,
            'heat_loss_coefficient': 0.02,
            'thermal_noise_std': 0.01
        }
        
        # Control state
        self.reset_control_state()
        
    def reset_control_state(self):
        """Reset all control state variables"""
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.stability_buffer = []
        self.thermal_history = []
        self.control_output_history = []
        self.disturbance_estimate = 0.0
        
    def enhanced_pid_control(self, current_temp: float, target_temp: float, dt: float = 0.1) -> float:
        """Enhanced PID control with adaptive parameters and disturbance rejection"""
        
        # Calculate error
        error = target_temp - current_temp
        
        # Adaptive PID parameter tuning
        if self.enable_adaptive_tuning:
            self._adapt_pid_parameters(error, current_temp)
        
        # PID terms
        proportional = self.kp_adaptive * error
        
        # Integral with windup protection
        self.integral_error += error * dt
        self.integral_error = np.clip(self.integral_error, -10.0, 10.0)  # Anti-windup
        integral = self.ki_adaptive * self.integral_error
        
        # Derivative with filtering
        derivative_raw = (error - self.previous_error) / dt if dt > 0 else 0.0
        derivative = self.kd_adaptive * self._filter_derivative(derivative_raw)
        
        # Feedforward control
        feedforward = 0.0
        if self.enable_feedforward:
            feedforward = self._calculate_feedforward(target_temp)
        
        # Disturbance rejection
        disturbance_compensation = 0.0
        if self.enable_disturbance_rejection:
            disturbance_compensation = self._estimate_disturbance_compensation(current_temp)
        
        # Combine control signals
        control_output = proportional + integral + derivative + feedforward + disturbance_compensation
        
        # Apply control limits and slew rate limiting
        control_output = self._apply_control_limits(control_output)
        
        # Update history
        self.previous_error = error
        self.control_output_history.append(control_output)
        if len(self.control_output_history) > 100:
            self.control_output_history.pop(0)
        
        return control_output
    
    def _adapt_pid_parameters(self, error: float, current_temp: float):
        """Adaptive PID parameter tuning based on system response"""
        
        # Error-based adaptation
        error_magnitude = abs(error)
        
        if error_magnitude > 2.0:  # Large error - increase proportional gain
            self.kp_adaptive = min(self.kp_adaptive * 1.05, 2.0)
        elif error_magnitude < 0.1:  # Small error - fine-tune with derivative
            self.kd_adaptive = min(self.kd_adaptive * 1.02, 0.5)
        
        # Stability-based adaptation
        stability_metric = self._calculate_stability_metric()
        if stability_metric < 0.7:  # Unstable - reduce aggressive control
            self.kp_adaptive *= 0.98
            self.kd_adaptive *= 0.95
        elif stability_metric > 0.95:  # Very stable - can be more aggressive
            self.ki_adaptive = min(self.ki_adaptive * 1.01, 0.3)
    
    def _filter_derivative(self, derivative_raw: float) -> float:
        """Low-pass filter for derivative term to reduce noise"""
        alpha = 0.1  # Filter coefficient
        if hasattr(self, 'filtered_derivative'):
            self.filtered_derivative = alpha * derivative_raw + (1 - alpha) * self.filtered_derivative
        else:
            self.filtered_derivative = derivative_raw
        return self.filtered_derivative
    
    def _calculate_feedforward(self, target_temp: float) -> float:
        """Calculate feedforward control based on target temperature"""
        # Simple feedforward based on target temperature
        baseline_power = target_temp * 0.1  # Assuming linear relationship
        
        # Ambient temperature compensation
        ambient_temp = self.config.get('ambient_temperature', 20.0)
        ambient_compensation = (target_temp - ambient_temp) * 0.05
        
        return baseline_power + ambient_compensation
    
    def _estimate_disturbance_compensation(self, current_temp: float) -> float:
        """Estimate and compensate for external disturbances"""
        
        # Update thermal history
        self.thermal_history.append(current_temp)
        if len(self.thermal_history) > 20:
            self.thermal_history.pop(0)
        
        if len(self.thermal_history) < 5:
            return 0.0
        
        # Estimate disturbance from temperature trend
        recent_temps = np.array(self.thermal_history[-5:])
        if len(recent_temps) > 1:
            temp_trend = np.polyfit(range(len(recent_temps)), recent_temps, 1)[0]
            
            # Compensate for unexpected temperature changes
            if abs(temp_trend) > 0.1:
                self.disturbance_estimate = -temp_trend * 2.0
            else:
                self.disturbance_estimate *= 0.9  # Decay estimate
        
        return self.disturbance_estimate
    
    def _apply_control_limits(self, control_output: float) -> float:
        """Apply control output limits and slew rate limiting"""
        
        # Power limits
        max_power = self.config.get('max_power', 100.0)
        min_power = self.config.get('min_power', 0.0)
        limited_output = np.clip(control_output, min_power, max_power)
        
        # Slew rate limiting
        if self.control_output_history:
            max_slew_rate = self.config.get('max_slew_rate', 10.0)
            previous_output = self.control_output_history[-1]
            max_change = max_slew_rate * 0.1  # Assuming 0.1s time step
            
            if limited_output - previous_output > max_change:
                limited_output = previous_output + max_change
            elif previous_output - limited_output > max_change:
                limited_output = previous_output - max_change
        
        return limited_output
    
    def _calculate_stability_metric(self) -> float:
        """Calculate current temperature stability metric"""
        
        if len(self.thermal_history) < 10:
            return 0.5  # Default moderate stability
        
        recent_temps = np.array(self.thermal_history[-10:])
        stability_std = np.std(recent_temps)
        
        # Normalize stability (lower std = higher stability)
        stability_metric = max(0.0, 1.0 - stability_std / 2.0)
        
        return stability_metric
    
    def get_stability_analysis(self) -> Dict:
        """Get detailed stability analysis"""
        
        if len(self.thermal_history) < 10:
            return {"status": "insufficient_data", "stability_score": 0.5}
        
        temps = np.array(self.thermal_history)
        
        # Calculate various stability metrics
        stability_metrics = {
            "temperature_std": np.std(temps),
            "temperature_range": np.max(temps) - np.min(temps),
            "stability_score": self._calculate_stability_metric(),
            "trend_slope": np.polyfit(range(len(temps)), temps, 1)[0] if len(temps) > 1 else 0.0,
            "oscillation_frequency": self._estimate_oscillation_frequency(temps),
            "settling_time": self._estimate_settling_time(temps)
        }
        
        # Overall stability assessment
        stability_score = stability_metrics["stability_score"]
        
        if stability_score > 0.95:
            status = "excellent"
        elif stability_score > 0.85:
            status = "good"
        elif stability_score > 0.7:
            status = "acceptable"
        else:
            status = "poor"
        
        stability_metrics["status"] = status
        stability_metrics["recommendations"] = self._generate_stability_recommendations(stability_metrics)
        
        return stability_metrics
    
    def _estimate_oscillation_frequency(self, temps: np.ndarray) -> float:
        """Estimate oscillation frequency in temperature signal"""
        if len(temps) < 20:
            return 0.0
        
        # Simple zero-crossing detection
        detrended = temps - np.mean(temps)
        zero_crossings = np.where(np.diff(np.sign(detrended)))[0]
        
        if len(zero_crossings) > 2:
            avg_period = np.mean(np.diff(zero_crossings)) * 2  # Full period
            frequency = 1.0 / avg_period if avg_period > 0 else 0.0
        else:
            frequency = 0.0
        
        return frequency
    
    def _estimate_settling_time(self, temps: np.ndarray) -> float:
        """Estimate settling time for temperature control"""
        if len(temps) < 10:
            return float('inf')
        
        target_temp = temps[-1]  # Assume current temp is near target
        tolerance = 0.1
        
        # Find last time temperature was outside tolerance
        for i in range(len(temps) - 1, -1, -1):
            if abs(temps[i] - target_temp) > tolerance:
                return len(temps) - i
        
        return 0.0  # Already settled
    
    def _generate_stability_recommendations(self, metrics: Dict) -> List[str]:
        """Generate recommendations for improving stability"""
        recommendations = []
        
        if metrics["temperature_std"] > 0.5:
            recommendations.append("Reduce PID gains to decrease oscillations")
        
        if metrics["oscillation_frequency"] > 0.1:
            recommendations.append("Add derivative filtering or reduce Kd")
        
        if metrics["settling_time"] > 20:
            recommendations.append("Increase proportional gain for faster response")
        
        if metrics["trend_slope"] > 0.05:
            recommendations.append("Check for thermal drift or bias")
        
        if not recommendations:
            recommendations.append("Temperature control is performing well")
        
        return recommendations

class ThermalSimulationEnhancer:
    """Enhanced thermal simulation with realistic physics"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.thermal_model = self._initialize_thermal_model()
        self.disturbance_model = self._initialize_disturbance_model()
        
    def _initialize_thermal_model(self) -> Dict:
        """Initialize enhanced thermal model parameters"""
        return {
            'thermal_mass': self.config.get('thermal_mass', 1.0),
            'heat_capacity': self.config.get('heat_capacity', 1.0),
            'thermal_conductivity': self.config.get('thermal_conductivity', 0.1),
            'convection_coefficient': self.config.get('convection_coefficient', 0.05),
            'radiation_coefficient': self.config.get('radiation_coefficient', 0.001),
            'ambient_temperature': self.config.get('ambient_temperature', 20.0),
            'thermal_time_constant': self.config.get('thermal_time_constant', 10.0)
        }
    
    def _initialize_disturbance_model(self) -> Dict:
        """Initialize disturbance model for realistic simulation"""
        return {
            'ambient_variation_std': 0.5,
            'load_disturbance_std': 0.2,
            'sensor_noise_std': 0.05,
            'periodic_disturbance_amplitude': 0.1,
            'periodic_disturbance_frequency': 0.01
        }
    
    def simulate_thermal_step(self, current_temp: float, control_input: float, dt: float = 0.1) -> float:
        """Simulate one step of thermal dynamics with enhanced realism"""
        
        # Basic thermal dynamics
        thermal_mass = self.thermal_model['thermal_mass']
        heat_capacity = self.thermal_model['heat_capacity']
        ambient_temp = self.thermal_model['ambient_temperature']
        time_constant = self.thermal_model['thermal_time_constant']
        
        # Heat transfer components
        heat_input = control_input
        heat_loss = self._calculate_heat_loss(current_temp, ambient_temp)
        
        # Thermal dynamics equation
        temp_change = (heat_input - heat_loss) / (thermal_mass * heat_capacity) * dt
        
        # Apply thermal lag
        lag_factor = 1.0 - np.exp(-dt / time_constant)
        temp_change *= lag_factor
        
        # Add disturbances
        disturbance = self._generate_thermal_disturbance()
        
        new_temp = current_temp + temp_change + disturbance * dt
        
        return new_temp
    
    def _calculate_heat_loss(self, current_temp: float, ambient_temp: float) -> float:
        """Calculate heat loss through conduction, convection, and radiation"""
        
        temp_diff = current_temp - ambient_temp
        
        # Conduction loss
        conduction_loss = self.thermal_model['thermal_conductivity'] * temp_diff
        
        # Convection loss
        convection_loss = self.thermal_model['convection_coefficient'] * temp_diff
        
        # Radiation loss (T^4 law approximation)
        radiation_loss = self.thermal_model['radiation_coefficient'] * (temp_diff ** 1.3)
        
        total_loss = conduction_loss + convection_loss + radiation_loss
        
        return total_loss
    
    def _generate_thermal_disturbance(self) -> float:
        """Generate realistic thermal disturbances"""
        
        # Ambient temperature variation
        ambient_disturbance = np.random.normal(0, self.disturbance_model['ambient_variation_std'])
        
        # Load disturbances
        load_disturbance = np.random.normal(0, self.disturbance_model['load_disturbance_std'])
        
        # Sensor noise
        sensor_noise = np.random.normal(0, self.disturbance_model['sensor_noise_std'])
        
        # Periodic disturbances (e.g., from ventilation systems)
        time_step = getattr(self, 'current_time', 0)
        periodic_disturbance = (
            self.disturbance_model['periodic_disturbance_amplitude'] * 
            np.sin(2 * np.pi * self.disturbance_model['periodic_disturbance_frequency'] * time_step)
        )
        
        total_disturbance = ambient_disturbance + load_disturbance + sensor_noise + periodic_disturbance
        
        return total_disturbance

def optimize_temperature_stability(config: Dict, simulation_steps: int = 1000) -> Dict:
    """Main function to optimize temperature stability"""
    
    # Initialize components
    controller = AdvancedTemperatureController(config)
    simulator = ThermalSimulationEnhancer(config)
    
    # Simulation parameters
    target_temp = config.get('target_temperature', 75.0)
    dt = config.get('simulation_dt', 0.1)
    
    # Track results
    temperature_history = []
    control_history = []
    stability_scores = []
    
    # Initial conditions
    current_temp = config.get('initial_temperature', 20.0)
    
    # Simulation loop
    for step in range(simulation_steps):
        # Calculate control output
        control_output = controller.enhanced_pid_control(current_temp, target_temp, dt)
        
        # Simulate thermal step
        current_temp = simulator.simulate_thermal_step(current_temp, control_output, dt)
        
        # Track history
        temperature_history.append(current_temp)
        control_history.append(control_output)
        
        # Calculate stability score every 50 steps
        if step % 50 == 0 and step > 100:
            stability_analysis = controller.get_stability_analysis()
            stability_scores.append(stability_analysis['stability_score'])
    
    # Final analysis
    final_stability = controller.get_stability_analysis()
    
    # Calculate improved metrics
    temps = np.array(temperature_history[-500:])  # Last 500 steps
    temp_std = np.std(temps)
    target_achievement = np.mean(np.abs(temps - target_temp) < 0.1)
    
    results = {
        'temperature_history': temperature_history,
        'control_history': control_history,
        'stability_scores': stability_scores,
        'final_stability_analysis': final_stability,
        'temperature_std': temp_std,
        'target_achievement_rate': target_achievement,
        'stability_improvement': final_stability['stability_score'],
        'recommendations': final_stability['recommendations']
    }
    
    return results

# Example usage and configuration
if __name__ == "__main__":
    # Enhanced temperature control configuration
    enhanced_config = {
        'kp_base': 1.2,
        'ki_base': 0.15,
        'kd_base': 0.25,
        'enable_feedforward': True,
        'enable_disturbance_rejection': True,
        'enable_adaptive_tuning': True,
        'max_power': 100.0,
        'min_power': 0.0,
        'max_slew_rate': 15.0,
        'target_temperature': 75.0,
        'initial_temperature': 20.0,
        'ambient_temperature': 20.0,
        'thermal_mass': 1.5,
        'heat_capacity': 1.2,
        'thermal_conductivity': 0.08,
        'convection_coefficient': 0.06,
        'radiation_coefficient': 0.002,
        'thermal_time_constant': 8.0,
        'simulation_dt': 0.1
    }
    
    # Run optimization
    results = optimize_temperature_stability(enhanced_config, simulation_steps=2000)
    
    print(f"Temperature Stability Optimization Results:")
    print(f"Final Stability Score: {results['stability_improvement']:.3f}")
    print(f"Target Achievement Rate: {results['target_achievement_rate']:.3f}")
    print(f"Temperature Std Dev: {results['temperature_std']:.3f}")
    print(f"Status: {results['final_stability_analysis']['status']}")
    print("\nRecommendations:")
    for rec in results['recommendations']:
        print(f"- {rec}")
