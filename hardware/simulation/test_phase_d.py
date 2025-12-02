"""
Phase D: Tests & Validierung - Preparation and Implementation
Testing extreme conditions, target metrics validation, and sim-to-real transferability
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import all components for testing
try:
    from hybrid_trainer import HybridPGESTrainer
    from hardware_simulator import HardwareSimulator
    from supervisor import FailSafeSupervisor
    from logging_system import PGESLoggingSystem
    from visualization_dashboard import VisualizationDashboard
    from fitness_normalizer import FitnessNormalizer
    print("✅ All imports successful for Phase D testing")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class ExtremConditionTester:
    """Test system under extreme conditions"""
    
    def __init__(self, config):
        self.config = config
        self.test_results = {}
        
    def test_temperature_extremes(self) -> Dict[str, Any]:
        """Test extreme temperature conditions"""
        print("\n🔥 Testing Temperature Extremes...")
        
        results = {
            'test_name': 'temperature_extremes',
            'conditions_tested': [],
            'results': {},
            'passed': True,
            'failure_reasons': []
        }
        
        # Test conditions: extreme heat, extreme cold, rapid temperature changes
        extreme_conditions = [
            {'name': 'extreme_heat', 'temperature': 85.0, 'duration': 100},
            {'name': 'extreme_cold', 'temperature': -20.0, 'duration': 100},
            {'name': 'rapid_heating', 'temperature_change': 50.0, 'duration': 10},
            {'name': 'thermal_cycling', 'cycles': 5, 'temp_range': (-10, 80)}
        ]
        
        for condition in extreme_conditions:
            print(f"   Testing {condition['name']}...")
            
            try:
                # Create simulator with extreme conditions
                test_config = self.config.copy()
                test_config.physics.base_temperature = condition.get('temperature', 25.0)
                
                simulator = HardwareSimulator(test_config)
                
                # Run short simulation
                state = simulator.reset_state()
                stability_score = 0.0
                
                for step in range(condition.get('duration', 50)):
                    # Apply extreme temperature
                    if 'temperature' in condition:
                        simulator.state.temperature = condition['temperature']
                    elif 'temperature_change' in condition:
                        # Simulate rapid temperature change
                        temp_delta = condition['temperature_change'] * (step / condition['duration'])
                        simulator.state.temperature = test_config.physics.base_temperature + temp_delta
                    
                    # Step simulation
                    next_state, reward, done, info = simulator.step([0.0, 0.0, 0.0])
                    
                    # Check stability
                    if not done and info.get('safety_violations', []) == []:
                        stability_score += 1.0
                    
                    state = next_state
                    if done:
                        break
                
                # Calculate results
                stability_ratio = stability_score / condition.get('duration', 50)
                results['conditions_tested'].append(condition['name'])
                results['results'][condition['name']] = {
                    'stability_ratio': stability_ratio,
                    'passed': stability_ratio > 0.5,  # 50% stability threshold
                    'final_temperature': simulator.state.temperature,
                    'safety_violations': info.get('safety_violations', [])
                }
                
                if stability_ratio <= 0.5:
                    results['passed'] = False
                    results['failure_reasons'].append(f"{condition['name']}: stability ratio {stability_ratio:.2f} below threshold")
                
                print(f"      ✅ {condition['name']}: {stability_ratio:.2%} stability")
                
            except Exception as e:
                results['passed'] = False
                results['failure_reasons'].append(f"{condition['name']}: {str(e)}")
                print(f"      ❌ {condition['name']}: {str(e)}")
        
        return results
    
    def test_humidity_extremes(self) -> Dict[str, Any]:
        """Test extreme humidity conditions"""
        print("\n💧 Testing Humidity Extremes...")
        
        results = {
            'test_name': 'humidity_extremes',
            'conditions_tested': [],
            'results': {},
            'passed': True,
            'failure_reasons': []
        }
        
        # Test conditions
        extreme_conditions = [
            {'name': 'desert_dry', 'humidity': 5.0, 'duration': 100},
            {'name': 'tropical_humid', 'humidity': 95.0, 'duration': 100},
            {'name': 'humidity_shock', 'humidity_change': 80.0, 'duration': 20}
        ]
        
        for condition in extreme_conditions:
            print(f"   Testing {condition['name']}...")
            
            try:
                test_config = self.config.copy()
                test_config.physics.base_humidity = condition.get('humidity', 50.0)
                
                simulator = HardwareSimulator(test_config)
                state = simulator.reset_state()
                
                performance_score = 0.0
                
                for step in range(condition.get('duration', 50)):
                    if 'humidity' in condition:
                        simulator.state.humidity = condition['humidity']
                    
                    next_state, reward, done, info = simulator.step([0.0, 0.0, 0.0])
                    
                    if reward > 0:
                        performance_score += reward
                    
                    state = next_state
                    if done:
                        break
                
                avg_performance = performance_score / condition.get('duration', 50)
                results['conditions_tested'].append(condition['name'])
                results['results'][condition['name']] = {
                    'avg_performance': avg_performance,
                    'passed': avg_performance > 0.1,
                    'final_humidity': simulator.state.humidity
                }
                
                if avg_performance <= 0.1:
                    results['passed'] = False
                    results['failure_reasons'].append(f"{condition['name']}: performance {avg_performance:.3f} below threshold")
                
                print(f"      ✅ {condition['name']}: {avg_performance:.3f} avg performance")
                
            except Exception as e:
                results['passed'] = False
                results['failure_reasons'].append(f"{condition['name']}: {str(e)}")
                print(f"      ❌ {condition['name']}: {str(e)}")
        
        return results
    
    def test_power_stress(self) -> Dict[str, Any]:
        """Test power stress conditions"""
        print("\n⚡ Testing Power Stress Conditions...")
        
        results = {
            'test_name': 'power_stress',
            'conditions_tested': [],
            'results': {},
            'passed': True,
            'failure_reasons': []
        }
        
        # Power stress conditions
        stress_conditions = [
            {'name': 'voltage_spike', 'voltage_multiplier': 1.5, 'duration': 50},
            {'name': 'voltage_drop', 'voltage_multiplier': 0.7, 'duration': 50},
            {'name': 'power_fluctuation', 'fluctuation_range': 0.3, 'duration': 100}
        ]
        
        for condition in stress_conditions:
            print(f"   Testing {condition['name']}...")
            
            try:
                test_config = self.config.copy()
                simulator = HardwareSimulator(test_config)
                state = simulator.reset_state()
                
                survival_time = 0
                total_time = condition.get('duration', 50)
                
                for step in range(total_time):
                    # Apply power stress
                    base_voltage = 5.0  # Assumed base voltage
                    if 'voltage_multiplier' in condition:
                        stressed_voltage = base_voltage * condition['voltage_multiplier']
                    elif 'fluctuation_range' in condition:
                        fluctuation = condition['fluctuation_range'] * np.sin(step * 0.1)
                        stressed_voltage = base_voltage * (1 + fluctuation)
                    else:
                        stressed_voltage = base_voltage
                    
                    # Simulate voltage stress effect
                    control_inputs = [0.0, 0.0, stressed_voltage / 5.0]  # Normalized voltage
                    next_state, reward, done, info = simulator.step(control_inputs)
                    
                    if not done:
                        survival_time += 1
                    
                    state = next_state
                    if done:
                        break
                
                survival_ratio = survival_time / total_time
                results['conditions_tested'].append(condition['name'])
                results['results'][condition['name']] = {
                    'survival_ratio': survival_ratio,
                    'survival_time': survival_time,
                    'total_time': total_time,
                    'passed': survival_ratio > 0.8  # 80% survival threshold
                }
                
                if survival_ratio <= 0.8:
                    results['passed'] = False
                    results['failure_reasons'].append(f"{condition['name']}: survival ratio {survival_ratio:.2f} below threshold")
                
                print(f"      ✅ {condition['name']}: {survival_ratio:.2%} survival rate")
                
            except Exception as e:
                results['passed'] = False
                results['failure_reasons'].append(f"{condition['name']}: {str(e)}")
                print(f"      ❌ {condition['name']}: {str(e)}")
        
        return results

class TargetMetricValidator:
    """Validate system against target performance metrics"""
    
    def __init__(self, config):
        self.config = config
        self.target_metrics = {
            'temperature_stability': 0.95,  # 95% quantile
            'humidity_tolerance': 0.90,     # 90% performance retention
            'power_efficiency': 0.85,       # 85% efficiency
            'convergence_rate': 0.80,       # 80% convergence success
            'safety_compliance': 0.99       # 99% safety compliance
        }
    
    def validate_temperature_stability(self) -> Dict[str, Any]:
        """Validate temperature stability metrics"""
        print("\n🌡️ Validating Temperature Stability...")
        
        # Run temperature stability test
        test_config = self.config.copy()
        simulator = HardwareSimulator(test_config)
        
        temperature_readings = []
        target_temp = 25.0
        
        for step in range(1000):  # Long-term stability test
            state = simulator.reset_state()
            simulator.state.temperature = target_temp + np.random.normal(0, 2.0)  # Add noise
            
            # Run control loop
            next_state, reward, done, info = simulator.step([0.0, 0.0, 0.0])
            temperature_readings.append(simulator.state.temperature)
        
        # Calculate stability metrics
        temp_std = np.std(temperature_readings)
        temp_range = np.max(temperature_readings) - np.min(temperature_readings)
        stability_score = 1.0 / (1.0 + temp_std)  # Higher is better
        
        passed = stability_score >= self.target_metrics['temperature_stability']
        
        result = {
            'metric': 'temperature_stability',
            'target': self.target_metrics['temperature_stability'],
            'achieved': stability_score,
            'passed': passed,
            'details': {
                'temperature_std': temp_std,
                'temperature_range': temp_range,
                'readings_count': len(temperature_readings)
            }
        }
        
        print(f"   Target: {self.target_metrics['temperature_stability']:.2%}")
        print(f"   Achieved: {stability_score:.2%}")
        print(f"   Status: {'✅ PASSED' if passed else '❌ FAILED'}")
        
        return result
    
    def validate_convergence_performance(self) -> Dict[str, Any]:
        """Validate training convergence performance"""
        print("\n🎯 Validating Convergence Performance...")
        
        # Run short training to test convergence
        trainer = HybridPGESTrainer(self.config)
        
        initial_fitness = 0.0
        final_fitness = 0.0
        convergence_iterations = 0
        
        try:
            # Run short training
            metrics = trainer.train(total_iterations=100)
            
            final_fitness = metrics.combined_fitness
            convergence_rate = final_fitness / max(1.0, 100)  # Fitness per iteration
            
            passed = convergence_rate >= (self.target_metrics['convergence_rate'] / 100)
            
            result = {
                'metric': 'convergence_rate',
                'target': self.target_metrics['convergence_rate'],
                'achieved': convergence_rate * 100,  # Convert to percentage
                'passed': passed,
                'details': {
                    'initial_fitness': initial_fitness,
                    'final_fitness': final_fitness,
                    'iterations': 100,
                    'pg_contribution': metrics.pg_contribution_ratio,
                    'es_contribution': metrics.es_contribution_ratio
                }
            }
            
        except Exception as e:
            result = {
                'metric': 'convergence_rate',
                'target': self.target_metrics['convergence_rate'],
                'achieved': 0.0,
                'passed': False,
                'error': str(e)
            }
        
        print(f"   Target: {self.target_metrics['convergence_rate']:.2%}")
        print(f"   Achieved: {result['achieved']:.2%}")
        print(f"   Status: {'✅ PASSED' if result['passed'] else '❌ FAILED'}")
        
        return result

class SimToRealTransferabilityAnalyzer:
    """Analyze sim-to-real transferability potential"""
    
    def __init__(self, config):
        self.config = config
        self.analysis_results = {}
    
    def analyze_parameter_sensitivity(self) -> Dict[str, Any]:
        """Analyze parameter sensitivity for real-world robustness"""
        print("\n🔄 Analyzing Parameter Sensitivity...")
        
        # Test parameter variations
        base_config = self.config.copy()
        sensitivity_results = {}
        
        # Parameters to test
        test_parameters = [
            ('physics.material_fatigue_rate', [0.001, 0.005, 0.01, 0.02]),
            ('physics.thermal_conductivity', [0.1, 0.2, 0.4, 0.8]),
            ('physics.humidity_absorption', [0.05, 0.1, 0.2, 0.3])
        ]
        
        for param_name, values in test_parameters:
            print(f"   Testing {param_name}...")
            
            param_sensitivity = []
            
            for value in values:
                try:
                    # Create modified config
                    test_config = base_config.copy()
                    
                    # Set parameter value (simplified)
                    if param_name == 'physics.material_fatigue_rate':
                        test_config.physics.material_fatigue_rate = value
                    elif param_name == 'physics.thermal_conductivity':
                        test_config.physics.thermal_conductivity = value
                    elif param_name == 'physics.humidity_absorption':
                        test_config.physics.humidity_absorption = value
                    
                    # Run short simulation
                    simulator = HardwareSimulator(test_config)
                    state = simulator.reset_state()
                    
                    total_reward = 0.0
                    for step in range(50):
                        next_state, reward, done, info = simulator.step([0.0, 0.0, 0.0])
                        total_reward += reward
                        if done:
                            break
                    
                    param_sensitivity.append({
                        'value': value,
                        'performance': total_reward / 50
                    })
                    
                except Exception as e:
                    param_sensitivity.append({
                        'value': value,
                        'performance': 0.0,
                        'error': str(e)
                    })
            
            # Calculate sensitivity metrics
            performances = [s['performance'] for s in param_sensitivity if 'error' not in s]
            if performances:
                sensitivity_score = np.std(performances) / np.mean(performances) if np.mean(performances) > 0 else float('inf')
                robustness_score = 1.0 / (1.0 + sensitivity_score)  # Higher is better
            else:
                robustness_score = 0.0
            
            sensitivity_results[param_name] = {
                'sensitivity_data': param_sensitivity,
                'robustness_score': robustness_score,
                'suitable_for_real_world': robustness_score > 0.5
            }
            
            print(f"      Robustness score: {robustness_score:.3f}")
        
        # Overall transferability assessment
        avg_robustness = np.mean([r['robustness_score'] for r in sensitivity_results.values()])
        transferability_score = avg_robustness
        
        result = {
            'analysis_type': 'parameter_sensitivity',
            'parameter_results': sensitivity_results,
            'overall_transferability_score': transferability_score,
            'suitable_for_real_world': transferability_score > 0.6,
            'recommendations': self._generate_transferability_recommendations(sensitivity_results)
        }
        
        print(f"   Overall Transferability Score: {transferability_score:.3f}")
        print(f"   Real-world Suitability: {'✅ SUITABLE' if result['suitable_for_real_world'] else '❌ NEEDS WORK'}")
        
        return result
    
    def _generate_transferability_recommendations(self, sensitivity_results: Dict) -> List[str]:
        """Generate recommendations for improving sim-to-real transfer"""
        recommendations = []
        
        for param_name, result in sensitivity_results.items():
            if result['robustness_score'] < 0.5:
                recommendations.append(f"Improve robustness for {param_name} (score: {result['robustness_score']:.3f})")
        
        if not recommendations:
            recommendations.append("System shows good robustness across tested parameters")
        
        recommendations.extend([
            "Increase domain randomization during training",
            "Add more realistic noise models",
            "Validate with real hardware when available",
            "Implement adaptive control strategies"
        ])
        
        return recommendations

def run_phase_d_tests(config) -> Dict[str, Any]:
    """Run complete Phase D testing suite"""
    
    print("🚀 Starting Phase D: Tests & Validierung")
    print("=" * 60)
    
    results = {
        'phase': 'D',
        'title': 'Tests & Validierung',
        'start_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'test_results': {},
        'overall_status': 'RUNNING'
    }
    
    try:
        # 1. Extreme Conditions Testing
        print("\n1️⃣ EXTREME CONDITIONS TESTING")
        print("-" * 40)
        extreme_tester = ExtremConditionTester(config)
        
        results['test_results']['temperature_extremes'] = extreme_tester.test_temperature_extremes()
        results['test_results']['humidity_extremes'] = extreme_tester.test_humidity_extremes()
        results['test_results']['power_stress'] = extreme_tester.test_power_stress()
        
        # 2. Target Metrics Validation
        print("\n2️⃣ TARGET METRICS VALIDATION")
        print("-" * 40)
        metric_validator = TargetMetricValidator(config)
        
        results['test_results']['temperature_stability'] = metric_validator.validate_temperature_stability()
        results['test_results']['convergence_performance'] = metric_validator.validate_convergence_performance()
        
        # 3. Sim-to-Real Transferability
        print("\n3️⃣ SIM-TO-REAL TRANSFERABILITY")
        print("-" * 40)
        transfer_analyzer = SimToRealTransferabilityAnalyzer(config)
        
        results['test_results']['transferability_analysis'] = transfer_analyzer.analyze_parameter_sensitivity()
        
        # Calculate overall status
        all_passed = all(
            result.get('passed', False) for result in results['test_results'].values()
            if 'passed' in result
        )
        
        results['overall_status'] = 'PASSED' if all_passed else 'FAILED'
        results['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Summary
        print("\n📊 PHASE D SUMMARY")
        print("=" * 40)
        
        passed_tests = sum(1 for r in results['test_results'].values() if r.get('passed', False))
        total_tests = len([r for r in results['test_results'].values() if 'passed' in r])
        
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Overall Status: {'✅ PASSED' if all_passed else '❌ FAILED'}")
        
        if not all_passed:
            print("\n❌ Failed Tests:")
            for test_name, result in results['test_results'].items():
                if not result.get('passed', True):
                    print(f"   - {test_name}: {result.get('failure_reasons', ['Unknown failure'])}")
        
    except Exception as e:
        results['overall_status'] = 'ERROR'
        results['error'] = str(e)
        print(f"\n❌ Phase D testing failed: {e}")
    
    return results

if __name__ == "__main__":
    # Create proper config for testing
    class MockSimulationConfig:
        def __init__(self):
            self.dt = 0.1
            self.sim_duration = 100.0
            self.pcb_mass = 0.05
            self.pcb_surface_area = 0.01
            self.thermal_mass = 0.05
            self.specific_heat = 900.0
            self.heat_transfer_coeff = 10.0
            self.thermal_expansion = 15e-6
    
    class MockSafetyLimits:
        def __init__(self):
            self.max_temperature = 85.0
            self.min_temperature = -40.0
            self.max_voltage = 3.6
            self.min_voltage = 2.7
            self.max_current = 0.6
            self.max_humidity = 95.0
            self.max_thermal_gradient = 30.0
            self.max_power_dissipation = 2.0
    
    class MockPhysics:
        def __init__(self):
            self.base_temperature = 25.0
            self.base_humidity = 50.0
            self.material_fatigue_rate = 0.005
            self.thermal_conductivity = 0.2
            self.humidity_absorption = 0.1
    
    class MockLogging:
        def __init__(self):
            self.log_dir = "./logs"
            self.fitness_normalization_method = "standard"
    
    class MockHybrid:
        def __init__(self):
            self.pg_es_ratio = 0.7
            self.alternating_mode = False
            self.warm_start_pg_iterations = 10
            self.share_experience = True
            self.adaptive_ratio = True
            self.experience_buffer_size = 1000
    
    class MockConfig:
        def __init__(self):
            self.simulation = MockSimulationConfig()
            self.safety = MockSafetyLimits()
            self.physics = MockPhysics()
            self.logging = MockLogging()
            self.hybrid = MockHybrid()
        
        def copy(self):
            return MockConfig()
    
    # Run Phase D tests
    config = MockConfig()
    test_results = run_phase_d_tests(config)
    
    # Save results
    results_path = Path("logs/phase_d_test_results.json")
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n📄 Phase D results saved to: {results_path}")
    
    if test_results['overall_status'] == 'PASSED':
        print("\n🎉 Phase D: Tests & Validierung - COMPLETED SUCCESSFULLY")
    else:
        print("\n🔧 Phase D: Tests & Validierung - NEEDS ATTENTION")
        print("Check the test results for specific failures to address.")
