#!/usr/bin/env python3
"""
ENERGIE-2.1: Comprehensive Sleep Mode Optimization Test
======================================================

Tests the optimized sleep mode implementation to verify:
1. Fast transition times (< 10ms for wake-up)
2. Proper peripheral shutdown in sleep
3. Reliable status restoration
4. Energy efficiency improvements
"""

import time
import sys
import logging
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.emulation.emulator import RP2040Emulator
    from src.emulation.simple_power_manager import AdaptiveMode
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SleepModeVerificationTest:
    """Comprehensive test suite for optimized sleep mode."""
    
    def __init__(self):
        self.test_results = {
            'transition_times': [],
            'peripheral_tests': [],
            'restoration_tests': [],
            'energy_tests': [],
            'reliability_tests': []
        }
    
    def setup_test_emulator(self) -> RP2040Emulator:
        """Set up emulator for testing."""
        emulator = RP2040Emulator(adaptive_mode=AdaptiveMode.POWER_SAVE)
        
        # Load test firmware
        test_firmware = {
            'path': 'test_sleep_optimization.bin',
            'total_size_bytes': 80 * 1024,
            'model_size_bytes': 40 * 1024,
            'ram_usage_bytes': 30 * 1024,
            'model_input_size': (48, 48)
        }
        emulator.load_firmware(test_firmware)
        
        return emulator
    
    def test_transition_performance(self, emulator: RP2040Emulator, cycles: int = 20) -> Dict:
        """Test sleep/wake transition performance."""
        print(f"\n--- Testing Transition Performance ({cycles} cycles) ---")
        
        sleep_times = []
        wake_times = []
        
        for i in range(cycles):
            # Measure sleep transition
            start_time = time.perf_counter()
            emulator.power_manager.enter_sleep_mode()
            sleep_time = (time.perf_counter() - start_time) * 1000
            sleep_times.append(sleep_time)
            
            # Brief sleep period
            time.sleep(0.01)
            
            # Measure wake transition
            start_time = time.perf_counter()
            emulator.power_manager.wake_up()
            wake_time = (time.perf_counter() - start_time) * 1000
            wake_times.append(wake_time)
            
            print(f"Cycle {i+1:2d}: Sleep {sleep_time:6.3f}ms, Wake {wake_time:6.3f}ms")
        
        results = {
            'sleep_times': sleep_times,
            'wake_times': wake_times,
            'avg_sleep_ms': sum(sleep_times) / len(sleep_times),
            'avg_wake_ms': sum(wake_times) / len(wake_times),
            'max_sleep_ms': max(sleep_times),
            'max_wake_ms': max(wake_times),
            'min_sleep_ms': min(sleep_times),
            'min_wake_ms': min(wake_times),
            'wake_under_10ms': all(t < 10.0 for t in wake_times)
        }
        
        print(f"Sleep - Avg: {results['avg_sleep_ms']:.3f}ms, Max: {results['max_sleep_ms']:.3f}ms")
        print(f"Wake  - Avg: {results['avg_wake_ms']:.3f}ms, Max: {results['max_wake_ms']:.3f}ms")
        print(f"10ms Requirement: {'✓ PASS' if results['wake_under_10ms'] else '✗ FAIL'}")
        
        self.test_results['transition_times'].append(results)
        return results
    
    def test_peripheral_shutdown(self, emulator: RP2040Emulator) -> Dict:
        """Test proper peripheral shutdown in sleep mode."""
        print(f"\n--- Testing Peripheral Shutdown ---")
        
        # Get initial state
        initial_ram = emulator.get_ram_usage()
        
        # Enter sleep mode
        emulator.power_manager.enter_sleep_mode()
        
        # Check peripheral states
        tests = {
            'sleep_mode_active': emulator.sleep_mode,
            'ram_reduced': emulator.get_ram_usage() < initial_ram,
            'camera_state': not getattr(emulator.camera, 'initialized', True),
            'peripheral_states_saved': hasattr(emulator, 'peripheral_states')
        }
        
        sleep_ram = emulator.get_ram_usage()
        ram_reduction_percent = ((initial_ram - sleep_ram) / initial_ram) * 100
        
        print(f"Initial RAM: {initial_ram / 1024:.1f}KB")
        print(f"Sleep RAM:   {sleep_ram / 1024:.1f}KB")
        print(f"Reduction:   {ram_reduction_percent:.1f}%")
        
        for test_name, passed in tests.items():
            print(f"{test_name}: {'✓ PASS' if passed else '✗ FAIL'}")
        
        # Wake up for next test
        emulator.power_manager.wake_up()
        
        results = {
            'tests': tests,
            'initial_ram_kb': initial_ram / 1024,
            'sleep_ram_kb': sleep_ram / 1024,
            'ram_reduction_percent': ram_reduction_percent,
            'all_tests_passed': all(tests.values())
        }
        
        self.test_results['peripheral_tests'].append(results)
        return results
    
    def test_status_restoration(self, emulator: RP2040Emulator) -> Dict:
        """Test complete status restoration after wake-up."""
        print(f"\n--- Testing Status Restoration ---")
        
        # Capture pre-sleep state
        pre_sleep_state = {
            'ram_usage': emulator.get_ram_usage(),
            'sleep_mode': emulator.sleep_mode,
            'temperature': emulator.current_temperature_c,
            'camera_initialized': getattr(emulator.camera, 'initialized', False)
        }
        
        print(f"Pre-sleep RAM: {pre_sleep_state['ram_usage'] / 1024:.1f}KB")
        print(f"Pre-sleep temp: {pre_sleep_state['temperature']:.1f}°C")
        
        # Sleep and wake cycle
        emulator.power_manager.enter_sleep_mode()
        time.sleep(0.05)  # Brief sleep
        emulator.power_manager.wake_up()
        
        # Check restoration
        post_wake_state = {
            'ram_usage': emulator.get_ram_usage(),
            'sleep_mode': emulator.sleep_mode,
            'temperature': emulator.current_temperature_c,
            'camera_initialized': getattr(emulator.camera, 'initialized', False)
        }
        
        tests = {
            'ram_restored': abs(post_wake_state['ram_usage'] - pre_sleep_state['ram_usage']) < 1024,
            'sleep_mode_cleared': not post_wake_state['sleep_mode'],
            'temperature_valid': post_wake_state['temperature'] > 0,
            'camera_restored': post_wake_state['camera_initialized'] == pre_sleep_state['camera_initialized']
        }
        
        print(f"Post-wake RAM: {post_wake_state['ram_usage'] / 1024:.1f}KB")
        print(f"Post-wake temp: {post_wake_state['temperature']:.1f}°C")
        
        for test_name, passed in tests.items():
            print(f"{test_name}: {'✓ PASS' if passed else '✗ FAIL'}")
        
        results = {
            'pre_sleep_state': pre_sleep_state,
            'post_wake_state': post_wake_state,
            'tests': tests,
            'all_tests_passed': all(tests.values())
        }
        
        self.test_results['restoration_tests'].append(results)
        return results
    
    def test_energy_efficiency(self, emulator: RP2040Emulator) -> Dict:
        """Test energy efficiency improvements."""
        print(f"\n--- Testing Energy Efficiency ---")
        
        # Get initial power stats
        initial_stats = emulator.power_manager.get_power_statistics()
        initial_energy = initial_stats.get('energy_consumed_mah', 0)
        
        # Simulate active period
        active_start = time.time()
        time.sleep(0.1)  # 100ms active
        active_duration = time.time() - active_start
        
        # Sleep period
        emulator.power_manager.enter_sleep_mode()
        sleep_start = time.time()
        time.sleep(0.2)  # 200ms sleep
        sleep_duration = time.time() - sleep_start
        emulator.power_manager.wake_up()
        
        # Get final power stats
        final_stats = emulator.power_manager.get_power_statistics()
        final_energy = final_stats.get('energy_consumed_mah', 0)
        
        energy_consumed = final_energy - initial_energy
        
        # Calculate theoretical minimum (sleep mode should use much less power)
        active_power_ma = 80.0  # Typical active current
        sleep_power_ma = 0.5    # Sleep mode current
        
        theoretical_energy = (active_power_ma * active_duration + sleep_power_ma * sleep_duration) / 3600
        
        efficiency_ratio = energy_consumed / theoretical_energy if theoretical_energy > 0 else 1.0
        
        print(f"Active duration: {active_duration * 1000:.1f}ms")
        print(f"Sleep duration: {sleep_duration * 1000:.1f}ms")
        print(f"Energy consumed: {energy_consumed:.6f}mAh")
        print(f"Theoretical min: {theoretical_energy:.6f}mAh")
        print(f"Efficiency ratio: {efficiency_ratio:.2f}")
        
        results = {
            'active_duration_ms': active_duration * 1000,
            'sleep_duration_ms': sleep_duration * 1000,
            'energy_consumed_mah': energy_consumed,
            'theoretical_min_mah': theoretical_energy,
            'efficiency_ratio': efficiency_ratio,
            'efficiency_good': efficiency_ratio < 2.0  # Within 2x of theoretical
        }
        
        self.test_results['energy_tests'].append(results)
        return results
    
    def test_reliability(self, emulator: RP2040Emulator, cycles: int = 50) -> Dict:
        """Test reliability over multiple sleep-wake cycles."""
        print(f"\n--- Testing Reliability ({cycles} cycles) ---")
        
        successful_cycles = 0
        failed_cycles = []
        
        for i in range(cycles):
            try:
                # Sleep-wake cycle
                emulator.power_manager.enter_sleep_mode()
                assert emulator.sleep_mode, f"Sleep mode not active in cycle {i+1}"
                
                time.sleep(0.01)  # Brief sleep
                
                emulator.power_manager.wake_up()
                assert not emulator.sleep_mode, f"Sleep mode not cleared in cycle {i+1}"
                
                successful_cycles += 1
                
            except Exception as e:
                failed_cycles.append(i + 1)
                print(f"Cycle {i+1} failed: {e}")
        
        success_rate = (successful_cycles / cycles) * 100
        
        print(f"Successful cycles: {successful_cycles}/{cycles}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Failed cycles: {failed_cycles[:5]}{'...' if len(failed_cycles) > 5 else ''}")
        
        results = {
            'total_cycles': cycles,
            'successful_cycles': successful_cycles,
            'failed_cycles': failed_cycles,
            'success_rate_percent': success_rate,
            'reliability_good': success_rate >= 98.0
        }
        
        self.test_results['reliability_tests'].append(results)
        return results
    
    def run_comprehensive_test(self) -> Dict:
        """Run all tests and return comprehensive results."""
        print("=" * 60)
        print("ENERGIE-2.1: Comprehensive Sleep Mode Optimization Test")
        print("=" * 60)
        
        emulator = self.setup_test_emulator()
        
        try:
            # Run all test categories
            transition_results = self.test_transition_performance(emulator, cycles=20)
            peripheral_results = self.test_peripheral_shutdown(emulator)
            restoration_results = self.test_status_restoration(emulator)
            energy_results = self.test_energy_efficiency(emulator)
            reliability_results = self.test_reliability(emulator, cycles=50)
            
            # Overall results
            overall_results = {
                'transition_performance': transition_results,
                'peripheral_shutdown': peripheral_results,
                'status_restoration': restoration_results,
                'energy_efficiency': energy_results,
                'reliability': reliability_results,
                'test_timestamp': time.time()
            }
            
            # Check if all requirements are met
            requirements_met = {
                'wake_time_under_10ms': transition_results['wake_under_10ms'],
                'peripheral_shutdown_ok': peripheral_results['all_tests_passed'],
                'status_restoration_ok': restoration_results['all_tests_passed'],
                'energy_efficiency_ok': energy_results['efficiency_good'],
                'reliability_ok': reliability_results['reliability_good']
            }
            
            overall_results['requirements_met'] = requirements_met
            overall_results['all_requirements_met'] = all(requirements_met.values())
            
            return overall_results
            
        finally:
            emulator.close()
    
    def print_final_report(self, results: Dict) -> None:
        """Print comprehensive final report."""
        print("\n" + "=" * 60)
        print("FINAL TEST REPORT")
        print("=" * 60)
        
        # Requirements summary
        requirements = results['requirements_met']
        print("REQUIREMENT VERIFICATION:")
        for req, met in requirements.items():
            status = "✓ PASS" if met else "✗ FAIL"
            print(f"  {req.replace('_', ' ').title()}: {status}")
        
        print(f"\nOVERALL STATUS: {'✓ ALL REQUIREMENTS MET' if results['all_requirements_met'] else '⚠ OPTIMIZATION NEEDED'}")
        
        # Performance summary
        transition = results['transition_performance']
        print(f"\nPERFORMANCE SUMMARY:")
        print(f"  Average wake time: {transition['avg_wake_ms']:.3f}ms")
        print(f"  Maximum wake time: {transition['max_wake_ms']:.3f}ms")
        print(f"  Success rate: {results['reliability']['success_rate_percent']:.1f}%")
        
        if results['all_requirements_met']:
            print(f"\n🎉 ENERGIE-2.1 SUCCESSFULLY COMPLETED!")
            print(f"Sleep mode implementation is optimized and meets all performance criteria.")
        else:
            print(f"\n⚠️  FURTHER OPTIMIZATION REQUIRED")
            print(f"Some requirements not met. Check individual test results above.")


def main():
    """Main test execution function."""
    test_suite = SleepModeVerificationTest()
    
    try:
        results = test_suite.run_comprehensive_test()
        test_suite.print_final_report(results)
        
        # Save results
        output_dir = Path("output/energie_2_1_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        results_file = output_dir / "comprehensive_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        return results['all_requirements_met']
        
    except Exception as e:
        print(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
