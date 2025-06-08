#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aufgabe 3.2 Integration and Testing: Comprehensive RL Integration Tests

Integration tests for the Pizza RL environment and PPO agent with existing infrastructure:
- RP2040 emulator integration
- Pizza verifier integration  
- Performance validation
- Real system integration
- Hyperparameter validation

Author: GitHub Copilot
"""

import os
import sys
import unittest
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import RL components
from src.rl.pizza_rl_environment import PizzaRLEnvironment, PizzaTask, EnvironmentState, ImageComplexityAnalyzer
from src.rl.ppo_pizza_agent import PPOPizzaAgent, PPOHyperparameters, PPOBuffer
from src.rl.adaptive_pizza_policy import (
    AdaptivePizzaRecognitionPolicy, SystemState, InferenceStrategy,
    ModelVariant, ProcessingIntensity
)

# Import integration layers
from src.integration.compatibility import (
    ModelCompatibilityManager, VerifierIntegration, RLIntegration,
    get_compatibility_manager, get_verifier_integration, get_rl_integration
)

# Import existing infrastructure
from src.verification.pizza_verifier import PizzaVerifier, VerifierData
from src.pizza_detector import MicroPizzaNet, MicroPizzaNetV2
from src.constants import CLASS_NAMES

# Import emulator if available
try:
    from src.emulation.emulator import RP2040Emulator
    EMULATOR_AVAILABLE = True
except ImportError:
    EMULATOR_AVAILABLE = False
    print("RP2040 Emulator not available for integration testing")

# Import logging system if available
try:
    from src.emulation.logging_system import PerformanceLogger
    LOGGING_SYSTEM_AVAILABLE = True
except ImportError:
    LOGGING_SYSTEM_AVAILABLE = False

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRLEnvironmentIntegration(unittest.TestCase):
    """Test integration of RL environment with existing systems."""
     def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_image_shape = (48, 48, 3)
        self.test_images = [np.random.randint(0, 256, self.test_image_shape, dtype=np.uint8) for _ in range(5)]

        # Create test environment with proper parameters
        self.env = PizzaRLEnvironment(
            max_steps_per_episode=10,
            battery_capacity_mah=1500.0,
            initial_battery_level=0.8,
            task_dataset_path=None,
            device='cpu',
            enable_logging=False
        )
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'env'):
            self.env.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_environment_basic_functionality(self):
        """Test basic environment operations."""
        logger.info("Testing basic environment functionality...")
        
        # Test initialization
        self.assertIsNotNone(self.env.action_space)
        self.assertIsNotNone(self.env.observation_space)
        
        # Test reset
        obs, info = self.env.reset()
        self.assertEqual(len(obs), 13)  # SystemState has 13 dimensions
        self.assertIsInstance(info, dict)
        
        # Test step
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.assertEqual(len(obs), 13)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
        
        logger.info("✓ Basic environment functionality working")
    
    def test_image_complexity_analyzer(self):
        """Test image complexity analysis."""
        logger.info("Testing image complexity analyzer...")
        
        analyzer = ImageComplexityAnalyzer()
        
        for test_image in self.test_images:
            complexity = analyzer.analyze_complexity(test_image)
            
            self.assertIsInstance(complexity, dict)
            self.assertIn('edge_density', complexity)
            self.assertIn('texture_complexity', complexity)
            self.assertIn('color_complexity', complexity)
            self.assertIn('overall_complexity', complexity)
            
            # Check ranges
            self.assertGreaterEqual(complexity['overall_complexity'], 0.0)
            self.assertLessEqual(complexity['overall_complexity'], 1.0)
        
        logger.info("✓ Image complexity analyzer working")
    
    def test_multi_objective_reward_calculation(self):
        """Test multi-objective reward system."""
        logger.info("Testing multi-objective reward calculation...")
        
        # Create test scenario
        task = PizzaTask(
            image=self.test_images[0],
            ground_truth_class="baked",
            required_accuracy=0.8,
            time_constraint_ms=100.0,
            safety_critical=True
        )
        
        # Test different prediction scenarios using image parameter
        test_cases = [
            {"predicted_class": "baked", "confidence": 0.9, "processing_time": 50.0, "energy_used": 10.0},
            {"predicted_class": "burnt", "confidence": 0.7, "processing_time": 80.0, "energy_used": 15.0},
            {"predicted_class": "raw", "confidence": 0.5, "processing_time": 120.0, "energy_used": 20.0},
        ]
        
        for case in test_cases:
            # Create task for this test case
            test_task = PizzaTask(
                image_path="test_image.jpg",
                image_data=self.test_images[0],
                required_accuracy=0.8,
                max_inference_time_ms=100.0,
                food_safety_critical=True,
                ground_truth_class=0  # "baked" class
            )
            
            reward = self.env._calculate_multi_objective_reward(
                task=test_task,
                predicted_class=case["predicted_class"],
                confidence=case["confidence"],
                processing_time_ms=case["processing_time"],
                energy_consumed_mj=case["energy_used"]
            )
            
            self.assertIsInstance(reward, (int, float))
            self.assertGreaterEqual(reward, -1.0)  # Minimum possible reward
            self.assertLessEqual(reward, 1.0)      # Maximum possible reward
            
            logger.info(f"Reward for {case['predicted_class']}: {reward:.3f}")
        
        logger.info("✓ Multi-objective reward calculation working")
    
    @unittest.skipUnless(EMULATOR_AVAILABLE, "RP2040 Emulator not available")
    def test_real_emulator_integration(self):
        """Test integration with real RP2040 emulator."""
        logger.info("Testing real RP2040 emulator integration...")
        
        # Create environment with real emulator
        env_real = PizzaRLEnvironment(
            use_real_emulator=True,
            use_real_verifier=False,
            max_episodes=5
        )
        
        try:
            # Test that emulator is properly initialized
            self.assertIsNotNone(env_real.emulator)
            self.assertIsInstance(env_real.emulator, RP2040Emulator)
            
            # Test environment operations with real emulator
            obs, info = env_real.reset()
            action = env_real.action_space.sample()
            obs, reward, terminated, truncated, info = env_real.step(action)
            
            # Check that system state reflects real emulator data
            system_state = env_real._create_system_state()
            self.assertIsInstance(system_state.battery_level, float)
            self.assertIsInstance(system_state.temperature, float)
            
            logger.info("✓ Real emulator integration working")
            
        finally:
            env_real.close()
    
    def test_verifier_integration(self):
        """Test integration with pizza verifier system."""
        logger.info("Testing verifier integration...")
        
        # Create mock verifier data
        verifier_data = VerifierData(
            pizza_image_path=str(self.temp_dir / "test_pizza.jpg"),
            model_prediction="baked",
            ground_truth_class="baked",
            confidence_score=0.85,
            quality_score=0.9
        )
        
        # Test verifier integration layer
        compatibility_manager = ModelCompatibilityManager()
        mock_verifier = PizzaVerifier(device='cpu')
        verifier_integration = VerifierIntegration(
            verifier=mock_verifier,
            compatibility_manager=compatibility_manager
        )
        
        # Test verifier data creation from prediction
        prediction_result = {
            'predicted_class': 'baked',
            'confidence': 0.85,
            'is_correct': True
        }
        
        created_data = verifier_integration.create_verifier_data_from_prediction(
            image_path=str(self.temp_dir / "test_pizza.jpg"),
            prediction_result=prediction_result,
            ground_truth="baked"
        )
        
        self.assertIsInstance(created_data, VerifierData)
        self.assertEqual(created_data.model_prediction, 'baked')
        self.assertEqual(created_data.ground_truth_class, 'baked')
        
        logger.info("✓ Verifier integration working")


class TestPPOAgentIntegration(unittest.TestCase):
    """Test integration of PPO agent with RL environment."""
     def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create environment and agent with proper parameters
        self.env = PizzaRLEnvironment(
            max_steps_per_episode=5,
            battery_capacity_mah=1500.0,
            initial_battery_level=0.8,
            task_dataset_path=None,
            device='cpu',
            enable_logging=False
        )
        
        # Create agent with test-friendly hyperparameters
        self.hyperparams = PPOHyperparameters(
            learning_rate=3e-4,
            batch_size=32,
            n_epochs=2,
            n_steps=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            target_kl=0.01
        )
        
        self.agent = PPOPizzaAgent(
            env=self.env,
            hyperparams=self.hyperparams,
            device='cpu'
        )
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'env'):
            self.env.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_agent_initialization(self):
        """Test agent initialization and components."""
        logger.info("Testing PPO agent initialization...")
        
        # Check that all components are properly initialized
        self.assertIsNotNone(self.agent.policy)
        self.assertIsNotNone(self.agent.optimizer)
        self.assertIsNotNone(self.agent.buffer)
        
        # Check policy architecture
        self.assertIsInstance(self.agent.policy, AdaptivePizzaRecognitionPolicy)
        
        # Check buffer
        self.assertIsInstance(self.agent.buffer, PPOBuffer)
        
        logger.info("✓ PPO agent initialization working")
    
    def test_rollout_collection(self):
        """Test rollout data collection."""
        logger.info("Testing rollout collection...")
        
        # Collect a small rollout
        rollout_data = self.agent.collect_rollouts(n_steps=10)
        
        self.assertIsInstance(rollout_data, dict)
        self.assertIn('observations', rollout_data)
        self.assertIn('actions', rollout_data)
        self.assertIn('rewards', rollout_data)
        self.assertIn('values', rollout_data)
        self.assertIn('log_probs', rollout_data)
        
        # Check data shapes
        self.assertEqual(len(rollout_data['observations']), 10)
        self.assertEqual(len(rollout_data['actions']), 10)
        self.assertEqual(len(rollout_data['rewards']), 10)
        
        logger.info("✓ Rollout collection working")
    
    def test_policy_update(self):
        """Test PPO policy update mechanism."""
        logger.info("Testing policy update...")
        
        # Collect some data
        rollout_data = self.agent.collect_rollouts(n_steps=32)
        
        # Perform policy update
        update_info = self.agent.update_policy(rollout_data)
        
        self.assertIsInstance(update_info, dict)
        self.assertIn('policy_loss', update_info)
        self.assertIn('value_loss', update_info)
        self.assertIn('entropy_loss', update_info)
        self.assertIn('approx_kl', update_info)
        
        # Check that losses are reasonable numbers
        for loss_name, loss_value in update_info.items():
            if 'loss' in loss_name or 'kl' in loss_name:
                self.assertIsInstance(loss_value, (int, float))
                self.assertNotEqual(loss_value, float('inf'))
                self.assertNotEqual(loss_value, float('-inf'))
        
        logger.info("✓ Policy update working")
    
    def test_training_loop_integration(self):
        """Test complete training loop integration."""
        logger.info("Testing training loop integration...")
        
        # Run a very short training loop
        training_info = self.agent.train(
            total_timesteps=100,
            eval_episodes=2,
            save_checkpoint_every=None,  # Don't save during tests
            log_interval=50
        )
        
        self.assertIsInstance(training_info, dict)
        self.assertIn('total_timesteps', training_info)
        self.assertIn('training_episodes', training_info)
        self.assertIn('final_evaluation', training_info)
        
        # Check evaluation results
        eval_results = training_info['final_evaluation']
        self.assertIn('mean_reward', eval_results)
        self.assertIn('mean_episode_length', eval_results)
        
        logger.info("✓ Training loop integration working")


class TestRLIntegrationLayer(unittest.TestCase):
    """Test the RL integration layer with existing systems."""
    
    def setUp(self):
        """Set up test environment."""
        self.compatibility_manager = get_compatibility_manager()
        self.verifier = PizzaVerifier(device='cpu')
        self.policy = AdaptivePizzaRecognitionPolicy()
        
        self.rl_integration = get_rl_integration(
            policy=self.policy,
            verifier=self.verifier
        )
    
    def test_system_state_creation(self):
        """Test system state creation from context."""
        logger.info("Testing system state creation...")
        
        system_state = self.rl_integration.create_system_state_from_context(
            battery_level=0.75,
            image_complexity=0.6,
            required_accuracy=0.85,
            time_constraints=100.0,
            temperature=35.0,
            memory_usage=0.4
        )
        
        self.assertIsInstance(system_state, SystemState)
        self.assertEqual(system_state.battery_level, 0.75)
        self.assertEqual(system_state.image_complexity, 0.6)
        self.assertEqual(system_state.required_accuracy, 0.85)
        
        logger.info("✓ System state creation working")
    
    def test_inference_strategy_optimization(self):
        """Test inference strategy optimization."""
        logger.info("Testing inference strategy optimization...")
        
        # Create test system state with correct parameters
        system_state = SystemState(
            battery_level=0.5,
            power_draw_current=15.0,
            energy_budget=50.0,
            image_complexity=0.7,
            brightness_level=0.6,
            contrast_level=0.8,
            has_motion_blur=False,
            required_accuracy=0.8,
            time_constraints=100.0,
            food_safety_critical=True,
            temperature=30.0,
            memory_usage=0.3,
            processing_load=0.4
        )
        
        available_models = ["MicroPizzaNet", "MicroPizzaNetV2"]
        
        strategy = self.rl_integration.optimize_inference_strategy(
            system_state=system_state,
            available_models=available_models
        )
        
        self.assertIsInstance(strategy, InferenceStrategy)
        self.assertIn(strategy.model_selection, available_models)
        
        logger.info("✓ Inference strategy optimization working")
    
    def test_strategy_performance_estimation(self):
        """Test strategy performance estimation."""
        logger.info("Testing strategy performance estimation...")
        
        # Create test strategy with correct parameters
        strategy = InferenceStrategy(
            model_variant=ModelVariant.MICRO_PIZZA_NET,
            processing_intensity=ProcessingIntensity.STANDARD,
            use_cmsis_nn=True,
            confidence_threshold=0.8,
            enable_temporal_smoothing=True,
            preprocessing_options={}
        )
        
        performance = self.rl_integration.estimate_strategy_performance(
            strategy=strategy,
            image_complexity=0.6
        )
        
        self.assertIsInstance(performance, dict)
        self.assertIn('estimated_accuracy', performance)
        self.assertIn('estimated_energy_mj', performance)
        self.assertIn('estimated_latency_ms', performance)
        
        # Check reasonable ranges
        self.assertGreaterEqual(performance['estimated_accuracy'], 0.0)
        self.assertLessEqual(performance['estimated_accuracy'], 1.0)
        self.assertGreaterEqual(performance['estimated_energy_mj'], 0.0)
        self.assertGreaterEqual(performance['estimated_latency_ms'], 0.0)
        
        logger.info("✓ Strategy performance estimation working")


class TestPerformanceValidation(unittest.TestCase):
    """Test performance characteristics of the RL system."""
    
    def setUp(self):
        """Set up performance test environment."""
        self.env = PizzaRLEnvironment(
            max_steps_per_episode=10,
            battery_capacity_mah=1500.0,
            initial_battery_level=0.8,
            task_dataset_path=None,
            device='cpu',
            enable_logging=False
        )
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'env'):
            self.env.close()
    
    def test_environment_step_performance(self):
        """Test environment step performance."""
        logger.info("Testing environment step performance...")
        
        self.env.reset()
        
        # Measure step time
        step_times = []
        for _ in range(100):
            action = self.env.action_space.sample()
            start_time = time.time()
            self.env.step(action)
            step_time = time.time() - start_time
            step_times.append(step_time)
        
        mean_step_time = np.mean(step_times)
        max_step_time = np.max(step_times)
        
        # Check performance requirements (should be fast for real-time usage)
        self.assertLess(mean_step_time, 0.1, "Mean step time should be less than 100ms")
        self.assertLess(max_step_time, 0.5, "Max step time should be less than 500ms")
        
        logger.info(f"✓ Environment performance: mean={mean_step_time*1000:.1f}ms, max={max_step_time*1000:.1f}ms")
    
    def test_memory_usage_stability(self):
        """Test memory usage stability during operation."""
        logger.info("Testing memory usage stability...")
        
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run environment for multiple episodes
        for episode in range(10):
            obs, info = self.env.reset()
            for step in range(50):
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)
                if terminated or truncated:
                    break
        
        # Check memory after operations
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        # Memory increase should be reasonable (less than 100MB for test)
        self.assertLess(memory_increase, 100, f"Memory increased by {memory_increase:.1f}MB")
        
        logger.info(f"✓ Memory usage stable: baseline={baseline_memory:.1f}MB, final={final_memory:.1f}MB")


class TestErrorHandlingAndRobustness(unittest.TestCase):
    """Test error handling and robustness of the RL system."""
    
    def test_invalid_action_handling(self):
        """Test handling of invalid actions."""
        logger.info("Testing invalid action handling...")
        
        env = PizzaRLEnvironment(
            max_steps_per_episode=5,
            battery_capacity_mah=1500.0,
            initial_battery_level=0.8,
            task_dataset_path=None,
            device='cpu',
            enable_logging=False
        )
        
        try:
            env.reset()
            
            # Test with out-of-bounds actions
            invalid_actions = [
                [-1, 0, 0, 0, 0],  # Negative model selection
                [10, 0, 0, 0, 0],  # Model selection too high
                [0, -1, 0, 0, 0],  # Negative processing intensity
                [0, 5, 0, 0, 0],   # Processing intensity too high
            ]
            
            for invalid_action in invalid_actions:
                # Should handle gracefully without crashing
                try:
                    obs, reward, terminated, truncated, info = env.step(invalid_action)
                    # If it doesn't crash, that's good
                except ValueError as e:
                    # Expected for invalid actions
                    self.assertIn("invalid", str(e).lower())
                
        finally:
            env.close()
        
        logger.info("✓ Invalid action handling working")
    
    def test_missing_component_fallbacks(self):
        """Test fallbacks when components are missing."""
        logger.info("Testing missing component fallbacks...")
        
        # Test with missing emulator (should use mock)
        env_no_emulator = PizzaRLEnvironment(
            max_steps_per_episode=5,
            battery_capacity_mah=1500.0,
            initial_battery_level=0.8,
            task_dataset_path=None,
            device='cpu',
            enable_logging=False
        )
        
        try:
            # Should work with mock components
            obs, info = env_no_emulator.reset()
            action = env_no_emulator.action_space.sample()
            obs, reward, terminated, truncated, info = env_no_emulator.step(action)
            
            self.assertIsNotNone(obs)
            self.assertIsInstance(reward, (int, float))
            
        finally:
            env_no_emulator.close()
        
        logger.info("✓ Missing component fallbacks working")


def run_integration_tests():
    """Run all integration tests and generate report."""
    logger.info("Starting RL Pizza Integration Tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestRLEnvironmentIntegration,
        TestPPOAgentIntegration,
        TestRLIntegrationLayer,
        TestPerformanceValidation,
        TestErrorHandlingAndRobustness
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate summary
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    successes = total_tests - failures - errors
    
    logger.info(f"\n{'='*60}")
    logger.info(f"RL PIZZA INTEGRATION TEST SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Successful: {successes}")
    logger.info(f"Failures: {failures}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Success Rate: {successes/total_tests*100:.1f}%")
    
    if failures > 0:
        logger.info(f"\nFAILURES:")
        for test, traceback in result.failures:
            logger.info(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if errors > 0:
        logger.info(f"\nERRORS:")
        for test, traceback in result.errors:
            logger.info(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    logger.info(f"{'='*60}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
