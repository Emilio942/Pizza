#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified RL Integration Test

Quick integration test to validate the RL system works correctly
with basic functionality testing.
"""

import sys
import os
from pathlib import Path
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def test_rl_environment_basic():
    """Test basic RL environment functionality."""
    try:
        logger.info("Testing RL Environment Basic Functionality...")
        
        # Import and create environment
        sys.path.append(str(project_root))
        sys.path.append(str(project_root / "src"))
        from src.rl.pizza_rl_environment import PizzaRLEnvironment
        
        # Create environment with mock components
        env = PizzaRLEnvironment(
            max_steps_per_episode=5
        )
        
        # Load dummy firmware if emulator is present
        if hasattr(env, 'emulator') and hasattr(env.emulator, 'load_firmware'):
             env.emulator.load_firmware({
                'path': 'test_firmware.bin',
                'total_size_bytes': 1024,
                'model_size_bytes': 512,
                'ram_usage_bytes': 256,
                'model_input_size': (48, 48)
            })
        
        logger.info("✓ Environment created successfully")
        
        # Test basic operations
        obs, info = env.reset()
        logger.info(f"✓ Environment reset successful, obs shape: {len(obs)}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        logger.info(f"✓ Environment step successful")
        logger.info(f"  - Observation shape: {len(obs)}")
        logger.info(f"  - Reward: {reward:.3f}")
        logger.info(f"  - Terminated: {terminated}")
        logger.info(f"  - Info keys: {list(info.keys())}")
        
        # Test multiple steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
        
        logger.info("✓ Multiple steps completed successfully")
        
        env.close()
        logger.info("✓ Environment closed successfully")
        
    except Exception as e:
        logger.error(f"✗ RL Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        raise e


def test_ppo_agent_basic():
    """Test basic PPO agent functionality."""
    try:
        logger.info("Testing PPO Agent Basic Functionality...")
        
        from src.rl.pizza_rl_environment import PizzaRLEnvironment
        from src.rl.ppo_pizza_agent import PPOPizzaAgent, PPOHyperparameters
        
        # Create environment
        env = PizzaRLEnvironment(
            max_steps_per_episode=3
        )
        
        # Load dummy firmware to avoid HardwareError during inference
        firmware_data = {
            "path": "test_firmware.bin",
            "total_size_bytes": 102400,
            "model_size_bytes": 51200,
            "ram_usage_bytes": 40960,
            "model_input_size": (48, 48)
        }
        env.emulator.load_firmware(firmware_data)
        
        # Create agent with minimal hyperparameters
        hyperparams = PPOHyperparameters(
            learning_rate=3e-4,
            batch_size=16,
            ppo_epochs=1,
            rollout_steps=32,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01
        )
        
        agent = PPOPizzaAgent(
            environment=env,
            hyperparams=hyperparams,
            device='cpu'
        )
        
        logger.info("✓ PPO Agent created successfully")
        
        # Test rollout collection
        rollout_data = agent.collect_rollout()
        logger.info(f"✓ Rollout collected: {len(rollout_data.get('rewards', []))} steps")
        
        # Test policy update
        update_info = agent.update_policy()
        logger.info(f"✓ Policy updated successfully")
        logger.info(f"  - Policy loss: {update_info['policy_loss']:.4f}")
        logger.info(f"  - Value loss: {update_info['value_loss']:.4f}")
        
        env.close()
        logger.info("✓ PPO Agent test completed successfully")
        
    except Exception as e:
        logger.error(f"✗ PPO Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        raise e


def test_integration_layers():
    """Test integration layer functionality."""
    try:
        logger.info("Testing Integration Layers...")
        
        from src.integration.compatibility import (
            ModelCompatibilityManager, 
            get_compatibility_manager
        )
        from src.verification.pizza_verifier import PizzaVerifier
        from src.rl.adaptive_pizza_policy import (
            SystemState, 
            InferenceStrategy,
            ModelVariant,
            ProcessingIntensity
        )
        
        # Test compatibility manager
        manager = get_compatibility_manager()
        logger.info("✓ Compatibility manager created")
        
        # Test model compatibility check
        models = ["MicroPizzaNet", "MicroPizzaNetV2"]
        for model in models:
            compatible = manager.check_model_compatibility(model)
            logger.info(f"✓ {model} compatibility: {compatible}")
        
        # Test verifier (mock mode)
        verifier = PizzaVerifier(device='cpu')
        logger.info("✓ Verifier created")
        
        # Test system state creation
        system_state = SystemState(
            battery_level=0.75,
            power_draw_current=12.0,
            energy_budget=100.0,
            temperature=30.0,
            memory_usage=0.3,
            processing_load=0.4,
            image_complexity=0.6,
            brightness_level=0.5,
            contrast_level=0.5,
            has_motion_blur=False,
            required_accuracy=0.8,
            time_constraints=100.0,
            food_safety_critical=False
        )
        logger.info("✓ System state created")
        
        # Test inference strategy
        strategy = InferenceStrategy(
            model_variant=ModelVariant.MICRO_PIZZA_NET,
            processing_intensity=ProcessingIntensity.ENHANCED,
            use_cmsis_nn=True,
            enable_temporal_smoothing=True,
            confidence_threshold=0.8,
            preprocessing_options={}
        )
        logger.info("✓ Inference strategy created")
        
        logger.info("✓ Integration layers test completed successfully")
        
    except Exception as e:
        logger.error(f"✗ Integration layers test failed: {e}")
        import traceback
        traceback.print_exc()
        raise e


def test_image_complexity_analyzer():
    """Test image complexity analysis."""
    try:
        logger.info("Testing Image Complexity Analyzer...")
        
        from src.rl.pizza_rl_environment import ImageComplexityAnalyzer
        
        # Create test images
        test_images = [
            np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8),  # Random noise
            np.ones((48, 48, 3), dtype=np.uint8) * 128,              # Uniform gray
            np.zeros((48, 48, 3), dtype=np.uint8)                    # Black image
        ]
        
        analyzer = ImageComplexityAnalyzer()
        
        for i, image in enumerate(test_images):
            complexity = analyzer.analyze_complexity(image)
            logger.info(f"✓ Image {i+1} complexity: {complexity['overall_complexity']:.3f}")
            
            # Verify structure
            required_keys = ['edge_density', 'texture_complexity', 'color_complexity', 'overall_complexity']
            for key in required_keys:
                assert key in complexity, f"Missing key: {key}"
                assert 0.0 <= complexity[key] <= 1.0, f"Invalid range for {key}: {complexity[key]}"
        
        logger.info("✓ Image complexity analyzer test completed successfully")
        
    except Exception as e:
        logger.error(f"✗ Image complexity analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        raise e



