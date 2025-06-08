#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script for Spatial-Aware Augmentation Pipeline
SPATIAL-5.2: Dataset Augmentation mit räumlichen Features - Testing & Validation

This script validates the functionality of the spatial-aware augmentation pipeline
by testing individual components and the complete pipeline with real pizza images.

Author: GitHub Copilot (2025-06-07)
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "scripts"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'spatial_augmentation_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test if all required modules can be imported correctly."""
    try:
        logger.info("Testing imports...")
        
        # Test spatial augmentation import
        from spatial_aware_augmentation import (
            SpatialAugmentationConfig,
            SpatialAugmentationResult,
            Spatial3DAwareTransforms,
            SpatialFeatureGuidedAugmentation,
            SpatialAwareAugmentationPipeline
        )
        logger.info("✓ Spatial augmentation modules imported successfully")
        
        # Test supporting modules
        from spatial_preprocessing import SpatialPreprocessingPipeline
        logger.info("✓ Spatial preprocessing module imported successfully")
        
        from multi_frame_spatial_analysis import MultiFrameSpatialAnalyzer
        logger.info("✓ Multi-frame spatial analysis module imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error during import: {e}")
        return False

def test_spatial_transforms():
    """Test individual spatial transformation functions."""
    try:
        logger.info("Testing spatial transforms...")
        
        from spatial_aware_augmentation import (
            Spatial3DAwareTransforms,
            SpatialAugmentationConfig
        )
        
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_depth = np.random.rand(224, 224)
        test_spatial_data = {
            'depth_map': test_depth,
            'surface_normals': np.random.rand(224, 224, 3),
            'roughness': np.random.rand(224, 224),
            'edges': np.random.rand(224, 224) > 0.5
        }
        
        config = SpatialAugmentationConfig()
        transforms = Spatial3DAwareTransforms(config)
        
        # Test 3D perspective transformation
        result = transforms.apply_3d_perspective_transform(test_image, test_spatial_data)
        assert result is not None and result.shape == test_image.shape
        logger.info("✓ 3D perspective transformation test passed")
        
        # Test depth-based lighting
        result = transforms.apply_depth_based_lighting(test_image, test_spatial_data)
        assert result is not None and result.shape == test_image.shape
        logger.info("✓ Depth-based lighting test passed")
        
        # Test surface deformation
        result = transforms.apply_surface_deformation(test_image, test_spatial_data)
        assert result is not None and result.shape == test_image.shape
        logger.info("✓ Surface deformation test passed")
        
        # Test volumetric texture mapping
        result = transforms.apply_volumetric_texture_mapping(test_image, test_spatial_data)
        assert result is not None and result.shape == test_image.shape
        logger.info("✓ Volumetric texture mapping test passed")
        
        # Test geometric reshaping
        result = transforms.apply_geometric_reshaping(test_image, test_spatial_data)
        assert result is not None and result.shape == test_image.shape
        logger.info("✓ Geometric reshaping test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Spatial transforms test failed: {e}")
        return False

def test_feature_guided_augmentation():
    """Test spatial feature-guided augmentation selection."""
    try:
        logger.info("Testing feature-guided augmentation...")
        
        from spatial_aware_augmentation import (
            SpatialFeatureGuidedAugmentation,
            SpatialAugmentationConfig
        )
        
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        config = SpatialAugmentationConfig()
        feature_guide = SpatialFeatureGuidedAugmentation(config)
        
        # Test spatial feature extraction
        features = feature_guide.extract_spatial_guidance_features(test_image)
        assert isinstance(features, dict)
        assert 'depth_regions' in features
        assert 'texture_complexity' in features
        logger.info("✓ Spatial feature extraction test passed")
        
        # Test augmentation strategy selection
        strategy = feature_guide.select_augmentation_strategy(features)
        assert isinstance(strategy, dict)
        assert 'transforms' in strategy
        assert 'weights' in strategy
        logger.info("✓ Augmentation strategy selection test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Feature-guided augmentation test failed: {e}")
        return False

def test_complete_pipeline_with_real_image():
    """Test the complete augmentation pipeline with a real pizza image."""
    try:
        logger.info("Testing complete pipeline with real image...")
        
        from spatial_aware_augmentation import SpatialAwareAugmentationPipeline
        
        # Load test image
        test_image_path = project_root / 'data' / 'test' / 'sample_pizza_image.jpg'
        if not test_image_path.exists():
            # Find any image in the test directory
            test_dir = project_root / 'data' / 'test'
            image_files = list(test_dir.rglob('*.jpg')) + list(test_dir.rglob('*.png'))
            if not image_files:
                logger.warning("No test images found, using synthetic image")
                test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                Image.fromarray(test_image).save(project_root / 'temp_test_image.jpg')
                test_image_path = project_root / 'temp_test_image.jpg'
            else:
                test_image_path = image_files[0]
        
        pipeline = SpatialAwareAugmentationPipeline()
        
        # Test single image augmentation
        result = pipeline.augment_single_image(str(test_image_path))
        assert result is not None
        assert result.success
        assert result.augmented_image is not None
        logger.info("✓ Single image augmentation test passed")
        
        # Test quality evaluation
        if result.quality_score is not None:
            logger.info(f"✓ Quality evaluation completed: {result.quality_score:.3f}")
        else:
            logger.info("✓ Quality evaluation skipped (Spatial-MLLM not available)")
        
        # Save test results
        output_dir = project_root / 'temp_output'
        output_dir.mkdir(exist_ok=True)
        
        if result.augmented_image is not None:
            cv2.imwrite(str(output_dir / 'test_augmented.jpg'), result.augmented_image)
            logger.info(f"✓ Test result saved to {output_dir / 'test_augmented.jpg'}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Complete pipeline test failed: {e}")
        return False

def test_batch_processing():
    """Test batch processing capabilities."""
    try:
        logger.info("Testing batch processing...")
        
        from spatial_aware_augmentation import SpatialAwareAugmentationPipeline
        
        # Create test images directory
        test_dir = project_root / 'temp_test_batch'
        test_dir.mkdir(exist_ok=True)
        
        # Create some test images
        for i in range(3):
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(test_image).save(test_dir / f'test_image_{i}.jpg')
        
        pipeline = SpatialAwareAugmentationPipeline()
        
        # Test dataset processing
        output_dir = project_root / 'temp_output_batch'
        results = pipeline.process_dataset(
            str(test_dir),
            str(output_dir),
            num_augmentations_per_image=2
        )
        
        assert len(results) > 0
        logger.info(f"✓ Batch processing completed: {len(results)} results")
        
        # Clean up
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Batch processing test failed: {e}")
        return False

def create_visual_test_report():
    """Create a visual test report showing augmentation results."""
    try:
        logger.info("Creating visual test report...")
        
        from spatial_aware_augmentation import SpatialAwareAugmentationPipeline
        
        # Load a test image
        test_image_path = project_root / 'data' / 'test' / 'sample_pizza_image.jpg'
        if not test_image_path.exists():
            test_dir = project_root / 'data' / 'test'
            image_files = list(test_dir.rglob('*.jpg')) + list(test_dir.rglob('*.png'))
            if image_files:
                test_image_path = image_files[0]
            else:
                logger.warning("No test images found for visual report")
                return False
        
        pipeline = SpatialAwareAugmentationPipeline()
        
        # Create multiple augmentations
        augmentations = []
        original_image = cv2.imread(str(test_image_path))
        augmentations.append(('Original', original_image))
        
        for i in range(4):
            result = pipeline.augment_single_image(str(test_image_path))
            if result and result.success and result.augmented_image is not None:
                augmentations.append((f'Augmentation {i+1}', result.augmented_image))
        
        # Create visual comparison
        fig, axes = plt.subplots(1, len(augmentations), figsize=(15, 3))
        if len(augmentations) == 1:
            axes = [axes]
        
        for i, (title, image) in enumerate(augmentations):
            if i < len(axes):
                # Convert BGR to RGB for matplotlib
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
                axes[i].imshow(image_rgb)
                axes[i].set_title(title)
                axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save visual report
        output_path = project_root / 'temp_output' / 'spatial_augmentation_test_report.png'
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Visual test report saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Visual test report creation failed: {e}")
        return False

def main():
    """Run all tests for the spatial augmentation pipeline."""
    logger.info("Starting SPATIAL-5.2 Augmentation Pipeline Tests")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Create output directory
    (project_root / 'temp_output').mkdir(exist_ok=True)
    
    # Run tests
    test_results['imports'] = test_basic_imports()
    test_results['transforms'] = test_spatial_transforms()
    test_results['feature_guided'] = test_feature_guided_augmentation()
    test_results['complete_pipeline'] = test_complete_pipeline_with_real_image()
    test_results['batch_processing'] = test_batch_processing()
    test_results['visual_report'] = create_visual_test_report()
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST RESULTS SUMMARY:")
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"  {test_name:20s}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Spatial augmentation pipeline is ready.")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed. Please review the issues.")
        return False

if __name__ == "__main__":
    main()
