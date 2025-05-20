# Standard Augmentation Pipeline Documentation

## Overview

This document describes the standard augmentation pipeline implemented in the pizza classification project. The augmentation pipeline applies a series of transformations to training images to increase model robustness and generalization across different lighting conditions, angles, and pizza variations.

## Implementation Details

The standard augmentation pipeline is implemented in `scripts/standard_augmentation.py` and provides a comprehensive set of augmentation techniques, each with configurable parameters and probabilities.

### Key Features

- **Modular Design**: Selectively enable/disable categories of augmentations
- **Configurable Intensity**: Low, medium, and high intensity presets
- **PyTorch Integration**: Optimized for use with PyTorch dataloaders
- **Pizza-Specific Augmentations**: Special transformations for pizza images (burning effects, etc.)
- **Probabilistic Application**: Each transformation has an associated probability

## Augmentation Categories

The pipeline includes the following categories of augmentations:

### 1. Geometric Transformations

Transformations that alter the spatial arrangement of pixels in the image.

| Transformation | Description | Parameter Range | Probability |
|----------------|-------------|-----------------|------------|
| Rotation | Random rotation within range | -25° to 25° | 0.7 |
| Random Crop | Crop and resize to original size | Scale: 0.8 to 1.0 | 0.8 |
| Horizontal Flip | Mirror image horizontally | - | 0.5 |
| Vertical Flip | Mirror image vertically | - | 0.1 |
| Perspective | Apply perspective distortion | Distortion scale: 0.15 | 0.3 |

### 2. Color Adjustments

Transformations that modify the color properties of the image.

| Transformation | Description | Parameter Range | Probability |
|----------------|-------------|-----------------|------------|
| Brightness | Adjust image brightness | 0.8 to 1.2 | 0.7 |
| Contrast | Adjust image contrast | 0.8 to 1.2 | 0.7 |
| Saturation | Adjust color saturation | 0.8 to 1.2 | 0.7 |
| Hue | Shift color hue | -0.05 to 0.05 | 0.7 |

### 3. Noise Addition

Adds various types of noise to the image to simulate image sensor limitations and real-world conditions.

| Noise Type | Description | Intensity Range | Probability |
|------------|-------------|-----------------|------------|
| Gaussian | Zero-mean random noise | 0.01 to 0.05 | 0.5 |
| Salt & Pepper | Random white and black pixels | 0.01 to 0.05 | 0.5 |
| Speckle | Multiplicative noise | 0.01 to 0.05 | 0.5 |

### 4. Blurring and Sharpening

Simulates focus issues and enhances edge definition.

| Transformation | Description | Parameter Range | Probability |
|----------------|-------------|-----------------|------------|
| Gaussian Blur | Applies Gaussian blur | Radius: 0.1 to 2.0 | 0.5 |
| Sharpening | Enhances edge definition | Factor: 0.8 to 1.5 | 0.5 |

### 5. Pizza-Specific Transformations

Special transformations designed specifically for pizza images.

| Transformation | Description | Parameter Range | Probability |
|----------------|-------------|-----------------|------------|
| Burning Effect | Simulates different burning patterns | Intensity: 0.1 to 0.8 | 0.3 |
| Oven Effect | Simulates lighting in an oven | Strength: 0.3 to 0.7 | 0.3 |
| Segment Effect | Applies different effects to segments | - | 0.3 |

## Intensity Presets

The augmentation pipeline provides three intensity presets:

### Low Intensity

Mild augmentations suitable for models that are already performing well but need slight improvement in robustness.

- Narrower parameter ranges
- Lower probabilities for aggressive transformations
- No vertical flips
- Minimal perspective distortion

### Medium Intensity (Default)

Balanced augmentations suitable for most training scenarios.

- Moderate parameter ranges
- Balanced probabilities
- Includes all transformation types

### High Intensity

Aggressive augmentations for maximizing robustness and generalization.

- Wider parameter ranges
- Higher probabilities for all transformations
- Includes vertical flips
- Stronger perspective distortions

## Usage in Training

To use the standard augmentation pipeline in training:

```python
from scripts.standard_augmentation import get_standard_augmentation_pipeline

# Create augmentation pipeline with medium intensity (default)
aug_pipeline = get_standard_augmentation_pipeline(image_size=224)

# Get transforms for use with PyTorch DataLoader
transform = aug_pipeline.get_transforms()

# Create dataset with augmentation
train_dataset = YourDataset(train_data_path, transform=transform)

# Create dataloader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

For different intensity levels:

```python
# Low intensity
aug_pipeline_low = get_standard_augmentation_pipeline(image_size=224, intensity='low')

# High intensity
aug_pipeline_high = get_standard_augmentation_pipeline(image_size=224, intensity='high')
```

## Custom Configuration

For more control, you can create a pipeline with custom parameters:

```python
from scripts.standard_augmentation import StandardAugmentationPipeline

custom_pipeline = StandardAugmentationPipeline(
    image_size=224,
    rotation_range=(-45, 45),  # More aggressive rotation
    brightness_range=(0.6, 1.4),  # More aggressive brightness
    horizontal_flip_prob=0.7,  # Higher flip probability
    pizza_specific_prob=0.5,  # More pizza-specific augmentations
)
```

## Visualizing Augmentations

To visualize the augmentations, run the script directly with an image path:

```bash
python scripts/standard_augmentation.py path/to/image.jpg
```

This will display the original image alongside augmented versions with low, medium, and high intensity settings.
