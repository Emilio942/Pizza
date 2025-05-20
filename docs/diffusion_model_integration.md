# Diffusion Model Integration for Pizza Dataset

This document describes how to use diffusion models to generate synthetic pizza images for 
training and enhancing the pizza classification model, especially for underrepresented classes
and difficult-to-capture scenarios.

## Overview

The diffusion model integration allows you to:

1. Analyze the current dataset distribution to identify underrepresented classes
2. Generate high-quality synthetic images with specific properties (burn patterns, lighting conditions, etc.)
3. Organize and filter the generated images based on quality and content
4. Integrate the synthetic images into the main dataset

## Quick Start

The simplest way to use this feature is to run the provided shell script:

```bash
./generate_diffusion_dataset.sh
```

This interactive script will:
1. Analyze your dataset
2. Let you choose which types of images to generate
3. Generate images using memory-optimized settings
4. Create a quality control report for you to review
5. Integrate approved images into your dataset

## Manual Usage

You can also use the integration script directly for more control:

```bash
# Analyze dataset distribution
python scripts/integrate_diffusion_images.py --analyze-dataset

# Generate images with specific preset and model
python scripts/integrate_diffusion_images.py --generate --preset burn_patterns --model sd-food --image-size 512

# Organize generated images into class directories
python scripts/integrate_diffusion_images.py --organize-images

# Create quality control report
python scripts/integrate_diffusion_images.py --quality-report

# Integrate images into main dataset
python scripts/integrate_diffusion_images.py --integrate

# Run all steps in sequence
python scripts/integrate_diffusion_images.py --all
```

## Available Presets

The following generation presets are available:

- `small_diverse`: A small but diverse set of images across all pizza classes
- `training_focused`: Large dataset optimized for model training with balanced classes
- `progression_heavy`: Focus on cooking progression stages, useful for transition states
- `burn_patterns`: Various burn patterns and configurations, useful for classification challenges

## Diffusion Models

The integration supports multiple diffusion models:

- `sd-food`: Food-specific fine-tuned model (default, most memory-efficient)
- `sdxl-turbo`: Faster generation with good quality
- `sdxl`: Highest quality but most memory-intensive
- `kandinsky`: Alternative model with different characteristics

## Memory Optimization

The integration uses memory optimization techniques to run on GPUs with limited VRAM:

- Default image size: 512x512 (change with `--image-size`)
- Batch size: 1 (change with `--batch-size`)
- CPU offloading: Enabled
- Sequential processing: Enabled

For GPUs with less than 8GB VRAM, the script will attempt different strategies automatically.

## Manual Quality Control

All generated images should be manually reviewed before integration:

1. Run the quality control report generator: `python scripts/integrate_diffusion_images.py --quality-report`
2. Open the HTML report in your browser
3. Review images for quality issues, incorrect classifications, etc.
4. Delete any problematic images from the temporary storage in `data/synthetic/[class_name]`
5. Run integration after quality control: `python scripts/integrate_diffusion_images.py --integrate`

## Advanced Prompt Engineering

For advanced users who want to customize the generation prompts:

1. Examine the `PIZZA_STAGES` variable in `src/augmentation/diffusion_pizza_generator.py`
2. Modify the prompts and negative prompts for your specific needs
3. Run generation with your customized parameters

## Troubleshooting

If you encounter memory issues during generation:
- Reduce image size (e.g., `--image-size 256`)
- Try using the "sd-food" model (most memory efficient)
- Run with CPU fallback if all else fails: edit `memory_optimized_generator.py` and add `--use_cpu`

## References

This integration uses the following components:
- `memory_optimized_generator.py`: Main entry point with memory optimizations
- `scripts/generate_pizza_dataset.py`: Base generation script
- `src/augmentation/diffusion_pizza_generator.py`: Diffusion model integration
- `src/augmentation/advanced_pizza_diffusion_control.py`: Controls for specific patterns
