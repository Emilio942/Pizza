# Enhanced Test Set Extension Tool

This tool analyzes the current test set and extends it with challenging test cases that mimic real-world conditions. It helps ensure the model is evaluated on a diverse and representative set of images.

## Features

- **Analyze Test Set Distribution**: Examines the current class distribution and challenging condition coverage
- **Generate Challenging Test Images**: Creates images with various challenging conditions:
  - Lighting variations (dark, bright, uneven, low contrast, shadowed, backlit)
  - Perspective variations (side views, top-down views, close-ups, diagonal angles)
  - Quality variations (noise, blur, compression artifacts, motion blur)
- **Prevent Data Leakage**: Verifies generated test images don't appear in training/validation sets
- **Visualize Distribution**: Generates charts showing class distribution before and after extension
- **Flexible Configuration**: Command-line options for customizing test set generation

## Usage

```bash
# Generate 100 challenging test images and check for data leakage
python extend_test_set_enhanced.py --target 100 --check-leakage

# Generate and integrate 150 challenging test images into the main test set
python extend_test_set_enhanced.py --target 150 --integrate

# Just analyze the current test set without generating new images
python extend_test_set_enhanced.py --target 0
```

## Command-line Options

- `--target N`: Generate N new test images (distributed across classes)
- `--integrate`: Copy generated images to the main test directory
- `--check-leakage`: Check for potential data leakage with training/validation sets

## Output

- Generated images are stored in `data/test_new_images/`
- Analysis results are saved in `output/data_analysis/`
- When `--integrate` is used, images are copied to `data/test/`

## Requirements

- Python 3.6+
- PIL/Pillow
- NumPy
- Matplotlib

## Notes

This tool extends the original `extend_test_set.py` with additional functionality to create more diverse and challenging test scenarios.
