# Formal Verification Framework for Pizza Detection AI System

This framework provides tools for mathematically proving the adherence of neural network models to specific properties, particularly for the Pizza Detection system.

## Overview

The formal verification framework allows you to mathematically prove that your pizza detection model adheres to certain desirable properties. Unlike traditional testing, which only checks specific inputs, formal verification provides mathematical guarantees about the model's behavior for ALL possible inputs within certain bounds.

This implementation leverages α,β-CROWN, a state-of-the-art neural network verification tool, to provide formal guarantees for the MicroPizzaNet family of models.

## Key Features

- **Robustness Verification**: Verify that small perturbations to an input image don't change the model's prediction
- **Brightness Invariance**: Prove that the model makes consistent predictions across various brightness levels
- **Class Separation**: Verify that critical classes (e.g., raw vs. cooked pizza) are never confused
- **Comprehensive Reports**: Generate detailed verification reports to document model properties
- **Integration with CI/CD**: Can be integrated with the existing pipeline for automated verification

## Installation Requirements

The verification framework requires additional dependencies beyond the core pizza detection system:

```bash
pip install auto_LiRPA
```

For GPU acceleration (recommended for larger models):

```bash
pip install auto_LiRPA torch-cuda
```

## Usage Examples

### Basic Verification

```python
from models.formal_verification.formal_verification import ModelVerifier, load_model_for_verification
import numpy as np

# Load a model for verification
model = load_model_for_verification(
    model_path="models/pizza_model.pth",
    model_type="MicroPizzaNet",
    num_classes=6
)

# Create a verifier
verifier = ModelVerifier(model, input_size=(48, 48), epsilon=0.01)

# Verify robustness on a sample image
img = np.load("test_images/pizza_sample.npy")  # shape: (H, W, C)
true_class = 2  # e.g., "baked" class

# Perform verification
result = verifier.verify_robustness(img, true_class)

print(f"Verified: {result.verified}")
print(result)
```

### Comprehensive Verification

```python
import glob
from PIL import Image
import numpy as np

# Load multiple test images for comprehensive verification
test_images = []
true_classes = []

for class_idx, class_name in enumerate(["raw", "baked", "burnt"]):
    image_paths = glob.glob(f"test_images/{class_name}/*.jpg")
    for path in image_paths[:5]:  # Take first 5 images of each class
        img = np.array(Image.open(path).resize((48, 48)))
        test_images.append(img)
        true_classes.append(class_idx)

# Define critical class pairs (classes that should never be confused)
critical_pairs = [(0, 1)]  # raw and baked should never be confused

# Run comprehensive verification
results = verifier.verify_all_properties(
    input_images=test_images,
    true_classes=true_classes,
    critical_class_pairs=critical_pairs,
    robustness_eps=0.01,
    brightness_range=(0.8, 1.2)
)

# Generate and save a verification report
report = verifier.generate_verification_report(
    results=results,
    output_path="verification_report.json"
)

# Print summary
print(f"Overall verification rate: {report['summary']['overall_verification_rate']:.2%}")
```

## Supported Properties

### 1. Robustness

Verifies that the model's prediction is stable under small perturbations to the input. This is crucial for ensuring that the model is not susceptible to adversarial attacks or small variations in lighting, camera position, etc.

The robustness property is defined with respect to a specific norm (L∞, L1, or L2) and a maximum perturbation size ε.

### 2. Brightness Invariance

Verifies that the model's prediction is consistent across a range of brightness levels. This is important for real-world applications where lighting conditions may vary.

### 3. Class Separation

Verifies that certain critical classes are never confused with each other. For example, ensuring that a raw pizza is never classified as fully cooked, which would be a safety issue.

## Integration with CI/CD Pipeline

The formal verification framework can be integrated into your CI/CD pipeline to automatically verify model properties at build time. Here's an example of how to integrate it:

```python
# In your CI/CD script
from models.formal_verification.formal_verification import ModelVerifier, load_model_for_verification

def verify_model_in_pipeline(model_path, verification_data_path, report_path):
    """Verify model properties as part of CI/CD pipeline"""
    
    # Load model and verification data
    model = load_model_for_verification(model_path, num_classes=6)
    verification_data = np.load(verification_data_path, allow_pickle=True).item()
    
    # Extract test images and true classes
    test_images = verification_data["images"]
    true_classes = verification_data["classes"]
    critical_pairs = verification_data["critical_pairs"]
    
    # Create verifier
    verifier = ModelVerifier(model, input_size=(48, 48), epsilon=0.01)
    
    # Run verification
    results = verifier.verify_all_properties(
        input_images=test_images,
        true_classes=true_classes,
        critical_class_pairs=critical_pairs
    )
    
    # Generate report
    report = verifier.generate_verification_report(
        results=results,
        output_path=report_path
    )
    
    # Check verification threshold
    if report["summary"]["overall_verification_rate"] < 0.95:
        raise Exception("Model failed to meet verification threshold")
    
    return report
```

## Extending the Framework

### Adding New Properties

To add a new verification property:

1. Add the property to the `VerificationProperty` enum
2. Implement a new verification method in the `ModelVerifier` class
3. Update the `verify_all_properties` method to include the new property
4. Update the reporting functions to handle the new property

### Supporting New Model Architectures

The framework is designed to work with different model architectures. To add support for a new architecture:

1. Ensure the model can be loaded with PyTorch
2. Add the model type to the `load_model_for_verification` function
3. Test basic verification with the model to ensure compatibility with α,β-CROWN

## Technical Details

The formal verification is performed using α,β-CROWN, which combines multiple verification techniques to provide tight bounds on the output of neural networks. The framework abstracts away the complexities of the underlying verification algorithms.

### Verification Process

1. The model is wrapped in a `BoundedModule` for verification
2. Input perturbations are defined (e.g., L∞ norm bounded perturbations)
3. Bounds on the model outputs are computed using CROWN or IBP+backward method
4. The bounds are analyzed to determine if the property is verified
5. Results are compiled into a structured report

## Troubleshooting

### Common Issues:

- **Verification is too slow**: Reduce the input size, use fewer test cases, or reduce the complexity of the properties being verified.
- **Memory errors**: Reduce batch size, use CPU mode, or simplify the model.
- **Verification always fails**: Try increasing epsilon, checking for model accuracy issues, or using a stronger verification algorithm (e.g., 'beta-crown' instead of 'crown').

### Performance Tips:

- Use the CPU for small models and images, but consider GPU for larger models
- Verification time scales with the size of the input, the complexity of the model, and the tightness of the bounds
- Start with a small verification problem and gradually scale up

## References

- [α,β-CROWN: Neural Network Verification](https://github.com/Verified-Intelligence/auto_LiRPA)
- [Formal Verification of Neural Networks: Survey and Challenges](https://arxiv.org/abs/1902.06559)
- [CROWN: A Neural Network Verification Algorithm](https://arxiv.org/abs/1811.00866)

## Example Scripts

This directory includes two example scripts that demonstrate how to use the verification framework:

### verify_model_example.py

A basic example script that demonstrates how to:
- Load a trained MicroPizzaNet model
- Load sample images from the dataset
- Verify robustness and brightness invariance
- Generate a verification report

Usage:
```bash
python verify_model_example.py
```

### batch_verification.py

An advanced script for comparing verification properties across multiple models:
- Loads multiple model variants (base, v2, with SE)
- Verifies all properties for each model
- Compares verification rates and performance
- Generates a comprehensive HTML report with visualizations

Usage:
```bash
python batch_verification.py --max-images 10 --epsilon 0.03
```

Arguments:
- `--data-dir`: Directory containing test images (default: augmented_pizza)
- `--output-dir`: Directory to save verification reports (default: models/formal_verification/reports)
- `--max-images`: Maximum number of images to verify per model (default: 5)
- `--epsilon`: Perturbation size for robustness verification (default: 0.03)
- `--backend`: Verification backend ('crown' or 'beta-crown') (default: 'crown')
- `--device`: Device to run verification on ('cpu' or 'cuda') (default: 'cpu')

## Best Practices

For effective formal verification of your pizza detection models:

1. **Start Small**: Begin with a small set of images and a simple model
2. **Increase Gradually**: Once successful, increase the complexity of your verification
3. **Set Reasonable Bounds**: Use reasonable values for parameters like epsilon
4. **Choose Critical Properties**: Focus on properties that are critical for your application
5. **Use as Part of CI/CD**: Integrate verification into your model development pipeline
6. **Use Counterexamples**: If verification fails, use the counterexamples to improve your model

## Performance Considerations

Formal verification is computationally intensive. For better performance:

- Use simpler models when possible
- Limit the number of images used for verification
- Adjust epsilon based on application needs
- Consider using beta-crown backend for more accurate results (but slower)
- Use CUDA if available for faster verification
