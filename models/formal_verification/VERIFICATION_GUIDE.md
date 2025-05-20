# Formal Verification Framework Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Formal Verification vs. Testing](#formal-verification-vs-testing)
3. [Installation](#installation)
4. [Verification Properties](#verification-properties)
5. [Integrating with Your Workflow](#integrating-with-your-workflow)
6. [Command-Line Usage](#command-line-usage)
7. [CI/CD Integration](#cicd-integration)
8. [Advanced Topics](#advanced-topics)
9. [Troubleshooting](#troubleshooting)
10. [References](#references)

## Introduction

The formal verification framework provides mathematical guarantees about the behavior of the pizza detection neural network. By using specialized verification algorithms, we can prove that certain properties hold for all possible inputs within defined constraints, rather than just testing with a finite set of examples.

This guide will help you understand, set up, and use the formal verification framework to ensure your pizza detection models meet critical safety and robustness requirements.

## Formal Verification vs. Testing

Traditional testing samples specific inputs to check if a model produces the correct outputs. However, neural networks operate in high-dimensional spaces where it's impossible to test all inputs. Formal verification provides mathematical proofs that certain properties hold for **all** inputs within a defined region.

**Benefits of formal verification:**

- Mathematical guarantees on model properties
- Detection of adversarial examples (if they exist)
- Proof of critical safety requirements (e.g., "raw pizza is never classified as cooked")
- Increased confidence in model robustness

**When to use formal verification:**

- Before deploying models to production
- After making significant model architecture changes
- When safety requirements are critical
- To validate robustness against perturbations

## Installation

The formal verification framework requires additional dependencies beyond the standard pizza detection system:

```bash
# Install α,β-CROWN verification framework
pip install torch==1.12.0  # Specific PyTorch version for compatibility
pip install auto_LiRPA

# For GPU acceleration (recommended for larger models)
pip install auto_LiRPA torch-cuda
```

> **Note**: The automatic fallback to mock implementation means you can still use and test the framework structure even if installing auto_LiRPA fails.

To check if the verification dependencies are correctly installed:

```python
from models.formal_verification.formal_verification import VERIFICATION_DEPENDENCIES_INSTALLED
print(f"Verification available: {VERIFICATION_DEPENDENCIES_INSTALLED}")
```

## Verification Properties

The framework supports verifying several important properties:

### 1. Robustness

Robustness verification ensures that small perturbations to an input (up to a defined ε bound) don't change the model's prediction. This is crucial for ensuring the model isn't susceptible to adversarial attacks or small variations in image capture conditions.

```python
# Example: Verify robustness with ε=0.01 in L∞ norm
result = verifier.verify_robustness(input_image, true_class, epsilon=0.01)
```

### 2. Brightness Invariance

This property verifies that the model makes consistent predictions across a range of brightness levels. This is important for real-world applications where lighting conditions can vary significantly.

```python
# Example: Verify brightness invariance between 80% and 120% of original brightness
result = verifier.verify_brightness_invariance(
    input_image, true_class, brightness_range=(0.8, 1.2)
)
```

### 3. Class Separation

This critical property verifies that certain classes are never confused with each other, even under perturbations. For example, ensuring that raw pizza is never classified as fully cooked, which would be a safety hazard.

```python
# Example: Verify that raw pizza (class 0) is never confused with well-done pizza (class 2)
result = verifier.verify_class_separation(class1=0, class2=2, examples=raw_examples)
```

## Integration with Your Workflow

### Using the Mock Implementation

The framework includes a fallback mock implementation that simulates verification results when auto_LiRPA is not available. This is useful for development, testing, and demonstration purposes:

```python
# The framework automatically uses mock implementation if auto_LiRPA is not installed
try:
    from models.formal_verification.formal_verification import (
        ModelVerifier, VerificationProperty, load_model_for_verification, VERIFICATION_DEPENDENCIES_INSTALLED
    )
    if not VERIFICATION_DEPENDENCIES_INSTALLED:
        raise ImportError("auto_LiRPA not installed")
except ImportError:
    print("Using mock implementation")
    from models.formal_verification.mock_verification import (
        ModelVerifier, VerificationProperty, load_model_for_verification
    )
```

The mock implementation follows the same API as the real implementation but uses simulated verification results instead of mathematical proving.

### Step 1: Load Your Model

Load your pre-trained model using the provided utility function:

```python
from models.formal_verification.formal_verification import load_model_for_verification

model = load_model_for_verification(
    model_path="models/pizza_model.pth",
    model_type="MicroPizzaNet",  # or "MicroPizzaNetV2", "MicroPizzaNetWithSE"
    num_classes=6
)
```

### Step 2: Initialize the Verifier

Create a ModelVerifier instance with your model and desired parameters:

```python
from models.formal_verification.formal_verification import ModelVerifier

verifier = ModelVerifier(
    model=model,
    input_size=(48, 48),  # Standard size for pizza detection models
    device='cuda' if torch.cuda.is_available() else 'cpu',
    epsilon=0.01,  # Default robustness parameter
    norm_type='L_inf',  # Norm type for robustness (L_inf, L_1, L_2)
    verify_backend='crown'  # or 'beta-crown' for tighter bounds
)
```

### Step 3: Verify Properties

Verify individual properties or run comprehensive verification:

```python
# Load sample images
import numpy as np
from PIL import Image

img1 = np.array(Image.open("test_images/raw_pizza.jpg").resize((48, 48)))
img2 = np.array(Image.open("test_images/cooked_pizza.jpg").resize((48, 48)))

# Run comprehensive verification
results = verifier.verify_all_properties(
    input_images=[img1, img2],
    true_classes=[0, 2],  # Raw and cooked classes
    critical_class_pairs=[(0, 2)],  # Raw vs. cooked class pair
    robustness_eps=0.01,
    brightness_range=(0.8, 1.2)
)
```

### Step 4: Generate and Analyze Report

Generate a comprehensive verification report:

```python
report = verifier.generate_verification_report(
    results=results,
    output_path="verification_report.json"
)

# Analyze report
overall_rate = report['summary']['overall_verification_rate']
print(f"Overall verification rate: {overall_rate:.2%}")

# Check specific properties
for prop, data in report['properties'].items():
    print(f"{prop}: {data['verified']}/{data['total']} verified ({data['verification_rate']:.2%})")
```

## Command-Line Usage

The framework includes a command-line tool for running verification:

```bash
# Verify properties on a single image
python models/formal_verification/verify_example.py \
    --model-path models/pizza_model.pth \
    --test-image test_images/pizza_sample.jpg \
    --test-class 1 \
    --property robustness \
    --epsilon 0.01 \
    --output verification_report.json

# Verify properties on multiple images from a directory
python models/formal_verification/verify_example.py \
    --model-path models/pizza_model.pth \
    --test-dir test_images/ \
    --property all \
    --epsilon 0.01 \
    --output verification_report.json
```

## CI/CD Integration

The formal verification framework can be integrated into your CI/CD pipeline to verify model properties before deployment. Here's how to do it:

### 1. Generate Verification Dataset

First, generate a verification dataset with representative examples:

```bash
python models/formal_verification/generate_verification_data.py \
    --data-dir data/test/ \
    --output-dir models/verification_data \
    --num-images 5
```

### 2. Add Verification Step to CI/CD Pipeline

Add a verification step to your CI/CD pipeline. The project includes a GitHub Actions workflow file (`ci_verification_workflow.yml`) that you can use as a template:

```yaml
name: Pizza Model Formal Verification

on:
  push:
    branches: [ main, develop ]
    paths:
      - 'src/pizza_detector.py'
      - 'models/*.pth'
      - 'models/formal_verification/**'

jobs:
  verify_model:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        model: 
          - {path: 'models/pizza_model_float32.pth', type: 'MicroPizzaNet'}
          - {path: 'models/pizza_model_v2.pth', type: 'MicroPizzaNetV2'}

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: |
        pip install torch==1.12.0
        pip install auto_LiRPA || echo "Using mock implementation"
        pip install -r requirements.txt
    - name: Run verification
      run: |
        python models/formal_verification/test_verification.py \
          --model-path ${{ matrix.model.path }} \
          --model-type ${{ matrix.model.type }}
```

### 3. Analyze Results

The verification results will be available in the specified report file. In CI mode, the script will exit with an error code if the verification rate is below the specified threshold.

## Advanced Topics

### Testing the Verification Framework

The framework includes comprehensive testing tools:

1. **Unit Tests**: Validate the framework's core functionality
   ```bash
   python -m unittest models/formal_verification/test_verification_unit.py
   ```

2. **Functional Tests**: Test the framework with real models
   ```bash
   python models/formal_verification/test_verification.py --model-path models/pizza_model.pth
   ```

3. **Benchmark Testing**: Compare verification performance across models
   ```bash
   python models/formal_verification/batch_verification.py --output-dir reports
   ```

### Custom Properties

To add custom verification properties, extend the `VerificationProperty` enum and implement a new verification method in the `ModelVerifier` class.

For example, to add a monotonicity property:

```python
# Add to VerificationProperty enum
class VerificationProperty(Enum):
    # ... existing properties ...
    MONOTONICITY = "monotonicity"  # New property

# Implement verification method in ModelVerifier
def verify_monotonicity(self, input_image, transformation_range):
    # Implementation logic
    pass

# Update verify_all_properties method to include the new property
```

### Performance Optimization

Formal verification can be computationally expensive. Here are some optimization tips:

1. **Use smaller models**: Verification complexity scales with model size
2. **Limit input size**: Smaller inputs are faster to verify
3. **Start with loose bounds**: Begin with larger ε values and gradually tighten
4. **Use GPU acceleration**: For larger models, GPU can significantly speed up verification
5. **Use beta-crown backend**: The 'beta-crown' backend provides tighter bounds but may be slower

## Troubleshooting

### Common Issues

1. **Verification is too slow**:
   - Reduce the input size
   - Use fewer test examples
   - Try a simpler property first
   - Use GPU acceleration

2. **Memory errors**:
   - Reduce the batch size
   - Use CPU mode for large models
   - Simplify the model architecture

3. **Verification always fails**:
   - Check that the model is accurate on the test examples
   - Increase the epsilon value
   - Check for preprocessing inconsistencies
   - Try a different verification algorithm (e.g., 'beta-crown')

4. **Import errors**:
   - Ensure auto_LiRPA is installed: `pip install torch==1.12.0 auto_LiRPA`
   - If auto_LiRPA installation fails, the framework will use the mock implementation
   - Check for version conflicts with other packages

5. **PyTorch version conflicts**:
   - auto_LiRPA requires PyTorch versions between 1.8.0 and 1.12.0
   - Create a separate environment for verification if needed:
     ```bash
     conda create -n verification python=3.8
     conda activate verification
     pip install torch==1.12.0
     pip install auto_LiRPA
     ```

### Debugging Tips

- Set smaller epsilon values first to get successful verifications
- Verify properties individually before running comprehensive verification
- Check the model's prediction on unperturbed inputs first
- Examine verification reports for patterns in failures

## References

1. [α,β-CROWN: Neural Network Verification](https://github.com/Verified-Intelligence/auto_LiRPA)
2. [Formal Verification of Neural Networks: Survey and Challenges](https://arxiv.org/abs/1902.06559)
3. [CROWN: A Neural Network Verification Algorithm](https://arxiv.org/abs/1811.00866)
4. [Complete Verification of Neural Networks via Layer-by-Layer Abstraction Refinement](https://arxiv.org/abs/2103.06624)
