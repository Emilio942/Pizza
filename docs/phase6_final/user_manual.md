# Pizza Quality Verification System - User Manual

## Getting Started

Welcome to the Pizza Quality Verification System! This guide will help you
get started with using the system for automated pizza quality assessment.

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- GPU support optional but recommended

## Quick Start

### 1. Basic Setup

```bash
# Install the system
pip install -r requirements.txt

# Initialize
python scripts/initialize_aufgabe_4_2.py
```

### 2. First Pizza Verification

```python
from src.verification.pizza_verifier import PizzaVerifier

# Create verifier
verifier = PizzaVerifier()

# Verify a pizza
result = verifier.verify_pizza('path/to/your/pizza.jpg')

# Check the result
print(f'Quality Score: {result.quality_score}')
print(f'Confidence: {result.confidence}')
```

## Using the Web API

### Starting the API Server

```bash
python src/api/pizza_api.py
```

The API will be available at `http://localhost:8000`

### Making Verification Requests

```bash
curl -X POST 'http://localhost:8000/verify' \
  -H 'Content-Type: application/json' \
  -d '{
    "image_path": "/path/to/pizza.jpg",
    "assessment_level": "detailed"
  }'
```

## Understanding Results

### Quality Score
- **0.0 - 0.3:** Poor quality
- **0.3 - 0.6:** Fair quality
- **0.6 - 0.8:** Good quality
- **0.8 - 1.0:** Excellent quality

### Assessment Levels
- **standard:** Basic quality assessment
- **detailed:** Comprehensive analysis
- **food_safety:** Safety-focused evaluation

## Advanced Features

### Continuous Monitoring

```python
from src.continuous_improvement.pizza_verifier_improvement import ContinuousPizzaVerifierImprovement

improvement = ContinuousPizzaVerifierImprovement(
    base_models_dir='models',
    rl_training_results_dir='results'
)
improvement.initialize()
# System will continuously improve performance
```

### Custom Model Configuration

```python
verifier = PizzaVerifier(
    model_path='path/to/custom/model.pth',
    device='cuda',  # or 'cpu'
    confidence_threshold=0.8
)
```

## Troubleshooting

### Common Issues

**Q: Model not found error**
A: Ensure models are downloaded and paths are correct

**Q: Low quality scores**
A: Check image quality and lighting conditions

**Q: Slow performance**
A: Enable GPU acceleration or use quantized models

### Getting Help

- Check the logs in `logs/` directory
- Review the technical documentation
- Verify system requirements

## Best Practices

1. **Image Quality:**
   - Use good lighting
   - Ensure pizza is clearly visible
   - Avoid shadows and reflections

2. **Performance:**
   - Use GPU when available
   - Enable caching for repeated assessments
   - Monitor system resources

3. **Accuracy:**
   - Use appropriate assessment levels
   - Validate results with ground truth when possible
   - Monitor confidence scores

---
*User manual updated on 2025-06-08 16:52:25*