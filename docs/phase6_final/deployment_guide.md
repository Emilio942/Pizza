# Pizza Verification System - Deployment Guide

## Development Deployment

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- Git

### Quick Start

1. **Clone the repository:**
```bash
git clone <repository-url>
cd pizza
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Initialize the system:**
```bash
python scripts/initialize_aufgabe_4_2.py
```

5. **Start the API server:**
```bash
python src/api/pizza_api.py
```

## Production Deployment

### Docker Deployment

1. **Build the Docker image:**
```bash
docker build -t pizza-verifier .
```

2. **Run the container:**
```bash
docker run -p 8000:8000 pizza-verifier
```

### Using Docker Compose

```bash
docker-compose up -d
```

## Hardware Deployment (RP2040)

### Prerequisites
- RP2040 development board
- CMSIS-NN toolchain
- ARM GCC compiler

### Deployment Steps

1. **Prepare the model:**
```python
from src.deployment.rp2040_verifier_deployment import RP2040VerifierDeployment

deployment = RP2040VerifierDeployment()
deployment.initialize()
deployment.quantize_verifier_model('models/micro_pizza_model.pth')
```

2. **Generate deployment code:**
```python
deployment.generate_deployment_code()
```

3. **Flash to RP2040:**
```bash
# Use your preferred flashing tool
# e.g., picotool, OpenOCD, etc.
```

## Environment Configuration

### Environment Variables

```bash
# API Configuration
export PIZZA_API_HOST=0.0.0.0
export PIZZA_API_PORT=8000

# Model Configuration
export PIZZA_MODEL_PATH=models/micro_pizza_model.pth
export PIZZA_USE_GPU=true

# Logging
export PIZZA_LOG_LEVEL=INFO
export PIZZA_LOG_FILE=logs/pizza_verifier.log
```

### Configuration Files

Create `config/production.json`:
```json
{
  "model": {
    "path": "models/micro_pizza_model.pth",
    "device": "cuda",
    "batch_size": 32
  },
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4
  },
  "monitoring": {
    "enabled": true,
    "interval": 30
  }
}
```

## Monitoring and Maintenance

### Health Checks
```bash
curl http://localhost:8000/health
```

### Logs
```bash
tail -f logs/pizza_verifier.log
```

### Performance Monitoring
```python
python scripts/monitor_integrated_systems.py
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory:**
   - Reduce batch size
   - Use CPU fallback

2. **Model not found:**
   - Check model path configuration
   - Ensure models are downloaded

3. **API not responding:**
   - Check port availability
   - Verify firewall settings

### Performance Optimization

1. **GPU Acceleration:**
   - Ensure CUDA is properly installed
   - Use appropriate PyTorch version

2. **Model Optimization:**
   - Use quantized models for production
   - Enable model caching

3. **API Optimization:**
   - Implement connection pooling
   - Use async endpoints
   - Add caching layer

## Security Considerations

1. **API Security:**
   - Implement authentication
   - Use HTTPS in production
   - Add rate limiting

2. **Model Security:**
   - Validate input data
   - Sanitize file paths
   - Implement access controls

## Scaling

### Horizontal Scaling
- Use load balancers
- Deploy multiple API instances
- Implement model serving solutions

### Vertical Scaling
- Increase server resources
- Optimize model inference
- Use GPU acceleration

---
*Deployment guide updated on 2025-06-08 16:52:25*