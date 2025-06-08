# Pizza Classification API Documentation
## SPATIAL-4.1: Complete API Integration with Spatial-MLLM

### Overview
The Pizza Classification API provides advanced pizza image classification using both Spatial-MLLM and standard CNN models with comprehensive A/B testing, spatial visualization, and performance monitoring capabilities.

### Base URL
```
http://localhost:8000
```

### API Features
- **Dual Model Support**: Spatial-MLLM and Standard CNN
- **Spatial Visualizations**: Feature maps, attention maps, prediction overlays
- **A/B Testing**: Comprehensive model comparison
- **Performance Monitoring**: Real-time metrics and status
- **Error Handling**: Robust fallback mechanisms
- **Caching**: Intelligent prediction caching

---

## Endpoints

### 1. Classification Endpoints

#### POST /classify
Classify a pizza image using available models.

**Parameters:**
- `file` (required): Image file (JPEG/PNG, max 10MB)
- `use_spatial` (optional): Force spatial model usage (boolean)
- `enable_visualization` (optional): Generate visualizations (boolean, default: true)
- `return_probabilities` (optional): Return class probabilities (boolean, default: true)
- `ab_test_group` (optional): A/B test group assignment ("A" or "B")

**Response:**
```json
{
  "predicted_class": "basic",
  "confidence": 0.85,
  "probabilities": {
    "basic": 0.85,
    "burnt": 0.03,
    "combined": 0.03,
    "mixed": 0.03,
    "progression": 0.03,
    "segment": 0.03
  },
  "model_used": "spatial-mllm",
  "inference_time": 1.65,
  "spatial_features": {
    "raw_response": "<answer>C</answer>",
    "processed_features": "spatial_analysis_placeholder"
  },
  "visualization_url": "/visualizations/1749216910",
  "ab_test_group": null,
  "timestamp": "2025-06-06T15:35:10.192132"
}
```

**Example cURL:**
```bash
curl -X POST -F "file=@pizza.jpg" -F "model_type=spatial-mllm" http://localhost:8000/classify
```

#### POST /ab-test/compare
Compare both models on the same image for comprehensive A/B testing.

**Parameters:**
- `file` (required): Image file (JPEG/PNG, max 10MB)

**Response:**
```json
{
  "comparison_results": {
    "spatial": {
      "predicted_class": "basic",
      "confidence": 0.85,
      "model_used": "spatial-mllm",
      "inference_time": 1.65,
      "visualization_url": "/visualizations/1749216910"
    },
    "standard": {
      "predicted_class": "basic", 
      "confidence": 0.92,
      "model_used": "standard-cnn",
      "inference_time": 0.12
    }
  },
  "timestamp": "2025-06-06T15:35:10.192132",
  "models_tested": ["spatial", "standard"]
}
```

---

### 2. Visualization Endpoints

#### GET /visualizations
List all available visualization directories.

**Response:**
```json
{
  "visualizations": [
    {
      "viz_id": "1749216910",
      "timestamp": "1749216910",
      "files": [
        "spatial_features_pizza.jpg.png",
        "attention_pizza.jpg.png",
        "original_pizza.jpg",
        "prediction_pizza.jpg.png"
      ]
    }
  ]
}
```

#### GET /visualizations/{viz_id}
Get information about a specific visualization.

**Parameters:**
- `viz_id` (required): Visualization ID from classification response

**Response:**
```json
{
  "viz_id": "1749216910",
  "timestamp": "1749216910", 
  "files": [
    "spatial_features_pizza.jpg.png",
    "attention_pizza.jpg.png",
    "original_pizza.jpg",
    "prediction_pizza.jpg.png"
  ]
}
```

#### GET /visualizations/{viz_id}/{filename}
Serve individual visualization files.

**Parameters:**
- `viz_id` (required): Visualization ID
- `filename` (required): File name from visualization list

**Response:** Binary image file

**Available visualization types:**
- `spatial_features_*.png`: Spatial feature maps with depth and surface normals
- `attention_*.png`: Attention visualization maps
- `prediction_*.png`: Prediction overlay with confidence scores
- `original_*`: Original uploaded image

---

### 3. Status and Monitoring Endpoints

#### GET /status
Get API and model status information.

**Response:**
```json
{
  "api_status": "online",
  "models": {
    "spatial-mllm": {
      "model_name": "Spatial-MLLM",
      "available": true,
      "loaded": true,
      "last_used": null,
      "total_inferences": 5,
      "average_inference_time": 1.64,
      "error_count": 0
    },
    "standard-cnn": {
      "model_name": "Standard CNN",
      "available": true,
      "loaded": false,
      "total_inferences": 0,
      "average_inference_time": 0.0,
      "error_count": 0
    }
  },
  "system_info": {
    "device": "cuda",
    "cuda_available": true,
    "models_loaded": true
  },
  "timestamp": "2025-06-06T15:29:03.817587"
}
```

#### GET /metrics
Get detailed API performance metrics.

**Response:**
```json
{
  "total_requests": 12,
  "successful_requests": 11,
  "failed_requests": 1,
  "average_response_time": 0.0,
  "models_status": {
    "spatial-mllm": {
      "model_name": "Spatial-MLLM",
      "available": true,
      "loaded": true,
      "total_inferences": 8,
      "average_inference_time": 1.58,
      "error_count": 0
    }
  },
  "uptime": 3600.5
}
```

#### GET /ab-test/status
Get A/B testing status and performance comparison.

**Response:**
```json
{
  "ab_test_active": true,
  "split_ratio": 0.5,
  "groups": {
    "A_spatial": {
      "model": "spatial-mllm",
      "requests": 8,
      "average_time": 1.58,
      "error_rate": 0.0,
      "success_rate": 1.0
    },
    "B_standard": {
      "model": "standard-cnn",
      "requests": 3,
      "average_time": 0.12,
      "error_rate": 0.0,
      "success_rate": 1.0
    }
  },
  "recommendations": {
    "preferred_model": "spatial-mllm",
    "performance_comparison": {
      "spatial_faster": false,
      "spatial_more_reliable": true
    }
  },
  "timestamp": "2025-06-06T15:35:10.192132"
}
```

#### GET /health
Simple health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-06-06T15:35:10.192132"
}
```

---

### 4. Documentation Endpoints

#### GET /
API documentation homepage with interactive interface.

#### GET /docs
Interactive Swagger/OpenAPI documentation.

---

## Pizza Classification Classes

The API classifies pizza images into the following categories:

1. **basic** - Standard pizza with basic toppings
2. **burnt** - Pizza with burning or overcooking characteristics
3. **combined** - Pizza with multiple combined elements
4. **mixed** - Pizza with mixed or varied toppings
5. **progression** - Pizza showing progression/stages
6. **segment** - Pizza with distinct segments or sections

---

## Spatial Visualization Types

### 1. Spatial Feature Maps
- **File pattern**: `spatial_features_*.png`
- **Content**: 2x2 grid showing:
  - Original image
  - Synthetic depth map (blue to red gradient)
  - Surface normals (X component)
  - Surface normals (Y component)
- **Purpose**: Visualize 3D spatial characteristics

### 2. Attention Visualization
- **File pattern**: `attention_*.png`
- **Content**: 1x3 grid showing:
  - Original image
  - Attention heat map
  - Attention overlay on original
- **Purpose**: Show model focus regions

### 3. Prediction Overlay
- **File pattern**: `prediction_*.png`
- **Content**: Original image with:
  - Prediction class and confidence
  - Model used and inference time
  - Top 5 class probabilities
- **Purpose**: Complete prediction visualization

---

## Error Handling

### HTTP Status Codes
- **200**: Success
- **400**: Bad Request (invalid image, parameters)
- **404**: Not Found (visualization not found)
- **413**: Payload Too Large (file > 10MB)
- **500**: Internal Server Error
- **503**: Service Unavailable (models not loaded)

### Error Response Format
```json
{
  "detail": "Error description"
}
```

### Common Errors
- **Invalid file type**: Only JPEG/PNG supported
- **File too large**: Maximum 10MB per image
- **Model not available**: Service temporarily unavailable
- **Visualization not found**: Invalid visualization ID

---

## Performance Characteristics

### Spatial-MLLM Model
- **Inference time**: ~1.5-2.0 seconds
- **Memory usage**: ~4-6GB GPU RAM
- **Accuracy**: High spatial awareness
- **Visualizations**: Full spatial analysis

### Standard CNN Model
- **Inference time**: ~0.1-0.2 seconds
- **Memory usage**: ~1-2GB GPU RAM
- **Accuracy**: Fast classification
- **Visualizations**: Not available

---

## A/B Testing Guide

### Manual A/B Testing
Use the `ab_test_group` parameter in `/classify`:
- `"A"` or `"spatial"`: Use Spatial-MLLM
- `"B"` or `"standard"`: Use Standard CNN

### Automatic Comparison
Use `/ab-test/compare` to test both models simultaneously.

### Performance Monitoring
Check `/ab-test/status` for:
- Request distribution
- Average inference times
- Error rates
- Performance recommendations

---

## Integration Examples

### Python Client
```python
import requests

# Basic classification
with open('pizza.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/classify',
        files={'file': f},
        data={'model_type': 'spatial-mllm'}
    )
    result = response.json()
    print(f"Prediction: {result['predicted_class']}")

# A/B testing
with open('pizza.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/ab-test/compare',
        files={'file': f}
    )
    comparison = response.json()
    print(comparison['comparison_results'])
```

### JavaScript (Browser)
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('model_type', 'spatial-mllm');

fetch('http://localhost:8000/classify', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(result => {
    console.log('Prediction:', result.predicted_class);
    
    // Load visualizations if available
    if (result.visualization_url) {
        loadVisualizations(result.visualization_url);
    }
});
```

### cURL Examples
```bash
# Basic classification
curl -X POST -F "file=@pizza.jpg" http://localhost:8000/classify

# Force spatial model
curl -X POST -F "file=@pizza.jpg" -F "use_spatial=true" http://localhost:8000/classify

# A/B test group assignment
curl -X POST -F "file=@pizza.jpg" -F "ab_test_group=A" http://localhost:8000/classify

# Model comparison
curl -X POST -F "file=@pizza.jpg" http://localhost:8000/ab-test/compare

# Get visualizations
curl http://localhost:8000/visualizations/1749216910

# Download specific visualization
curl -o feature_map.png http://localhost:8000/visualizations/1749216910/spatial_features_pizza.jpg.png
```

---

## Configuration

### Model Settings
- **Spatial Model**: Diankun/Spatial-MLLM-subset-sft
- **Device**: Auto-detect (CUDA if available, CPU fallback)
- **Max file size**: 10MB
- **Supported formats**: JPEG, PNG
- **Cache TTL**: 5 minutes

### Performance Tuning
- **Max concurrent requests**: 5
- **Inference timeout**: 30 seconds
- **Visualization generation**: Background tasks
- **Memory optimization**: Half-precision for CUDA

---

## Deployment Notes

### Requirements
- Python 3.8+
- PyTorch with CUDA support (recommended)
- 8GB+ GPU RAM for optimal performance
- FastAPI and dependencies

### Starting the Server
```bash
cd /path/to/pizza/src/api
python pizza_api.py --host 0.0.0.0 --port 8000
```

### Production Considerations
- Use reverse proxy (nginx) for static files
- Configure proper logging levels
- Set up monitoring and alerts
- Implement rate limiting
- Use HTTPS in production

---

## Support and Troubleshooting

### Common Issues
1. **Models not loading**: Check GPU memory and CUDA installation
2. **Slow inference**: Verify GPU utilization
3. **Visualization errors**: Check disk space and permissions
4. **API timeouts**: Increase timeout settings

### Monitoring
- Check `/status` for model health
- Monitor `/metrics` for performance trends
- Use `/health` for load balancer checks

### Logging
Logs include:
- Model loading status
- Inference performance
- Error details
- API request metrics

---

*Last updated: June 6, 2025*
*API Version: SPATIAL-4.1*
