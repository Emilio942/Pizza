# Pizza Verification API Documentation

## Overview

The Pizza Verification API provides RESTful endpoints for pizza quality
assessment and verification using advanced AI models.

## Base URL
```
http://localhost:8000
```

## Authentication
Currently, the API operates without authentication for development purposes.

## Endpoints

### POST /verify
Verify pizza quality from an image.

**Request Body:**
```json
{
  "image_path": "string",
  "model_prediction": "string",
  "confidence_score": 0.85,
  "ground_truth_class": "string (optional)",
  "assessment_level": "standard"
}
```

**Response:**
```json
{
  "quality_score": 0.92,
  "confidence": 0.87,
  "assessment_details": {
    "visual_quality": 0.89,
    "ingredient_distribution": 0.95,
    "cooking_level": 0.88
  },
  "recommendations": [
    "Excellent pizza quality",
    "Good ingredient balance"
  ],
  "processing_time_ms": 45.2,
  "timestamp": "2025-06-08T16:48:52.168064"
}
```

### GET /health
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-06-08T16:48:52.168064",
  "version": "1.0.0"
}
```

### GET /models
List available models.

**Response:**
```json
{
  "models": [
    "micro_pizza_model",
    "pruned_pizza_model",
    "rl_optimized_model"
  ],
  "default_model": "micro_pizza_model"
}
```

## Error Handling

The API uses standard HTTP status codes:

- `200 OK` - Request successful
- `400 Bad Request` - Invalid request parameters
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

**Error Response Format:**
```json
{
  "error": "Error description",
  "code": "ERROR_CODE",
  "timestamp": "2025-06-08T16:48:52.168064"
}
```

## Rate Limiting

Currently no rate limiting is implemented. Production deployment
should include appropriate rate limiting mechanisms.

## Examples

### cURL Examples

**Verify a pizza:**
```bash
curl -X POST 'http://localhost:8000/verify' \
  -H 'Content-Type: application/json' \
  -d '{
    "image_path": "/path/to/pizza.jpg",
    "model_prediction": "good_pizza",
    "confidence_score": 0.85
  }'
```

**Check health:**
```bash
curl -X GET 'http://localhost:8000/health'
```

### Python Examples

```python
import requests

# Verify pizza
response = requests.post(
    'http://localhost:8000/verify',
    json={
        'image_path': '/path/to/pizza.jpg',
        'model_prediction': 'good_pizza',
        'confidence_score': 0.85
    }
)

result = response.json()
print(f'Quality Score: {result["quality_score"]}')
```
