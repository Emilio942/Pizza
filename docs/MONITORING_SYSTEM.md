# Enhanced Spatial Monitoring System Documentation

## Overview

This document provides comprehensive documentation for the enhanced spatial monitoring and logging system implemented as part of **SPATIAL-4.3: Monitoring und Logging erweitern**.

## Architecture

The enhanced monitoring system consists of four main components:

### 1. Spatial Feature Logger (`spatial_feature_logger.py`)
- **Purpose**: Logs spatial feature extraction processes with detailed metrics
- **Key Features**:
  - Context manager for feature extraction timing
  - Thread-safe metrics buffering
  - Prometheus metrics integration
  - JSON export with 24-hour data retention
  - Background processing for performance

### 2. Dual-Encoder Performance Metrics (`dual_encoder_metrics.py`)
- **Purpose**: Profiles dual-encoder architecture performance
- **Key Features**:
  - Individual encoder performance tracking (visual + spatial)
  - GPU monitoring with NVIDIA ML support
  - Connector metrics between encoders
  - Batch processing analysis
  - Bottleneck identification
  - Efficiency scoring

### 3. Spatial Anomaly Detection (`spatial_anomaly_detection.py`)
- **Purpose**: Detects anomalies in spatial features and processing
- **Key Features**:
  - Multiple detection methods (statistical, ML-based, spatial consistency)
  - Severity classification (low/medium/high/critical)
  - Confidence scoring
  - Spatial location identification
  - Feature baseline tracking

### 4. Enhanced Dashboards
- **Main Dashboard**: Extended Grafana dashboard with spatial visualizations
- **Anomaly Dashboard**: Specialized dashboard for anomaly monitoring and alerts

## Installation and Setup

### Prerequisites
```bash
# Python dependencies
pip install prometheus-client numpy scikit-learn nvidia-ml-py3

# System dependencies
sudo apt-get install prometheus grafana
```

### Configuration

1. **Prometheus Configuration** (`config/monitoring/prometheus.yml`):
   - Scrapes metrics from spatial monitoring endpoints
   - Configured for 15-second intervals
   - Retention period: 30 days

2. **Grafana Dashboards**:
   - Import `config/monitoring/extended-grafana-dashboard.json`
   - Import `config/monitoring/spatial-anomaly-dashboard.json`

3. **Spatial Alerting Rules** (`config/monitoring/spatial_rules.yml`):
   - Critical anomaly alerts
   - Performance threshold alerts
   - Feature extraction failure alerts

## Usage

### Basic Usage

```python
from scripts.monitoring.spatial_feature_logger import SpatialFeatureLogger
from scripts.monitoring.dual_encoder_metrics import DualEncoderProfiler
from scripts.monitoring.spatial_anomaly_detection import SpatialAnomalyDetector

# Initialize components
feature_logger = SpatialFeatureLogger()
profiler = DualEncoderProfiler()
anomaly_detector = SpatialAnomalyDetector()

# Log feature extraction
with feature_logger.log_extraction("margherita", 512*512):
    features = extract_spatial_features(image)
    feature_logger.log_feature_quality(quality_score)

# Profile dual encoder
with profiler.profile_batch_processing(batch_size):
    results = process_with_dual_encoder(batch)

# Detect anomalies
anomalies = anomaly_detector.detect_anomalies(features, "pizza_type")
```

### Advanced Usage

#### Custom Metrics
```python
# Custom feature extraction metrics
logger.log_custom_metric("custom_feature_count", count, {"pizza_type": "pepperoni"})

# Custom performance metrics
profiler.log_custom_timing("preprocessing", duration)

# Custom anomaly detection
detector.add_custom_detection_method("custom_method", detection_function)
```

#### Batch Processing
```python
# Process multiple images with monitoring
for batch in image_batches:
    with profiler.profile_batch_processing(len(batch)):
        for image in batch:
            with feature_logger.log_extraction(pizza_type, image_size):
                features = extract_features(image)
                anomalies = detector.detect_anomalies(features, pizza_type)
```

## Metrics Reference

### Spatial Feature Metrics
- `spatial_feature_extraction_duration_seconds`: Feature extraction timing
- `spatial_feature_extractions_total`: Total extractions by pizza type
- `spatial_feature_quality_score`: Quality score distribution
- `spatial_feature_extraction_errors_total`: Extraction errors

### Dual-Encoder Metrics
- `dual_encoder_visual_processing_duration_seconds`: Visual encoder timing
- `dual_encoder_spatial_processing_duration_seconds`: Spatial encoder timing
- `dual_encoder_connector_duration_seconds`: Connector processing time
- `dual_encoder_gpu_utilization_percent`: GPU utilization by encoder
- `dual_encoder_efficiency_score`: Overall processing efficiency
- `dual_encoder_bottleneck_detected`: Bottleneck detection flag

### Anomaly Detection Metrics
- `spatial_anomalies_detected_total`: Anomalies by type and severity
- `spatial_anomaly_confidence_score`: Confidence score distribution
- `spatial_anomaly_detection_duration_seconds`: Detection processing time
- `spatial_anomaly_false_positive_rate`: False positive tracking

## Dashboard Guide

### Main Dashboard Panels

#### Spatial Feature Extraction Section
- **Feature Extraction Timing**: Tracks extraction performance over time
- **Feature Quality Distribution**: Heatmap of quality scores
- **Pizza Type Classification**: Pie chart of processed pizza types
- **Processing Success Rate**: Success rates by pizza type

#### Dual-Encoder Performance Section  
- **Visual vs Spatial Encoder Performance**: Comparative timing graphs
- **Encoder GPU Utilization**: GPU usage tracking
- **Processing Efficiency Score**: Overall system efficiency
- **Bottleneck Detection**: Active bottleneck indicators

#### Spatial Anomaly Detection Section
- **Anomaly Detection Rate**: Rate of anomalies by severity
- **Anomaly Types Distribution**: Breakdown of anomaly types
- **Confidence Score Distribution**: Heatmap of detection confidence
- **Processing Time**: Anomaly detection performance

### Anomaly Alert Dashboard
- **Critical Anomaly Overview**: High-priority anomaly tracking
- **Anomaly Timeline**: Time-series anomaly detection
- **Spatial Location Heatmap**: Geographic distribution of anomalies
- **Recent Anomalies Log**: Live log of detected anomalies

## Alerting Configuration

### Alert Rules

#### Critical Alerts
```yaml
- alert: CriticalSpatialAnomaly
  expr: spatial_anomalies_detected_total{severity="critical"} > 0
  for: 0s
  labels:
    severity: critical
  annotations:
    summary: "Critical spatial anomaly detected"
    description: "{{ $labels.anomaly_type }} anomaly in {{ $labels.pizza_type }}"

- alert: SpatialFeatureExtractionFailed
  expr: rate(spatial_feature_extraction_errors_total[5m]) > 0.1
  for: 1m
  labels:
    severity: warning
  annotations:
    summary: "High feature extraction error rate"
```

#### Performance Alerts
```yaml
- alert: DualEncoderBottleneck
  expr: dual_encoder_bottleneck_detected == 1
  for: 30s
  labels:
    severity: warning
  annotations:
    summary: "Dual encoder bottleneck detected"

- alert: SlowFeatureExtraction
  expr: histogram_quantile(0.95, rate(spatial_feature_extraction_duration_seconds_bucket[5m])) > 2.0
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "Slow spatial feature extraction"
```

## Testing

### Running Tests
```bash
# Run comprehensive test suite
cd /home/emilio/Documents/ai/pizza
python scripts/monitoring/test_monitoring_system.py

# Run specific test types
python -c "
from scripts.monitoring.test_monitoring_system import MonitoringSystemTester
import asyncio

async def test_single_type():
    tester = MonitoringSystemTester()
    result = await tester.test_single_pizza(tester.pizza_test_cases[0], 1)
    print(result)

asyncio.run(test_single_type())
"
```

### Test Coverage
The test suite covers:
- ✅ All pizza types (8 different types)
- ✅ Load testing with concurrent processing
- ✅ Edge cases (large/small images, error conditions)
- ✅ Anomaly detection accuracy
- ✅ Performance regression testing
- ✅ End-to-end monitoring pipeline

### Test Results Interpretation
- **Success Rate**: Should be >95% for production readiness
- **Average Processing Time**: Baseline for performance monitoring
- **Anomaly Detection Rate**: Validates detection sensitivity
- **GPU Utilization**: Ensures efficient resource usage

## Performance Optimization

### Best Practices
1. **Feature Extraction**:
   - Use batch processing for multiple images
   - Enable GPU acceleration when available
   - Cache frequently accessed feature patterns

2. **Monitoring Overhead**:
   - Buffer metrics in background threads
   - Use sampling for high-frequency metrics
   - Implement metric aggregation for storage efficiency

3. **Anomaly Detection**:
   - Tune detection thresholds based on false positive rates
   - Use lightweight statistical methods for real-time detection
   - Implement adaptive baselines for concept drift

### Scaling Considerations
- **Horizontal Scaling**: Deploy monitoring across multiple instances
- **Data Retention**: Configure appropriate retention policies
- **Alert Fatigue**: Implement alert suppression and grouping

## Troubleshooting

### Common Issues

#### High Memory Usage
```python
# Reduce buffer sizes
feature_logger = SpatialFeatureLogger(buffer_size=100)  # Default: 1000

# Enable metric sampling
profiler = DualEncoderProfiler(sampling_rate=0.1)  # Sample 10% of metrics
```

#### GPU Monitoring Issues
```bash
# Install NVIDIA ML Python bindings
pip install nvidia-ml-py3

# Verify GPU access
nvidia-smi
```

#### Missing Metrics
```python
# Check Prometheus configuration
# Ensure scrape endpoints are accessible
# Verify metric registration
```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
feature_logger = SpatialFeatureLogger(debug=True)
```

## API Reference

### SpatialFeatureLogger

#### Methods
- `log_extraction(pizza_type, feature_count)`: Context manager for extraction logging
- `log_feature_quality(score)`: Log quality score
- `log_custom_metric(name, value, labels)`: Log custom metrics
- `export_metrics_json()`: Export metrics as JSON
- `get_metrics_summary()`: Get current metrics summary

### DualEncoderProfiler

#### Methods
- `profile_batch_processing(batch_size)`: Context manager for batch profiling
- `log_visual_encoder_performance(duration, gpu_util)`: Log visual encoder metrics
- `log_spatial_encoder_performance(duration, gpu_util)`: Log spatial encoder metrics
- `log_connector_performance(duration)`: Log connector metrics
- `detect_bottlenecks()`: Identify performance bottlenecks

### SpatialAnomalyDetector

#### Methods
- `detect_anomalies(features, pizza_type)`: Main anomaly detection
- `detect_statistical_anomalies(data)`: Statistical outlier detection
- `detect_ml_anomalies(features)`: ML-based anomaly detection
- `detect_spatial_consistency_anomalies(locations)`: Spatial consistency checks
- `update_baselines(features)`: Update detection baselines

## Configuration Reference

### Environment Variables
```bash
# Monitoring configuration
SPATIAL_MONITORING_ENABLED=true
SPATIAL_METRICS_BUFFER_SIZE=1000
SPATIAL_ANOMALY_THRESHOLD=0.8
SPATIAL_GPU_MONITORING=true

# Prometheus configuration
PROMETHEUS_ENDPOINT=http://localhost:9090
METRICS_EXPORT_INTERVAL=15

# Grafana configuration
GRAFANA_ENDPOINT=http://localhost:3000
DASHBOARD_REFRESH_RATE=30s
```

### File Locations
```
/home/emilio/Documents/ai/pizza/
├── scripts/monitoring/
│   ├── spatial_feature_logger.py      # Feature extraction logging
│   ├── dual_encoder_metrics.py        # Dual encoder profiling
│   ├── spatial_anomaly_detection.py   # Anomaly detection
│   └── test_monitoring_system.py      # Comprehensive testing
├── config/monitoring/
│   ├── prometheus.yml                 # Prometheus configuration
│   ├── extended-grafana-dashboard.json # Main dashboard
│   ├── spatial-anomaly-dashboard.json # Anomaly dashboard
│   └── spatial_rules.yml             # Alerting rules
└── logs/
    ├── spatial_features.log           # Feature extraction logs
    ├── dual_encoder.log              # Encoder performance logs
    └── anomaly_detection.log         # Anomaly detection logs
```

## Production Deployment

### Deployment Checklist
- [ ] Prometheus server configured and running
- [ ] Grafana dashboards imported and configured
- [ ] Alert rules deployed and tested
- [ ] Log rotation configured
- [ ] Monitoring endpoints accessible
- [ ] GPU monitoring enabled (if applicable)
- [ ] Test suite passing with production data
- [ ] Performance baselines established
- [ ] Alert notification channels configured

### Monitoring the Monitoring
- Set up meta-monitoring to track monitoring system health
- Monitor metrics ingestion rates
- Track dashboard query performance
- Alert on monitoring system failures

## Support and Maintenance

### Regular Maintenance Tasks
1. **Weekly**:
   - Review anomaly detection accuracy
   - Check dashboard performance
   - Validate alert configurations

2. **Monthly**:
   - Update detection baselines
   - Review and archive old metrics data
   - Performance optimization review

3. **Quarterly**:
   - Full system performance audit
   - Update monitoring documentation
   - Expand test coverage for new features

### Getting Help
- Check logs in `/tmp/monitoring_logs/`
- Run diagnostic tests with `test_monitoring_system.py`
- Review Prometheus targets at `http://localhost:9090/targets`
- Check Grafana dashboard health

## Changelog

### v2.0.0 (SPATIAL-4.3)
- ✅ Added comprehensive spatial feature logging
- ✅ Implemented dual-encoder performance monitoring
- ✅ Created spatial anomaly detection system
- ✅ Extended Grafana dashboards with spatial visualizations
- ✅ Added comprehensive testing framework
- ✅ Implemented GPU monitoring support
- ✅ Created specialized anomaly alert dashboard

### v1.0.0 (SPATIAL-4.2)
- Initial monitoring system with basic metrics
- Prometheus and Grafana integration
- Basic alerting rules

---

*This documentation is part of the SPATIAL-4.3 implementation for the Enhanced Spatial-MLLM Pizza Monitoring System.*
