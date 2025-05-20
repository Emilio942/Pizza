#!/bin/bash
# Install dependencies for Pizza Detection CI/CD Pipeline

echo "Installing dependencies for Pizza Detection CI/CD Pipeline..."
pip install -r requirements.txt 

# Make the pipeline scripts executable
echo "Making pipeline scripts executable..."
chmod +x scripts/ci/run_pipeline.sh
chmod +x scripts/ci/validate_scripts.py
chmod +x scripts/ci/validate_integration.py
chmod +x scripts/ci/generate_pipeline_report.py

echo "Creating necessary directories..."
mkdir -p output/pipeline_runs
mkdir -p output/preprocessing
mkdir -p output/test_results
mkdir -p output/optimization

echo "Dependencies installed. You can now run the pipeline with:"
echo "./scripts/ci/run_pipeline.sh"
