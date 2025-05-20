# Pizza Detection System CI/CD Pipeline

This directory contains scripts for the Pizza Detection System's Continuous Integration and Continuous Deployment (CI/CD) pipeline. The pipeline automates the execution, validation, and integration of various scripts related to the Pizza Detection System.

## Overview

The CI/CD pipeline consists of the following main components:

1. **Script Validation**: Verifies all scripts for syntax errors, dependency issues, and basic execution.
2. **Preprocessing**: Handles image preprocessing and data augmentation.
3. **Model Optimization**: Executes model optimization scripts including comparison, pruning, and quantization.
4. **Testing**: Runs test suites to verify the functionality of the system.
5. **Integration Validation**: Checks how well scripts work together and identifies potential issues.
6. **Reporting**: Generates detailed reports with charts and recommendations.

## Pipeline Structure

The pipeline is structured as follows:

- **`run_pipeline.sh`**: The main entry point for the pipeline that orchestrates all phases.
- **`validate_scripts.py`**: Validates Python scripts for syntax errors, dependencies, and basic execution.
- **`validate_integration.py`**: Analyzes script relationships and identifies integration issues.
- **`generate_pipeline_report.py`**: Generates comprehensive HTML and Markdown reports from pipeline data.
- **`run_optimization_pipeline.sh`**: A specialized pipeline for model optimization scripts.

## Usage

To run the complete CI/CD pipeline:

```bash
cd /home/emilio/Documents/ai/pizza
./scripts/ci/run_pipeline.sh
```

### Running Only Script Validation

To only validate scripts:

```bash
cd /home/emilio/Documents/ai/pizza
./scripts/ci/validate_scripts.py --output validation_report.json --scripts-dir scripts
```

### Running Only the Optimization Pipeline

To only run the model optimization pipeline:

```bash
cd /home/emilio/Documents/ai/pizza
./scripts/ci/run_optimization_pipeline.sh
```

## Output

After running the pipeline, outputs and reports will be available in the following directory:

```
/home/emilio/Documents/ai/pizza/output/pipeline_runs/TIMESTAMP/
```

The most recent run is also symbolically linked to:

```
/home/emilio/Documents/ai/pizza/output/pipeline_latest/
```

### Reports

The pipeline generates the following reports:

- **HTML Report**: `pipeline_report.html` - A comprehensive report with charts and tables
- **Markdown Report**: `pipeline_report.md` - A plain text report with essential information
- **Script Status**: `script_status.json` - Detailed status of each script execution
- **Validation Report**: `validation_report.json` - Script validation results
- **Integration Report**: `integration_report.json` - Script integration analysis

## Troubleshooting

If the pipeline fails, check the generated logs in:

```
/home/emilio/Documents/ai/pizza/output/pipeline_runs/TIMESTAMP/logs/
```

For specific script failures, check the individual log files named after each script.

## Maintenance

### Adding New Scripts

When adding new scripts to the Pizza Detection System:

1. Ensure they follow the same structure and error handling conventions as existing scripts
2. Update `validate_integration.py` to include the new scripts in the appropriate categories

### Updating Dependencies

If you update dependencies:

1. Run the validation script to verify all dependencies are still satisfied
2. Update `requirements.txt` with the new dependencies

## Pipeline Maintenance Schedule

For optimal performance:

- Run the complete pipeline after major code changes
- Run the validation script daily to catch early issues
- Clean old pipeline runs periodically (older than 30 days)
- Review and update error handling as needed
