#!/bin/bash
# Run formal verification examples

# Create directories for reports
mkdir -p "$(dirname "$0")/reports"
mkdir -p "$(dirname "$0")/reports/visualizations"

echo "=========================================="
echo "Running basic verification example..."
echo "=========================================="
python "$(dirname "$0")/verify_model_example.py"

echo
echo "=========================================="
echo "Running batch verification for model comparison..."
echo "=========================================="
python "$(dirname "$0")/batch_verification.py" --max-images 3 --epsilon 0.03 --output-dir "$(dirname "$0")/reports"

echo
echo "=========================================="
echo "Running comprehensive verification test..."
echo "=========================================="
python "$(dirname "$0")/test_verification.py" --max-images 5 --epsilon 0.03 --output-dir "$(dirname "$0")/reports" --only-mock

echo
echo "Verification examples completed. See the 'reports' directory for results."
