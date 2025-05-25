#!/usr/bin/env bash
# Complete script to evaluate different input sizes, generate visualizations and reports

# Function to display usage information
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --sizes SIZE1 SIZE2...  Specify input sizes to evaluate (default: 32 40 48)"
    echo "  --skip-eval             Skip evaluation and only generate reports"
    echo "  --help                  Display this help message"
    echo ""
    echo "Example:"
    echo "  $0 --sizes 24 32 40 48"
}

# Parse command line arguments
SIZES=(32 40 48)
SKIP_EVAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --sizes)
            shift
            SIZES=()
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                SIZES+=("$1")
                shift
            done
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Create output directories
mkdir -p output/evaluation
mkdir -p output/visualization

echo "===== Pizza Detection Model: Input Size Evaluation ====="
echo ""

if [[ "$SKIP_EVAL" == "false" ]]; then
    echo "Step 1: Evaluating input sizes: ${SIZES[*]}"
    echo "(This process will take some time as it trains multiple models)"
    echo ""
    
    # Run the evaluation script with the specified sizes
    python scripts/evaluate_input_sizes.py --sizes "${SIZES[@]}"
    
    echo ""
    echo "Evaluation complete."
    echo ""
else
    echo "Skipping evaluation as requested."
    echo ""
fi

echo "Step 2: Generating visualizations and reports"
echo ""

# Generate visualization plots
echo "Generating visualization plots..."
python scripts/visualize_input_size_results.py

# Generate comprehensive report
echo "Generating comprehensive evaluation report..."
python scripts/generate_input_size_report.py

echo ""
echo "===== Input Size Evaluation Complete ====="
echo "Results available at:"
echo "- Evaluation data: output/evaluation/eval_size_XXxXX.json"
echo "- Summary: output/evaluation/input_size_summary.json"
echo "- Report: output/evaluation/input_size_report.md"
echo "- Plots: output/visualization/input_size_evaluation.png"
echo "        output/visualization/input_size_tradeoff.png"
