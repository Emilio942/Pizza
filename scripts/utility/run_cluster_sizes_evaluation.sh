#!/bin/bash
# Evaluate weight clustering with different cluster sizes and INT4 quantization

# Create output directory
OUTPUT_DIR="output/clustering_evaluation"
PLOTS_DIR="$OUTPUT_DIR/plots"
mkdir -p $OUTPUT_DIR
mkdir -p $PLOTS_DIR

# Display start message
echo "Starting weight clustering evaluation with different cluster sizes..."
echo "This will evaluate clusters of size 16, 32, and 64 with/without INT4 quantization"
echo "Results will be saved to: $OUTPUT_DIR"
echo ""

# Run the evaluation script
python scripts/evaluate_cluster_sizes.py \
    --model_path models/micro_pizza_model.pth \
    --output_dir $OUTPUT_DIR \
    --device cpu

# Check if the evaluation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Evaluation completed successfully!"
    echo "Report saved to: $OUTPUT_DIR/clustered_sizes_evaluation.md"
    echo "Detailed results saved to: $OUTPUT_DIR/all_results.json"
    
    # Generate visualizations
    echo ""
    echo "Generating visualization plots..."
    python scripts/visualize_clustering_results.py \
        --results_json $OUTPUT_DIR/all_results.json \
        --output_dir $PLOTS_DIR
    
    if [ $? -eq 0 ]; then
        echo "Visualization plots saved to: $PLOTS_DIR"
    else
        echo "Warning: Visualization generation failed."
    fi
    
    echo ""
    echo "To view the markdown report with proper formatting, use:"
    echo "  cat $OUTPUT_DIR/clustered_sizes_evaluation.md | less -R"
    echo ""
    echo "Models saved:"
    echo "  - $OUTPUT_DIR/clustered_16.pth"
    echo "  - $OUTPUT_DIR/clustered_32.pth"
    echo "  - $OUTPUT_DIR/clustered_64.pth"
    echo "  - $OUTPUT_DIR/int4_clustered_16.pth"
    echo "  - $OUTPUT_DIR/int4_clustered_32.pth"
    echo "  - $OUTPUT_DIR/int4_clustered_64.pth"
else
    echo ""
    echo "Error: Evaluation failed. Check the log for details."
    echo "Log file: cluster_sizes_evaluation.log"
fi
