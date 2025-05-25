#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a comprehensive report of the input size evaluation results.

This script creates a detailed report in Markdo        # Add trade-off analysis if we have multiple sizes
        if len(size_keys) > 1:
            largest_size = size_keys[-1]
            largest_data = data[largest_size]
            largest_accuracy = largest_data.get("accuracy")
            
            # Skip accuracy comparison if values are None
            if largest_accuracy is not None and best_size["accuracy"] is not None:
                accuracy_loss = largest_accuracy - best_size["accuracy"]
                if accuracy_loss > 0:
                    report.extend([
                        f"- Using {best_size['size']} instead of {largest_size} results in an accuracy loss of {accuracy_loss:.2%}.",
                        f"- However, it reduces RAM usage by {largest_data['ram_usage']['total_ram_kb'] - best_size['ram_kb']:.2f} KB.",
                        ""
                    ])at includes:
1. Accuracy results for each size
2. RAM usage breakdown for each size
3. Trade-off analysis
4. Recommendation for optimal size based on constraints

The report is saved to output/evaluation/input_size_report.md
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Directories
EVAL_DIR = project_root / "output" / "evaluation"
OUTPUT_DIR = project_root / "output" / "evaluation"

def load_evaluation_data():
    """
    Load detailed evaluation data from all size reports
    
    Returns:
        Dictionary with size as key and detailed results as value
    """
    results = {}
    
    for eval_file in EVAL_DIR.glob("eval_size_*.json"):
        with open(eval_file, 'r') as f:
            data = json.load(f)
            size = data.get("input_size")
            if size:
                size_key = f"{size}x{size}"
                results[size_key] = data
    
    return results

def generate_markdown_report(data):
    """
    Generate a detailed markdown report
    
    Args:
        data: Dictionary with detailed evaluation results by size
    
    Returns:
        String containing the markdown report
    """
    # Sort sizes
    size_keys = sorted(data.keys(), key=lambda x: int(x.split('x')[0]))
    
    # Create report header
    report = [
        "# Input Size Evaluation Report",
        f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Overview",
        "",
        "This report evaluates the impact of different input image sizes on:",
        "- Model accuracy",
        "- RAM usage (framebuffer size + tensor arena size)",
        "",
        "The target is to find an optimal input size that balances accuracy and RAM usage, ",
        "with a constraint of staying under 204KB total RAM.",
        "",
        "## Results Summary",
        "",
        "| Size | Accuracy | Framebuffer (KB) | Tensor Arena (KB) | Total RAM (KB) | % of RP2040 RAM |",
        "|------|----------|-----------------|------------------|---------------|-----------------|"
    ]
    
    # Add summary table rows
    for size_key in size_keys:
        size_data = data[size_key]
        accuracy = size_data.get("accuracy")
        if accuracy is None:
            accuracy_str = "N/A"
        else:
            accuracy_str = f"{accuracy:.2%}"
        
        ram_usage = size_data.get("ram_usage", {})
        framebuffer_kb = ram_usage.get("framebuffer_kb", "N/A")
        tensor_arena_kb = ram_usage.get("tensor_arena_kb", "N/A")
        total_ram_kb = ram_usage.get("total_ram_kb", "N/A")
        ram_percentage = ram_usage.get("percentage_of_rp2040_ram", "N/A")
        
        row = f"| {size_key} | {accuracy_str} | {framebuffer_kb} | {tensor_arena_kb} | {total_ram_kb} | {ram_percentage}% |"
        report.append(row)
    
    # Add detailed results for each size
    report.extend([
        "",
        "## Detailed Results by Size",
        ""
    ])
    
    for size_key in size_keys:
        size_data = data[size_key]
        size = size_key.split('x')[0]
        accuracy = size_data.get("accuracy", "N/A")
        if accuracy is None:
            accuracy_str = "N/A"
        else:
            accuracy_str = f"{accuracy:.2%}"
        
        ram_usage = size_data.get("ram_usage", {})
        
        report.extend([
            f"### Size: {size_key}",
            "",
            f"- **Accuracy**: {accuracy_str}",
            f"- **RAM Usage**:",
            f"  - Framebuffer: {ram_usage.get('framebuffer_kb', 'N/A')} KB",
            f"  - Tensor Arena: {ram_usage.get('tensor_arena_kb', 'N/A')} KB",
            f"  - **Total**: {ram_usage.get('total_ram_kb', 'N/A')} KB ({ram_usage.get('percentage_of_rp2040_ram', 'N/A')}% of RP2040's 264KB RAM)",
            ""
        ])
    
    # Analysis and recommendation
    report.extend([
        "## Analysis and Recommendation",
        ""
    ])
    
    # Find the optimal size based on RAM constraint and accuracy
    valid_sizes = []
    for size_key in size_keys:
        size_data = data[size_key]
        ram_usage = size_data.get("ram_usage", {})
        total_ram_kb = ram_usage.get("total_ram_kb", float('inf'))
        
        if total_ram_kb <= 204:  # RAM constraint
            valid_sizes.append({
                "size": size_key,
                "accuracy": size_data.get("accuracy", 0),
                "ram_kb": total_ram_kb
            })
    
    if valid_sizes:
        # Sort by accuracy (descending), handling None values
        valid_sizes.sort(key=lambda x: x["accuracy"] if x["accuracy"] is not None else 0, reverse=True)
        best_size = valid_sizes[0]
        
        accuracy_str = "N/A"
        if best_size['accuracy'] is not None:
            accuracy_str = f"{best_size['accuracy']:.2%}"
        
        report.extend([
            "### Recommendation",
            "",
            f"Based on the evaluation, the optimal input size is **{best_size['size']}**:",
            "",
            f"- It achieves the best accuracy ({accuracy_str}) among sizes that fit within the RAM constraint.",
            f"- It uses {best_size['ram_kb']} KB of RAM, which is below the 204KB limit.",
            "",
            "### Trade-offs",
            ""
        ])
        
        # Add trade-off analysis
        if len(size_keys) > 1:
            largest_size = size_keys[-1]
            largest_data = data[largest_size]
            largest_accuracy = largest_data.get("accuracy", 0)
            
            accuracy_loss = largest_accuracy - best_size["accuracy"]
            if accuracy_loss > 0:
                report.extend([
                    f"- Using {best_size['size']} instead of {largest_size} results in an accuracy loss of {accuracy_loss:.2%}.",
                    f"- However, it reduces RAM usage by {largest_data['ram_usage']['total_ram_kb'] - best_size['ram_kb']:.2f} KB.",
                    ""
                ])
    else:
        report.extend([
            "None of the evaluated sizes meet the RAM constraint of 204KB.",
            "You may need to explore even smaller input sizes or more aggressive model optimization techniques.",
            ""
        ])
    
    # Conclusion
    report.extend([
        "## Conclusion",
        "",
        "The evaluation demonstrates the trade-off between input image size, model accuracy, and RAM usage. ",
        "Smaller input sizes significantly reduce RAM requirements but at the cost of some accuracy. ",
        "For the RP2040 microcontroller with its 264KB RAM constraint, choosing the right input size is crucial ",
        "to balance performance and memory usage."
    ])
    
    return "\n".join(report)

def main():
    """Main function"""
    print("Generating comprehensive input size evaluation report...")
    
    # Load data
    eval_data = load_evaluation_data()
    
    if not eval_data:
        print("No evaluation data found. Please run the evaluation first.")
        return
    
    # Generate report
    report_md = generate_markdown_report(eval_data)
    
    # Save report
    report_path = OUTPUT_DIR / "input_size_report.md"
    with open(report_path, 'w') as f:
        f.write(report_md)
    
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
