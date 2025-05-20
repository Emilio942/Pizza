#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Validation Tool for the Pizza Detection CI/CD Pipeline

This script analyzes the relationships between scripts and verifies that they 
properly integrate with each other in the pipeline.

Usage:
    python validate_integration.py --output REPORT_FILE --status-file STATUS_FILE [--log-dir LOG_DIR]
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# Set up logging
def setup_logging(log_dir: str) -> logging.Logger:
    logger = logging.getLogger("integration_validator")
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_dir is provided
    if log_dir:
        log_file = os.path.join(log_dir, "validate_integration.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Define script categories for better organization and dependency checking
SCRIPT_CATEGORIES = {
    "preprocessing": [
        "test_image_preprocessing.py",
        "simulate_clahe_resources.py",
        "extract_frames.py",
        "augment_dataset.py",
        "filter_dataset.py",
        "generate_pizza_dataset.py"
    ],
    "training": [
        "train_pizza_model.py",
        "hyperparameter_search.py"
    ],
    "optimization": [
        "compare_tiny_cnns.py",
        "compare_se_models.py",
        "compare_inverted_residual.py",
        "compare_hard_swish.py",
        "run_pruning_clustering.py",
        "knowledge_distillation.py",
        "quantization_aware_training.py",
        "weight_pruning.py",
        "run_optimization_pipeline.sh"
    ],
    "testing": [
        "run_pizza_tests.py",
        "automated_test_suite.py",
        "test_pizza_detection.py",
        "test_temporal_smoothing.py"
    ],
    "visualization": [
        "visualize_gradcam.py",
        "visualize_pruning.py",
        "demo_status_display.py"
    ],
    "simulation": [
        "simulate_lighting_conditions.py",
        "simulate_power_consumption.py",
        "simulate_memory_constraints.py",
        "demo_power_management.py",
        "framebuffer_demo.py"
    ],
    "integration": [
        "pizza_diffusion_integration.py",
        "diffusion_training_integration.py",
        "diffusion_data_agent.py"
    ],
    "utilities": [
        "cleanup.py",
        "classify_images.py",
        "analyze_performance_logs.py",
        "label_tool.py"
    ]
}

# Function to identify script category
def identify_script_category(script_name: str) -> str:
    for category, scripts in SCRIPT_CATEGORIES.items():
        if any(s.lower() == script_name.lower() for s in scripts):
            return category
    return "unknown"

# Function to analyze script relationships based on status file
def analyze_relationships(status_data: Dict) -> Dict:
    scripts = status_data.get("scripts", [])
    
    # Group scripts by category
    categorized = {}
    for script in scripts:
        name = script.get("name", "")
        category = identify_script_category(name)
        
        if category not in categorized:
            categorized[category] = []
        
        categorized[category].append(script)
    
    # Calculate category success rates
    category_stats = {}
    for category, scripts in categorized.items():
        total = len(scripts)
        successful = sum(1 for s in scripts if s.get("status") == "success")
        
        category_stats[category] = {
            "total": total,
            "successful": successful,
            "success_rate": round(successful / total * 100, 2) if total > 0 else 0,
            "scripts": scripts
        }
    
    return category_stats

# Function to identify potential integration issues
def identify_issues(category_stats: Dict, logger: logging.Logger) -> List[Dict]:
    issues = []
    
    # Check each category's success rate
    for category, stats in category_stats.items():
        if stats["success_rate"] < 100:
            logger.warning(f"Category '{category}' has a success rate of only {stats['success_rate']}%")
            
            for script in stats["scripts"]:
                if script["status"] != "success":
                    issues.append({
                        "type": "failed_script",
                        "category": category,
                        "script": script["name"],
                        "details": f"Script failed after {script['retries']} retries"
                    })
    
    # Check for missing category dependencies
    dependency_chain = [
        "preprocessing", 
        "training", 
        "optimization", 
        "testing", 
        "integration"
    ]
    
    for i in range(1, len(dependency_chain)):
        current = dependency_chain[i]
        previous = dependency_chain[i-1]
        
        if current in category_stats and previous in category_stats:
            if category_stats[previous]["success_rate"] < 50 and category_stats[current]["success_rate"] < 100:
                issues.append({
                    "type": "dependency_chain",
                    "categories": f"{previous} -> {current}",
                    "details": f"Low success rate in '{previous}' may have caused issues in '{current}'"
                })
    
    return issues

# Main function
def main():
    parser = argparse.ArgumentParser(description="Validate integration of scripts in the CI/CD Pipeline")
    parser.add_argument("--output", required=True, help="Output JSON file for integration report")
    parser.add_argument("--status-file", required=True, help="JSON file with script execution status")
    parser.add_argument("--log-dir", default=None, help="Directory for log files")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    
    # Load status data
    with open(args.status_file, "r", encoding="utf-8") as f:
        status_data = json.load(f)
    
    logger.info(f"Analyzing script relationships from status file: {args.status_file}")
    
    # Analyze relationships
    category_stats = analyze_relationships(status_data)
    
    # Identify issues
    issues = identify_issues(category_stats, logger)
    
    # Create integration report
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "category_stats": category_stats,
        "issues": issues,
        "summary": {
            "total_categories": len(category_stats),
            "total_issues": len(issues),
            "overall_success_rate": round(
                sum(s["success_rate"] for s in category_stats.values()) / len(category_stats), 
                2
            ) if category_stats else 0
        }
    }
    
    # Write report to file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    # Log summary
    logger.info(f"Integration validation complete. Found {len(issues)} potential issues.")
    logger.info(f"Report written to {args.output}")
    
    # Return non-zero exit code if there are critical issues
    critical_issues = [i for i in issues if i["type"] == "dependency_chain"]
    return 1 if critical_issues else 0

if __name__ == "__main__":
    # Import datetime here to avoid potential circular imports
    import datetime
    sys.exit(main())
