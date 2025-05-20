#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Report Generator for the Pizza Detection CI/CD Pipeline

This script generates comprehensive HTML and Markdown reports from pipeline data,
including script validation results, integration analysis, and performance metrics.

Usage:
    python generate_pipeline_report.py --status-file STATUS_FILE --validation-report VALIDATION_REPORT
                                      --integration-report INTEGRATION_REPORT --output-dir OUTPUT_DIR
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("report_generator")

def load_json_file(file_path: str) -> Dict:
    """Load and parse a JSON file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {str(e)}")
        return {}

def generate_status_charts(status_data: Dict, output_dir: str) -> List[str]:
    """Generate charts for script execution status"""
    os.makedirs(output_dir, exist_ok=True)
    chart_files = []
    
    # Extract script data
    scripts = status_data.get("scripts", [])
    
    if not scripts:
        return chart_files
    
    # Chart 1: Status distribution
    statuses = [s.get("status", "unknown") for s in scripts]
    status_counts = {"success": 0, "failed": 0, "warning": 0, "unknown": 0}
    
    for status in statuses:
        if status in status_counts:
            status_counts[status] += 1
        else:
            status_counts["unknown"] += 1
    
    # Create pie chart
    plt.figure(figsize=(8, 6))
    labels = list(status_counts.keys())
    sizes = list(status_counts.values())
    colors = ['#4CAF50', '#F44336', '#FFC107', '#9E9E9E']  # green, red, yellow, gray
    
    # Only include non-zero slices
    filtered_labels = []
    filtered_sizes = []
    filtered_colors = []
    
    for i, size in enumerate(sizes):
        if size > 0:
            filtered_labels.append(labels[i])
            filtered_sizes.append(size)
            filtered_colors.append(colors[i])
    
    plt.pie(
        filtered_sizes, 
        labels=filtered_labels, 
        colors=filtered_colors,
        autopct='%1.1f%%', 
        startangle=90, 
        shadow=True
    )
    plt.axis('equal')
    plt.title('Script Execution Status')
    
    status_chart_path = os.path.join(output_dir, "status_distribution.png")
    plt.savefig(status_chart_path)
    plt.close()
    chart_files.append(status_chart_path)
    
    # Chart 2: Script execution duration
    script_names = [s.get("name", "unknown") for s in scripts]
    durations = [float(s.get("duration", 0)) for s in scripts]
    
    if len(script_names) > 20:
        # If there are too many scripts, show only the top 20 by duration
        sorted_indices = np.argsort(durations)[::-1][:20]
        script_names = [script_names[i] for i in sorted_indices]
        durations = [durations[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(script_names, durations)
    
    # Color bars based on status
    for i, idx in enumerate(sorted_indices if len(script_names) > 20 else range(len(scripts))):
        status = scripts[idx].get("status", "unknown")
        if status == "success":
            bars[i].set_color('#4CAF50')  # green
        elif status == "failed":
            bars[i].set_color('#F44336')  # red
        elif status == "warning":
            bars[i].set_color('#FFC107')  # yellow
        else:
            bars[i].set_color('#9E9E9E')  # gray
    
    plt.xlabel('Execution Duration (seconds)')
    plt.title('Script Execution Duration')
    plt.tight_layout()
    
    duration_chart_path = os.path.join(output_dir, "execution_duration.png")
    plt.savefig(duration_chart_path)
    plt.close()
    chart_files.append(duration_chart_path)
    
    return chart_files

def generate_validation_charts(validation_data: Dict, output_dir: str) -> List[str]:
    """Generate charts for script validation results"""
    os.makedirs(output_dir, exist_ok=True)
    chart_files = []
    
    summary = validation_data.get("summary", {})
    if not summary:
        return chart_files
    
    # Chart: Validation results
    labels = ['Passed', 'Warnings', 'Failed']
    sizes = [
        summary.get("passed", 0),
        summary.get("warnings", 0),
        summary.get("failed", 0)
    ]
    colors = ['#4CAF50', '#FFC107', '#F44336']  # green, yellow, red
    
    # Remove zero values
    filtered_labels = []
    filtered_sizes = []
    filtered_colors = []
    
    for i, size in enumerate(sizes):
        if size > 0:
            filtered_labels.append(labels[i])
            filtered_sizes.append(size)
            filtered_colors.append(colors[i])
    
    plt.figure(figsize=(8, 6))
    plt.pie(
        filtered_sizes, 
        labels=filtered_labels,
        colors=filtered_colors, 
        autopct='%1.1f%%', 
        startangle=90, 
        shadow=True
    )
    plt.axis('equal')
    plt.title('Script Validation Results')
    
    validation_chart_path = os.path.join(output_dir, "validation_results.png")
    plt.savefig(validation_chart_path)
    plt.close()
    chart_files.append(validation_chart_path)
    
    return chart_files

def generate_category_success_chart(integration_data: Dict, output_dir: str) -> List[str]:
    """Generate charts for category success rates"""
    os.makedirs(output_dir, exist_ok=True)
    chart_files = []
    
    category_stats = integration_data.get("category_stats", {})
    if not category_stats:
        return chart_files
    
    # Chart: Category success rates
    categories = list(category_stats.keys())
    success_rates = [stats.get("success_rate", 0) for stats in category_stats.values()]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, success_rates)
    
    # Color bars based on success rate
    for i, rate in enumerate(success_rates):
        if rate >= 90:
            bars[i].set_color('#4CAF50')  # green
        elif rate >= 70:
            bars[i].set_color('#FFEB3B')  # yellow
        elif rate >= 50:
            bars[i].set_color('#FF9800')  # orange
        else:
            bars[i].set_color('#F44336')  # red
    
    plt.axhline(y=90, color='#4CAF50', linestyle='--', alpha=0.7)
    plt.axhline(y=70, color='#FFEB3B', linestyle='--', alpha=0.7)
    plt.axhline(y=50, color='#FF9800', linestyle='--', alpha=0.7)
    
    plt.xlabel('Script Category')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate by Script Category')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    
    for i, v in enumerate(success_rates):
        plt.text(i, v + 2, f"{v}%", ha='center')
    
    plt.tight_layout()
    
    category_chart_path = os.path.join(output_dir, "category_success_rates.png")
    plt.savefig(category_chart_path)
    plt.close()
    chart_files.append(category_chart_path)
    
    return chart_files

def generate_html_report(
    status_data: Dict, 
    validation_data: Dict, 
    integration_data: Dict, 
    chart_files: List[str],
    output_dir: str
) -> str:
    """Generate an HTML report from the data"""
    pipeline_start_str = status_data.get("pipeline_start", datetime.now().isoformat())
    pipeline_end_str = status_data.get("pipeline_end", datetime.now().isoformat())
    
    # Ensure both datetimes use the same format (with or without timezone info)
    if 'Z' in pipeline_start_str and 'Z' not in pipeline_end_str:
        pipeline_end_str = pipeline_end_str + 'Z'
    elif 'Z' not in pipeline_start_str and 'Z' in pipeline_end_str:
        pipeline_start_str = pipeline_start_str + 'Z'
        
    pipeline_start = datetime.fromisoformat(pipeline_start_str.replace('Z', '+00:00'))
    pipeline_end = datetime.fromisoformat(pipeline_end_str.replace('Z', '+00:00'))
    
    total_duration = (pipeline_end - pipeline_start).total_seconds() / 60.0  # in minutes
    
    scripts = status_data.get("scripts", [])
    successful_scripts = [s for s in scripts if s.get("status") == "success"]
    failed_scripts = [s for s in scripts if s.get("status") == "failed"]
    
    validation_summary = validation_data.get("summary", {})
    validation_results = validation_data.get("results", [])
    
    integration_summary = integration_data.get("summary", {})
    integration_issues = integration_data.get("issues", [])
    category_stats = integration_data.get("category_stats", {})
    
    # Create relative paths for charts
    chart_paths = [os.path.basename(f) for f in chart_files]
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pizza Detection Pipeline Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3, h4 {{
                color: #2c3e50;
            }}
            .summary-box {{
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .success {{
                background-color: #e8f5e9;
                border-left: 5px solid #4CAF50;
            }}
            .warning {{
                background-color: #fff8e1;
                border-left: 5px solid #FFC107;
            }}
            .error {{
                background-color: #ffebee;
                border-left: 5px solid #F44336;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f8f9fa;
                font-weight: bold;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .chart-container {{
                display: flex;
                justify-content: center;
                margin: 20px 0;
                flex-wrap: wrap;
            }}
            .chart {{
                max-width: 100%;
                height: auto;
                margin: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                border-radius: 5px;
            }}
            .badge {{
                display: inline-block;
                padding: 5px 10px;
                border-radius: 15px;
                color: white;
                font-size: 12px;
                margin-right: 5px;
            }}
            .badge-success {{ background-color: #4CAF50; }}
            .badge-warning {{ background-color: #FFC107; color: black; }}
            .badge-error {{ background-color: #F44336; }}
            .badge-info {{ background-color: #2196F3; }}
        </style>
    </head>
    <body>
        <h1>Pizza Detection Pipeline Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary-box {'success' if len(failed_scripts) == 0 else 'warning' if len(failed_scripts) <= len(scripts) / 5 else 'error'}">
            <h2>Pipeline Summary</h2>
            <p><strong>Duration:</strong> {total_duration:.2f} minutes</p>
            <p><strong>Total Scripts:</strong> {len(scripts)}</p>
            <p><strong>Success Rate:</strong> {len(successful_scripts) / len(scripts) * 100:.2f}% ({len(successful_scripts)} successful, {len(failed_scripts)} failed)</p>
            <p><strong>Overall Status:</strong> {'Success' if len(failed_scripts) == 0 else 'Partial Success' if len(failed_scripts) <= len(scripts) / 5 else 'Failure'}</p>
        </div>
        
        <div class="chart-container">
    """
    
    # Add charts
    for chart_path in chart_paths:
        html_content += f'<img src="{chart_path}" alt="Pipeline Chart" class="chart">\n'
    
    html_content += """
        </div>
        
        <h2>Script Execution Details</h2>
        <table>
            <tr>
                <th>Script</th>
                <th>Status</th>
                <th>Duration (s)</th>
                <th>Retries</th>
            </tr>
    """
    
    # Add script rows
    for script in scripts:
        status_class = "badge-success" if script.get("status") == "success" else "badge-error"
        status_text = script.get("status", "unknown").capitalize()
        
        html_content += f"""
            <tr>
                <td>{script.get("name", "Unknown")}</td>
                <td><span class="badge {status_class}">{status_text}</span></td>
                <td>{script.get("duration", 0):.2f}</td>
                <td>{script.get("retries", 0)}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Validation Results</h2>
    """
    
    # Add validation summary
    if validation_summary:
        html_content += f"""
        <div class="summary-box {'success' if validation_summary.get('failed', 0) == 0 else 'warning'}">
            <p><strong>Total Scripts Validated:</strong> {validation_summary.get("total_scripts", 0)}</p>
            <p><strong>Passed:</strong> {validation_summary.get("passed", 0)}</p>
            <p><strong>With Warnings:</strong> {validation_summary.get("warnings", 0)}</p>
            <p><strong>Failed:</strong> {validation_summary.get("failed", 0)}</p>
            <p><strong>Pass Rate:</strong> {validation_summary.get("pass_rate", 0)}%</p>
        </div>
        """
    
    # Add validation details if more than 5 failed validations
    failed_validations = [r for r in validation_results if r.get("status") == "failed"]
    if len(failed_validations) > 0:
        html_content += """
        <h3>Failed Validations</h3>
        <table>
            <tr>
                <th>Script</th>
                <th>Errors</th>
            </tr>
        """
        
        for result in failed_validations:
            html_content += f"""
            <tr>
                <td>{result.get("file_name", "Unknown")}</td>
                <td>{"<br>".join(result.get("errors", ["Unknown error"]))}</td>
            </tr>
            """
        
        html_content += "</table>"
    
    # Add integration results
    html_content += """
        <h2>Integration Analysis</h2>
    """
    
    if integration_summary:
        html_content += f"""
        <div class="summary-box {'success' if integration_summary.get('total_issues', 0) == 0 else 'warning'}">
            <p><strong>Overall Success Rate:</strong> {integration_summary.get("overall_success_rate", 0)}%</p>
            <p><strong>Total Categories:</strong> {integration_summary.get("total_categories", 0)}</p>
            <p><strong>Total Issues:</strong> {integration_summary.get("total_issues", 0)}</p>
        </div>
        """
    
    # Add category stats
    if category_stats:
        html_content += """
        <h3>Category Success Rates</h3>
        <table>
            <tr>
                <th>Category</th>
                <th>Success Rate</th>
                <th>Successful Scripts</th>
                <th>Total Scripts</th>
            </tr>
        """
        
        for category, stats in category_stats.items():
            success_rate = stats.get("success_rate", 0)
            status_class = "badge-success" if success_rate >= 90 else "badge-warning" if success_rate >= 70 else "badge-error"
            
            html_content += f"""
            <tr>
                <td>{category}</td>
                <td><span class="badge {status_class}">{success_rate}%</span></td>
                <td>{stats.get("successful", 0)}</td>
                <td>{stats.get("total", 0)}</td>
            </tr>
            """
        
        html_content += "</table>"
    
    # Add integration issues
    if integration_issues:
        html_content += """
        <h3>Integration Issues</h3>
        <table>
            <tr>
                <th>Type</th>
                <th>Details</th>
            </tr>
        """
        
        for issue in integration_issues:
            html_content += f"""
            <tr>
                <td>{issue.get("type", "Unknown").replace("_", " ").title()}</td>
                <td>{issue.get("details", "No details available")}</td>
            </tr>
            """
        
        html_content += "</table>"
    
    # Close HTML
    html_content += """
        <h2>Recommendations</h2>
        <ul>
    """
    
    # Generate recommendations based on issues
    recommendations = []
    
    if failed_scripts:
        recommendations.append("Review and fix the failed scripts to improve pipeline reliability.")
    
    if failed_validations:
        recommendations.append("Fix syntax and dependency issues in scripts that failed validation.")
    
    if integration_issues:
        recommendations.append("Address integration issues between script categories.")
    
    for category, stats in category_stats.items():
        if stats.get("success_rate", 0) < 70:
            recommendations.append(f"Focus on improving scripts in the '{category}' category.")
    
    if not recommendations:
        recommendations.append("All scripts are running well. Consider adding more tests or optimizations.")
    
    for recommendation in recommendations:
        html_content += f"<li>{recommendation}</li>\n"
    
    html_content += """
        </ul>
        
        <hr>
        <footer>
            <p>Pizza Detection CI/CD Pipeline</p>
            <p>Generated automatically</p>
        </footer>
    </body>
    </html>
    """
    
    # Write HTML file
    html_path = os.path.join(output_dir, "pipeline_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return html_path

def generate_markdown_report(
    status_data: Dict, 
    validation_data: Dict, 
    integration_data: Dict,
    output_dir: str
) -> str:
    """Generate a Markdown report from the data"""
    # Set default values for pipeline times
    current_time = datetime.now()
    default_time_str = current_time.replace(microsecond=0).isoformat()
    
    # Get pipeline start and end times, with defaults
    pipeline_start_str = status_data.get("pipeline_start", default_time_str)
    pipeline_end_str = status_data.get("pipeline_end", default_time_str) if "pipeline_end" in status_data else default_time_str
    
    # Make sure both are in the same format
    # Strip any timezone info to ensure both are naive datetimes
    if '+' in pipeline_start_str:
        pipeline_start_str = pipeline_start_str.split('+')[0]
    if 'Z' in pipeline_start_str:
        pipeline_start_str = pipeline_start_str.replace('Z', '')
    if '+' in pipeline_end_str:
        pipeline_end_str = pipeline_end_str.split('+')[0]
    if 'Z' in pipeline_end_str:
        pipeline_end_str = pipeline_end_str.replace('Z', '')
    
    try:
        # Try to parse the ISO format
        pipeline_start = datetime.fromisoformat(pipeline_start_str)
        pipeline_end = datetime.fromisoformat(pipeline_end_str)
        total_duration = (pipeline_end - pipeline_start).total_seconds() / 60.0  # in minutes
    except (ValueError, TypeError) as e:
        # If parsing fails, use a default duration
        logger.warning(f"Error parsing datetime: {e}, using default duration")
        total_duration = 0.0
    
    scripts = status_data.get("scripts", [])
    successful_scripts = [s for s in scripts if s.get("status") == "success"]
    failed_scripts = [s for s in scripts if s.get("status") == "failed"]
    
    validation_summary = validation_data.get("summary", {})
    validation_results = validation_data.get("results", [])
    
    integration_summary = integration_data.get("summary", {})
    integration_issues = integration_data.get("issues", [])
    category_stats = integration_data.get("category_stats", {})
    
    # Start Markdown content
    md_content = f"""# Pizza Detection Pipeline Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Pipeline Summary

- **Duration:** {total_duration:.2f} minutes
- **Total Scripts:** {len(scripts)}
- **Success Rate:** {len(successful_scripts) / len(scripts) * 100:.2f}% ({len(successful_scripts)} successful, {len(failed_scripts)} failed)
- **Overall Status:** {'Success' if len(failed_scripts) == 0 else 'Partial Success' if len(failed_scripts) <= len(scripts) / 5 else 'Failure'}

## Script Execution Details

| Script | Status | Duration (s) | Retries |
|--------|--------|--------------|---------|
"""
    
    # Add script rows
    for script in scripts:
        status_text = script.get("status", "unknown").capitalize()
        md_content += f"| {script.get('name', 'Unknown')} | {status_text} | {script.get('duration', 0):.2f} | {script.get('retries', 0)} |\n"
    
    # Add validation summary
    md_content += "\n## Validation Results\n\n"
    
    if validation_summary:
        md_content += f"""- **Total Scripts Validated:** {validation_summary.get("total_scripts", 0)}
- **Passed:** {validation_summary.get("passed", 0)}
- **With Warnings:** {validation_summary.get("warnings", 0)}
- **Failed:** {validation_summary.get("failed", 0)}
- **Pass Rate:** {validation_summary.get("pass_rate", 0)}%

"""
    
    # Add failed validations
    failed_validations = [r for r in validation_results if r.get("status") == "failed"]
    if failed_validations:
        md_content += "### Failed Validations\n\n"
        md_content += "| Script | Errors |\n|--------|--------|\n"
        
        for result in failed_validations:
            errors = "<br>".join(result.get("errors", ["Unknown error"]))
            md_content += f"| {result.get('file_name', 'Unknown')} | {errors} |\n"
    
    # Add integration results
    md_content += "\n## Integration Analysis\n\n"
    
    if integration_summary:
        md_content += f"""- **Overall Success Rate:** {integration_summary.get("overall_success_rate", 0)}%
- **Total Categories:** {integration_summary.get("total_categories", 0)}
- **Total Issues:** {integration_summary.get("total_issues", 0)}

"""
    
    # Add category stats
    if category_stats:
        md_content += "### Category Success Rates\n\n"
        md_content += "| Category | Success Rate | Successful Scripts | Total Scripts |\n|----------|--------------|-------------------|---------------|\n"
        
        for category, stats in category_stats.items():
            md_content += f"| {category} | {stats.get('success_rate', 0)}% | {stats.get('successful', 0)} | {stats.get('total', 0)} |\n"
    
    # Add integration issues
    if integration_issues:
        md_content += "\n### Integration Issues\n\n"
        md_content += "| Type | Details |\n|------|--------|\n"
        
        for issue in integration_issues:
            issue_type = issue.get("type", "Unknown").replace("_", " ").title()
            md_content += f"| {issue_type} | {issue.get('details', 'No details available')} |\n"
    
    # Add recommendations
    md_content += "\n## Recommendations\n\n"
    
    # Generate recommendations based on issues
    recommendations = []
    
    if failed_scripts:
        recommendations.append("- Review and fix the failed scripts to improve pipeline reliability.")
    
    if failed_validations:
        recommendations.append("- Fix syntax and dependency issues in scripts that failed validation.")
    
    if integration_issues:
        recommendations.append("- Address integration issues between script categories.")
    
    for category, stats in category_stats.items():
        if stats.get("success_rate", 0) < 70:
            recommendations.append(f"- Focus on improving scripts in the '{category}' category.")
    
    if not recommendations:
        recommendations.append("- All scripts are running well. Consider adding more tests or optimizations.")
    
    md_content += "\n".join(recommendations)
    
    # Add failed script details if any
    if failed_scripts:
        md_content += "\n\n## Failed Scripts Details\n\n"
        
        for script in failed_scripts:
            md_content += f"### {script.get('name', 'Unknown')}\n\n"
            md_content += f"- **Duration:** {script.get('duration', 0):.2f} seconds\n"
            md_content += f"- **Retries:** {script.get('retries', 0)}\n"
            md_content += f"- **Log File:** {script.get('log', 'N/A')}\n\n"
    
    # Write Markdown file
    md_path = os.path.join(output_dir, "pipeline_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    return md_path

def main():
    parser = argparse.ArgumentParser(description="Generate Pipeline Report")
    parser.add_argument("--status-file", required=True, help="JSON file with script execution status")
    parser.add_argument("--validation-report", required=True, help="JSON file with script validation results")
    parser.add_argument("--integration-report", required=True, help="JSON file with integration analysis")
    parser.add_argument("--output-dir", required=True, help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    charts_dir = os.path.join(args.output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    # Load data files
    logger.info("Loading data files...")
    status_data = load_json_file(args.status_file)
    validation_data = load_json_file(args.validation_report)
    integration_data = load_json_file(args.integration_report)
    
    # Generate charts
    logger.info("Generating charts...")
    status_charts = generate_status_charts(status_data, charts_dir)
    validation_charts = generate_validation_charts(validation_data, charts_dir)
    category_charts = generate_category_success_chart(integration_data, charts_dir)
    
    all_charts = status_charts + validation_charts + category_charts
    
    # Generate HTML report
    logger.info("Generating HTML report...")
    html_path = generate_html_report(
        status_data, 
        validation_data, 
        integration_data, 
        all_charts,
        args.output_dir
    )
    
    # Generate Markdown report
    logger.info("Generating Markdown report...")
    md_path = generate_markdown_report(
        status_data, 
        validation_data, 
        integration_data,
        args.output_dir
    )
    
    logger.info(f"HTML report generated: {html_path}")
    logger.info(f"Markdown report generated: {md_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
