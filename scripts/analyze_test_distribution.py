#!/usr/bin/env python3
"""
Analyze Class Distribution - Test Set
This script analyzes the distribution of pizza images across different classes in the test dataset,
writes a JSON report and generates bar/pie charts.
Also verifies that no test images are present in the training/source datasets.
"""
import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import datetime


def verify_no_leakage(test_image_basenames, train_dirs_absolute_paths):
    """
    Return list of unique basenames present in both test_image_basenames and any of the train_dirs.
    """
    train_image_basenames = set()
    for td_path_str in train_dirs_absolute_paths:
        td_path = Path(td_path_str)
        if not td_path.is_dir(): # Check if it exists and is a directory
            # Optionally print a warning if a configured train_dir is not found/valid
            # print(f"Warning: Training directory {td_path_str} not found or is not a directory for leakage check.")
            continue
        # Recursively find all common image types in the training directory
        for ext in ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG']:
            for image_path in td_path.rglob(f'*.{ext}'):
                train_image_basenames.add(image_path.name)

    # Find intersection of test image basenames and training image basenames
    leaked_basenames = set(test_image_basenames).intersection(train_image_basenames)
    return sorted(list(leaked_basenames))


def analyze_test_distribution():
    project_root = Path(__file__).resolve().parent.parent
    class_file = project_root / "data" / "class_definitions.json"

    if not class_file.is_file():
        print(f"Error: Class definitions file not found at {class_file}")
        return

    try:
        with open(class_file, 'r') as f:
            class_definitions = json.load(f)
        if not isinstance(class_definitions, dict):
            print(f"Error: {class_file} is not in the expected format (should be a dictionary of classes).")
            return
        classes = list(class_definitions.keys()) # Ensure it's a list for consistent ordering if needed
        if not classes:
            print(f"Error: No classes found in {class_file}.")
            return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {class_file}")
        return
    except Exception as e:
        print(f"Error reading or parsing {class_file}: {e}")
        return

    test_root = project_root / "data" / "test"
    if not test_root.is_dir():
        print(f"Error: Test directory {test_root} not found or is not a directory.")
        return

    output_dir = project_root / "output" / "data_analysis"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create output directory {output_dir}: {e}")
        return

    class_counts = {}
    all_test_image_basenames = []

    for cls_name in classes:
        cls_dir = test_root / cls_name
        current_class_image_count = 0
        if cls_dir.is_dir():
            for ext in ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG']:
                image_paths_for_ext = list(cls_dir.rglob(f'*.{ext}'))
                current_class_image_count += len(image_paths_for_ext)
                for img_path in image_paths_for_ext:
                    all_test_image_basenames.append(img_path.name)
            class_counts[cls_name] = current_class_image_count
        else:
            class_counts[cls_name] = 0
            if cls_dir.exists():
                print(f"Warning: Expected directory for class '{cls_name}' at {cls_dir}, but found a file.")

    total_images = sum(class_counts.values())
    percentages = {cls: round((count / total_images) * 100, 2) if total_images > 0 else 0
                   for cls, count in class_counts.items()}

    potential_train_source_dirs_relative = [
        'augmented_pizza', 'augmented_pizza_legacy', 'data/raw',
        'data/processed', 'data/classified', 'data/augmented',
        'data/train', 'data/validation' # Add common train/validation folder names
    ]
    train_dirs_absolute = [str(project_root / td_rel) for td_rel in potential_train_source_dirs_relative]
    
    # Filter to only include directories that actually exist for the report
    existing_train_dirs_for_report = [
        str(Path(p).relative_to(project_root)) for p in train_dirs_absolute if Path(p).is_dir()
    ]

    leaked_files = verify_no_leakage(all_test_image_basenames, train_dirs_absolute)

    report = {
        'dataset_name': 'test_set_distribution',
        'analysis_timestamp': datetime.datetime.now().isoformat(),
        'class_counts': class_counts,
        'total_images': total_images,
        'percentages': percentages,
        'leakage_check': {
            'checked_against_dirs': existing_train_dirs_for_report,
            'leaked_files_count': len(leaked_files),
            'leaked_files_basenames': leaked_files
        }
    }

    report_path = output_dir / 'class_distribution_test.json'
    try:
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f'Analysis report saved to {report_path}')
    except IOError as e:
        print(f"Error: Could not write report to {report_path}: {e}")
        return

    if total_images > 0:
        try:
            plt.figure(figsize=(max(10, len(classes) * 0.8), 7))
            bars = plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
            plt.title(f'Test Set: Class Distribution (Total: {total_images} images)')
            plt.xlabel('Class')
            plt.ylabel('Number of Images')
            plt.xticks(rotation=45, ha='right')
            for bar_item in bars: # Renamed bar to bar_item to avoid conflict if plt.bar is used later
                height = bar_item.get_height()
                plt.text(bar_item.get_x() + bar_item.get_width() / 2., height,
                         f'{int(height)}', ha='center', va='bottom', fontsize=9)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            bar_chart_path = output_dir / 'class_distribution_test_bar.png'
            plt.savefig(bar_chart_path)
            plt.close()
            print(f"Bar chart saved to {bar_chart_path}")

            pie_data = {k: v for k, v in class_counts.items() if v > 0}
            if pie_data:
                plt.figure(figsize=(8, 8))
                plt.pie(pie_data.values(), labels=pie_data.keys(), autopct='%1.1f%%', startangle=140,
                        wedgeprops={'edgecolor': 'white'})
                plt.title(f'Test Set: Class Distribution (%) (Total: {total_images} images)')
                plt.axis('equal')
                plt.tight_layout()
                pie_chart_path = output_dir / 'class_distribution_test_pie.png'
                plt.savefig(pie_chart_path)
                plt.close()
                print(f"Pie chart saved to {pie_chart_path}")
            else:
                print("Skipping pie chart: No classes with images > 0 found.")
        except Exception as e:
            print(f"Error generating charts: {e}")
    else:
        print("Skipping chart generation: No images found in the test set.")

    if leaked_files:
        print(f"Warning: {len(leaked_files)} unique test image basenames found in potential training/source directories.")
        print(f"Leaked basenames: {leaked_files if len(leaked_files) < 10 else str(leaked_files[:10]) + '...'}")
        print("Please check the 'leakage_check.leaked_files_basenames' section in the JSON report for the full list.")
    else:
        print("No leakage detected between test set and checked training/source directories.")

if __name__ == '__main__':
    analyze_test_distribution()
