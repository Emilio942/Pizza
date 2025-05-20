#!/usr/bin/env python3
"""
Balance Dataset Script - Pizza Dataset

This script balances the pizza dataset by generating additional augmented samples
for underrepresented classes to ensure that each class has at least 80% of the
samples compared to the largest class.
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
import shutil
import sys

def load_class_distribution():
    """Load the class distribution report."""
    project_root = Path(__file__).parent
    report_path = project_root / "output" / "data_analysis" / "class_distribution_train.json"
    
    if not report_path.exists():
        print("Class distribution report not found. Running analysis...")
        subprocess.run(["python", "analyze_class_distribution.py"], check=True)
    
    with open(report_path, 'r') as f:
        distribution = json.load(f)
    
    return distribution

def identify_underrepresented_classes(distribution, threshold=0.8):
    """
    Identify classes that have fewer samples than the threshold percentage of the largest class.
    
    Args:
        distribution: Class distribution dictionary
        threshold: Minimum sample percentage relative to largest class (default: 0.8 for 80%)
        
    Returns:
        Dictionary with classes and the number of samples needed to reach the threshold
    """
    total_counts = distribution["total"]
    
    # Find the class with the most samples
    largest_class = max(total_counts.items(), key=lambda x: x[1])
    largest_class_name, largest_count = largest_class
    
    # Calculate the target minimum number of samples per class
    target_min_samples = int(largest_count * threshold)
    
    print(f"Largest class: {largest_class_name} with {largest_count} samples")
    print(f"Target minimum samples per class ({threshold*100}%): {target_min_samples}")
    
    # Identify underrepresented classes and calculate samples needed
    underrepresented = {}
    for class_name, count in total_counts.items():
        if count < target_min_samples:
            samples_needed = target_min_samples - count
            underrepresented[class_name] = {
                "current_samples": count,
                "target_samples": target_min_samples,
                "samples_needed": samples_needed,
                "percentage_of_largest": round((count / largest_count) * 100, 2)
            }
    
    return underrepresented, largest_class_name, largest_count

def generate_augmented_samples(class_name, source_dir, output_dir, num_samples, target_size=224):
    """
    Generate augmented samples for a specific class.
    
    Args:
        class_name: Name of the class
        source_dir: Directory containing original images for this class
        output_dir: Directory to save augmented images
        num_samples: Number of samples to generate
        target_size: Target image size
    """
    print(f"\nGenerating {num_samples} augmented samples for class: {class_name}")
    
    # Calculate augmentation parameters
    source_img_count = len(list(Path(source_dir).glob("*.*")))
    if source_img_count == 0:
        print(f"  Error: No source images found in {source_dir}")
        return False
        
    # Calculate how many augmented images per original image
    samples_per_image = max(5, num_samples // source_img_count + 1)
    
    cmd = [
        "python", "scripts/augment_dataset.py",
        "--input-dir", str(source_dir),
        "--output-dir", str(output_dir),
        "--num-per-image", str(samples_per_image),
        "--aug-types", "basic",  # Use only basic augmentation
        "--target-size", str(target_size),
        "--save-stats"
    ]
    
    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  Augmentation successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error during augmentation: {e}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False

def balance_dataset(distribution, threshold=0.8, target_size=224):
    """
    Balance the dataset by generating additional augmented samples.
    
    Args:
        distribution: Class distribution dictionary
        threshold: Minimum sample percentage relative to largest class
        target_size: Target image size
    """
    # Identify underrepresented classes
    underrepresented_classes, largest_class, largest_count = identify_underrepresented_classes(
        distribution, threshold=threshold
    )
    
    if not underrepresented_classes:
        print("\nNo underrepresented classes found! All classes meet the threshold.")
        return True
    
    # Print underrepresented classes
    print("\nUnderrepresented classes:")
    for class_name, info in underrepresented_classes.items():
        print(f"  {class_name}:")
        for key, value in info.items():
            print(f"    {key}: {value}")
    
    # Setup directories
    project_root = Path(__file__).parent
    augmented_dir = project_root / "augmented_pizza"
    
    # Process each underrepresented class
    for class_name, info in underrepresented_classes.items():
        # Create backup of original class directory
        class_dir = augmented_dir / class_name
        backup_dir = augmented_dir / f"{class_name}_backup"
        
        if class_dir.exists() and not backup_dir.exists():
            print(f"\nBacking up original {class_name} directory")
            # Copy directory (don't move - we still need original for augmentation)
            shutil.copytree(class_dir, backup_dir)
        
        # Generate augmented samples for this class
        samples_needed = info["samples_needed"]
        
        # Create temporary output directory
        temp_output_dir = project_root / "temp_balanced" / class_name
        os.makedirs(temp_output_dir, exist_ok=True)
        
        # Generate augmented samples
        success = generate_augmented_samples(
            class_name, 
            class_dir, 
            temp_output_dir, 
            samples_needed,
            target_size
        )
        
        if not success:
            continue
        
        # Count generated files
        generated_files = list(temp_output_dir.glob("*.*"))
        print(f"  Generated {len(generated_files)} files for {class_name}")
        
        # Copy augmented files to original class directory to update distribution
        copied_count = 0
        for file in generated_files:
            if copied_count >= samples_needed:
                break
                
            # Create unique target name to avoid overwriting
            target_file = class_dir / f"augmented_{class_name}_{copied_count}{file.suffix}"
            shutil.copy2(file, target_file)
            copied_count += 1
        
        print(f"  Added {copied_count} augmented samples to {class_name} directory")
    
    # Clean up temporary directory
    temp_dir = project_root / "temp_balanced"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    return True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Balance the pizza dataset by generating augmented samples for underrepresented classes.")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Minimum percentage of samples compared to largest class (default: 0.8 for 80%%)")
    parser.add_argument("--target-size", type=int, default=224,
                        help="Target image size in pixels")
    return parser.parse_args()

def main():
    """Main function to balance the dataset."""
    args = parse_args()
    
    # Load class distribution
    print("Loading class distribution...")
    distribution = load_class_distribution()
    
    # Balance the dataset
    success = balance_dataset(
        distribution, 
        threshold=args.threshold,
        target_size=args.target_size
    )
    
    if success:
        # Re-analyze the class distribution
        print("\nRe-analyzing class distribution after balancing...")
        subprocess.run(["python", "analyze_class_distribution.py"], check=True)
        
        # Load updated distribution
        updated_distribution = load_class_distribution()
        
        # Report results
        total_counts = updated_distribution["total"]
        largest_class = max(total_counts.items(), key=lambda x: x[1])
        largest_class_name, largest_count = largest_class
        
        print("\nBalancing complete! Updated class distribution:")
        for class_name, count in total_counts.items():
            percentage = count / largest_count * 100
            print(f"  {class_name}: {count} samples ({percentage:.2f}% of largest class)")
        
        # Check if all classes meet the threshold
        all_balanced = True
        for class_name, count in total_counts.items():
            if count < args.threshold * largest_count:
                all_balanced = False
                print(f"  Warning: {class_name} still below threshold with {count} samples ({count/largest_count*100:.2f}%)")
        
        if all_balanced:
            print(f"\nSuccess! All classes meet the {args.threshold*100}% threshold.")
        else:
            print(f"\nPartial success. Some classes still don't meet the {args.threshold*100}% threshold.")
    else:
        print("\nFailed to balance the dataset.")

if __name__ == "__main__":
    main()
