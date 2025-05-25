#!/usr/bin/env python3
"""
Enhanced Dataset Balancing Script - Pizza Dataset

This script balances the pizza dataset by generating multiple augmented samples
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
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

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

# Enhanced direct augmentation functions
def apply_augmentation(img):
    """Apply a random set of augmentations to the image"""
    # List of possible augmentations
    augmentations = [
        lambda x: x.rotate(random.randint(-30, 30), expand=True),
        lambda x: ImageOps.mirror(x),
        lambda x: ImageOps.flip(x),
        lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.7, 1.3)),
        lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.7, 1.3)),
        lambda x: ImageEnhance.Color(x).enhance(random.uniform(0.7, 1.3)),
        lambda x: x.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 2))),
        lambda x: ImageOps.posterize(x, random.randint(4, 7)),
        lambda x: ImageOps.solarize(x, random.randint(100, 200)),
    ]
    
    # Apply 3-5 random augmentations
    num_augs = random.randint(3, 5)
    selected_augs = random.sample(augmentations, num_augs)
    
    result = img.copy()
    for aug_func in selected_augs:
        result = aug_func(result)
    
    # Ensure consistent size (resize to the original size)
    original_size = img.size
    result = result.resize(original_size, Image.LANCZOS)
    
    return result

def generate_augmented_samples_direct(class_name, source_dir, output_dir, num_samples):
    """
    Generate augmented samples for a specific class using direct PIL transformations.
    
    Args:
        class_name: Name of the class
        source_dir: Directory containing original images for this class
        output_dir: Directory to save augmented images
        num_samples: Number of samples to generate
    """
    print(f"\nGenerating {num_samples} augmented samples for class: {class_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of source image files
    source_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        source_files.extend(list(Path(source_dir).glob(ext)))
    
    if not source_files:
        # If no source files in the class directory, check the legacy directory
        legacy_dir = Path("/home/emilio/Documents/ai/pizza/augmented_pizza_legacy")
        if legacy_dir.exists():
            # Look for files matching the class name
            for file_path in legacy_dir.glob(f"*_{class_name}_*.jpg"):
                source_files.append(file_path)
    
    if not source_files:
        print(f"  Error: No source images found for class {class_name}")
        return False
    
    print(f"  Found {len(source_files)} source images")
    
    # Generate new samples
    generated_count = 0
    target_count = num_samples
    
    while generated_count < target_count:
        # Randomly select a source image
        source_file = random.choice(source_files)
        
        try:
            # Open the image
            img = Image.open(source_file).convert('RGB')
            
            # Apply augmentations
            augmented_img = apply_augmentation(img)
            
            # Save the augmented image
            output_path = os.path.join(output_dir, f"{class_name}_augmented_{generated_count:04d}.jpg")
            augmented_img.save(output_path, quality=90)
            
            generated_count += 1
            
            # Print progress
            if generated_count % 5 == 0 or generated_count == target_count:
                print(f"  Generated {generated_count}/{target_count} images")
                
        except Exception as e:
            print(f"  Error processing {source_file}: {e}")
    
    print(f"  Successfully generated {generated_count} augmented images for {class_name}")
    return True

def balance_dataset_direct(distribution, threshold=0.8):
    """
    Balance the dataset using direct augmentation approach.
    
    Args:
        distribution: Class distribution dictionary
        threshold: Minimum sample percentage relative to largest class
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
        # Create backup of original class directory if it doesn't exist
        class_dir = augmented_dir / class_name
        backup_dir = augmented_dir / f"{class_name}_backup"
        
        if class_dir.exists() and not backup_dir.exists():
            print(f"\nBacking up original {class_name} directory")
            # Copy directory (don't move - we still need original for augmentation)
            shutil.copytree(class_dir, backup_dir)
        
        # Create temporary output directory
        temp_output_dir = project_root / "temp_balanced" / class_name
        os.makedirs(temp_output_dir, exist_ok=True)
        
        # Generate samples directly
        samples_needed = info["samples_needed"]
        success = generate_augmented_samples_direct(
            class_name,
            class_dir,
            temp_output_dir,
            samples_needed
        )
        
        if not success:
            continue
        
        # Copy all generated files to the class directory
        generated_files = list(temp_output_dir.glob("*.*"))
        print(f"  Copying {len(generated_files)} files to {class_name} directory")
        
        for file in generated_files:
            target_file = class_dir / file.name
            shutil.copy2(file, target_file)
    
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
    return parser.parse_args()

def main():
    """Main function to balance the dataset."""
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Load class distribution
    print("Loading class distribution...")
    distribution = load_class_distribution()
    
    # Balance the dataset using direct approach
    success = balance_dataset_direct(
        distribution, 
        threshold=args.threshold
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
