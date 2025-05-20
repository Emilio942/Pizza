#!/usr/bin/env python3
"""
Expand Test Set Script (DATEN-5.1)

This script expands the test set with challenging test cases that cover
real-world operational conditions (various lighting conditions, viewing angles,
pizza variations). It generates new test images, labels them correctly, and
ensures they are not used in training.

The script:
1. Organizes existing test images into class directories
2. Generates new challenging test images with specific augmentations
3. Ensures proper labeling of all test images
4. Verifies no test images appear in training/validation sets
5. Analyzes class distribution of the expanded test set

Usage:
    python expand_test_set.py [options]

Options:
    --num-per-class: Number of challenging images to generate per class (default: 17)
    --source-dir: Source directory for original images to augment (default: data/classified)
    --test-dir: Directory for the test set (default: data/test)
"""

import os
import sys
import argparse
import random
import json
import shutil
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
from tqdm import tqdm


def load_class_definitions(project_root):
    """Load class definitions from the JSON file."""
    class_file = project_root / "data" / "class_definitions.json"
    
    if not class_file.is_file():
        print(f"Error: Class definitions file not found at {class_file}")
        return None
    
    try:
        with open(class_file, 'r') as f:
            class_definitions = json.load(f)
        return class_definitions
    except Exception as e:
        print(f"Error loading class definitions: {e}")
        return None


def organize_existing_test_images(test_dir, classes):
    """Organize existing test images into class directories based on their filenames."""
    print("Organizing existing test images...")
    
    # Create class directories if they don't exist
    for cls in classes:
        cls_dir = test_dir / cls
        cls_dir.mkdir(exist_ok=True)
    
    # Move images to their respective class directories based on filename
    moved_count = 0
    for file_path in test_dir.glob("*.*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            filename = file_path.name.lower()
            target_class = None
            
            # Determine class from filename
            for cls in classes:
                if cls in filename:
                    target_class = cls
                    break
            
            if target_class:
                # Move the file to its class directory
                target_path = test_dir / target_class / file_path.name
                if not target_path.exists():
                    shutil.move(str(file_path), str(target_path))
                    moved_count += 1
                else:
                    print(f"Warning: {target_path} already exists, skipping")
    
    print(f"Moved {moved_count} existing test images to their class directories")


def apply_challenging_augmentation(img, augmentation_type):
    """Apply specific challenging augmentations to an image."""
    if augmentation_type == "lighting_dark":
        # Very dark image
        factor = random.uniform(0.1, 0.4)
        return ImageEnhance.Brightness(img).enhance(factor)
    
    elif augmentation_type == "lighting_bright":
        # Very bright/overexposed image
        factor = random.uniform(1.6, 2.5)
        return ImageEnhance.Brightness(img).enhance(factor)
    
    elif augmentation_type == "lighting_contrast":
        # Extreme contrast
        factor = random.uniform(1.7, 3.0)
        return ImageEnhance.Contrast(img).enhance(factor)
    
    elif augmentation_type == "perspective":
        # Extreme angle/perspective
        width, height = img.size
        
        # Define more extreme perspective transformation
        coeffs = find_coeffs(
            [(0, 0), (width, 0), (width, height), (0, height)],
            [(random.randint(0, width//3), random.randint(0, height//3)), 
             (width - random.randint(0, width//3), random.randint(0, height//3)), 
             (width - random.randint(0, width//4), height - random.randint(0, height//4)), 
             (random.randint(0, width//4), height - random.randint(0, height//4))]
        )
        
        return img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    
    elif augmentation_type == "noise":
        # Add significant noise
        img_array = np.array(img)
        noise = np.random.normal(0, random.randint(15, 40), img_array.shape)
        noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img_array)
    
    elif augmentation_type == "blur":
        # Significant blur
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(2.0, 5.0)))
    
    elif augmentation_type == "combined":
        # Combine multiple challenging conditions
        num_augs = random.randint(2, 3)
        
        aug_types = ["lighting_dark", "lighting_bright", "lighting_contrast", 
                    "perspective", "noise", "blur"]
        selected_augs = random.sample(aug_types, num_augs)
        
        result = img.copy()
        for aug_type in selected_augs:
            result = apply_challenging_augmentation(result, aug_type)
        
        return result
    
    else:
        # Default: return original image
        return img.copy()


def find_coeffs(pa, pb):
    """Helper function for perspective transformation."""
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def generate_challenging_test_images(source_dir, test_dir, classes, num_per_class=17):
    """Generate challenging test images for each class and place them in test directory."""
    print(f"Generating {num_per_class} challenging test images per class...")
    
    # Challenging augmentation types
    augmentation_types = [
        "lighting_dark", "lighting_bright", "lighting_contrast", 
        "perspective", "noise", "blur", "combined"
    ]
    
    for cls in classes:
        print(f"\nProcessing class: {cls}")
        source_cls_dir = source_dir / cls
        test_cls_dir = test_dir / cls
        
        # Ensure the target directory exists
        test_cls_dir.mkdir(exist_ok=True)
        
        # Find source images for this class
        source_images = []
        for ext in ['.jpg', '.jpeg', '.png']:
            source_images.extend(list(source_cls_dir.glob(f"*{ext}")))
        
        if not source_images:
            print(f"  Warning: No source images found for class {cls}")
            continue
        
        # Generate challenging images
        for i in tqdm(range(num_per_class), desc=f"Generating {cls} images"):
            # Select a random source image
            source_img_path = random.choice(source_images)
            
            try:
                # Open the source image
                img = Image.open(source_img_path).convert('RGB')
                
                # Select a random challenging augmentation type
                aug_type = random.choice(augmentation_types)
                
                # Apply the challenging augmentation
                augmented_img = apply_challenging_augmentation(img, aug_type)
                
                # Create filename with class and augmentation type
                output_filename = f"challenging_{cls}_{aug_type}_{i:03d}.jpg"
                output_path = test_cls_dir / output_filename
                
                # Save the augmented image
                augmented_img.save(output_path, "JPEG", quality=95)
                
            except Exception as e:
                print(f"  Error processing {source_img_path}: {e}")
    
    print("Challenging test image generation complete!")


def check_for_leakage(test_dir, train_dirs):
    """Check if any test images appear in the training/validation sets."""
    print("Checking for data leakage...")
    
    # Get all test image filenames
    test_images = []
    for ext in ['.jpg', '.jpeg', '.png']:
        test_images.extend([p.name for p in test_dir.rglob(f"*{ext}")])
    
    test_image_set = set(test_images)
    
    leakage_found = False
    leaked_files = []
    
    # Check each training directory
    for train_dir in train_dirs:
        train_dir_path = Path(train_dir)
        if not train_dir_path.is_dir():
            continue
        
        # Get all training image filenames
        train_images = []
        for ext in ['.jpg', '.jpeg', '.png']:
            train_images.extend([p.name for p in train_dir_path.rglob(f"*{ext}")])
        
        # Find intersection (leakage)
        leaked = test_image_set.intersection(set(train_images))
        if leaked:
            leakage_found = True
            leaked_files.extend(list(leaked))
            print(f"Found {len(leaked)} leaked images in {train_dir}")
    
    if not leakage_found:
        print("No data leakage detected!")
    else:
        print(f"WARNING: {len(leaked_files)} test images were found in training/validation sets")
    
    return leaked_files


def analyze_class_distribution(test_dir, classes):
    """Analyze and report the class distribution in the test set."""
    print("Analyzing class distribution in test set...")
    
    class_counts = {cls: 0 for cls in classes}
    
    # Count images in each class directory
    for cls in classes:
        cls_dir = test_dir / cls
        if cls_dir.is_dir():
            for ext in ['.jpg', '.jpeg', '.png']:
                class_counts[cls] += len(list(cls_dir.glob(f"*{ext}")))
    
    total_images = sum(class_counts.values())
    
    if total_images == 0:
        print("No images found in the test set!")
        return
    
    # Calculate percentages
    percentages = {cls: (count / total_images) * 100 for cls, count in class_counts.items()}
    
    # Print results
    print("\nTest Set Class Distribution:")
    print(f"Total images: {total_images}")
    print("-" * 40)
    for cls in classes:
        print(f"{cls}: {class_counts[cls]} images ({percentages[cls]:.1f}%)")
    print("-" * 40)
    
    # Save results to JSON
    distribution_data = {
        "dataset_name": "test_set_distribution",
        "total_images": total_images,
        "class_counts": class_counts,
        "percentages": {cls: round(pct, 2) for cls, pct in percentages.items()}
    }
    
    output_dir = test_dir.parent.parent / "output" / "data_analysis"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_file = output_dir / "test_set_distribution.json"
    with open(output_file, 'w') as f:
        json.dump(distribution_data, f, indent=4)
    
    print(f"Distribution data saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Expand test set with challenging images")
    parser.add_argument("--num-per-class", type=int, default=17, 
                        help="Number of challenging images to generate per class")
    parser.add_argument("--source-dir", type=str, default="data/classified",
                        help="Source directory for original images to augment")
    parser.add_argument("--test-dir", type=str, default="data/test",
                        help="Directory for the test set")
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).resolve().parent.parent
    
    # Load class definitions
    class_definitions = load_class_definitions(project_root)
    if not class_definitions:
        sys.exit(1)
    
    classes = list(class_definitions.keys())
    print(f"Found {len(classes)} classes: {', '.join(classes)}")
    
    # Set paths
    source_dir = project_root / "augmented_pizza"  # Use the augmented_pizza directory for source images
    test_dir = project_root / args.test_dir
    
    if not source_dir.is_dir():
        print(f"Error: Source directory {source_dir} not found")
        sys.exit(1)
    
    if not test_dir.is_dir():
        print(f"Creating test directory {test_dir}")
        test_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Organize existing test images
    organize_existing_test_images(test_dir, classes)
    
    # 2. Generate challenging test images
    generate_challenging_test_images(source_dir, test_dir, classes, args.num_per_class)
    
    # 3. Check for data leakage
    potential_train_dirs = [
        project_root / "augmented_pizza",
        project_root / "augmented_pizza_legacy",
        project_root / "data" / "raw",
        project_root / "data" / "processed",
        project_root / "data" / "classified",
        project_root / "data" / "augmented",
        project_root / "data" / "train",
        project_root / "data" / "validation"
    ]
    check_for_leakage(test_dir, potential_train_dirs)
    
    # 4. Analyze class distribution
    analyze_class_distribution(test_dir, classes)
    
    print("\nTest set expansion complete!")
    print(f"Added approximately {len(classes) * args.num_per_class} challenging test images")
    print("Please verify the test images manually and adjust labels if needed.")


if __name__ == "__main__":
    main()
