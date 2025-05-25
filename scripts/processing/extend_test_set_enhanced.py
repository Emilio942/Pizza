#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to analyze the test dataset and generate challenging test scenarios

This script performs the following tasks:
1. Analyzes the current test set distribution across classes
2. Identifies areas where the model might have weaknesses 
3. Generates challenging test images with various conditions
4. Verifies test images are not duplicated in training/validation sets
5. Analyzes the class distribution of the extended test set
"""

import os
import json
import random
import shutil
import argparse
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import re
from datetime import datetime

# Project directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEST_DIR = os.path.join(DATA_DIR, "test")
TEST_EXT_DIR = os.path.join(DATA_DIR, "test_extension")
TEST_NEW_DIR = os.path.join(DATA_DIR, "test_new_images")
CLASS_DEF_FILE = os.path.join(DATA_DIR, "class_definitions.json")
AUGMENTED_DIR = os.path.join(PROJECT_ROOT, "augmented_pizza")
RAW_DIR = os.path.join(DATA_DIR, "raw")
TRAIN_DIR = os.path.join(DATA_DIR, "processed", "train")
VALIDATION_DIR = os.path.join(DATA_DIR, "processed", "validation")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "data_analysis")

# Define lighting conditions
LIGHTING_CONDITIONS = [
    "normal",           # Normal lighting
    "dark",             # Dark conditions
    "bright",           # Overexposed
    "uneven",           # Uneven lighting
    "low_contrast",     # Low contrast
    "shadowed",         # Partial shadows
    "backlit"           # Backlit (silhouette effect)
]

# Define perspective variations
PERSPECTIVE_VARIATIONS = [
    "normal",           # Normal perspective
    "angle_side",       # Side view 
    "angle_top",        # Top-down view
    "angle_close",      # Close-up view
    "angle_diagonal"    # Diagonal view
]

# Define texture and quality variations
QUALITY_VARIATIONS = [
    "normal",           # Normal quality
    "noisy",            # Add noise
    "blurry",           # Add blur
    "jpeg_artifact",    # Add JPEG artifacts
    "motion_blur"       # Add motion blur
]

# Load class definitions
with open(CLASS_DEF_FILE, 'r') as f:
    CLASS_DEFINITIONS = json.load(f)

CLASS_NAMES = list(CLASS_DEFINITIONS.keys())

def compute_image_hash(image_path):
    """
    Compute a hash for an image to detect duplicates
    Uses average hash which is less sensitive to small changes
    """
    try:
        img = Image.open(image_path).convert('L').resize((16, 16), Image.LANCZOS)
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        hash_value = ''.join('1' if pixel > avg else '0' for pixel in pixels)
        return hash_value
    except Exception as e:
        print(f"Error computing hash for {image_path}: {e}")
        return None

def analyze_test_set():
    """Analyze the distribution of images in the test set."""
    class_counts = Counter()
    challenging_conditions = defaultdict(Counter)
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(TEST_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory not found: {class_dir}")
            continue
            
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                class_counts[class_name] += 1
                
                # Categorize challenging conditions
                if 'challenging' in filename:
                    if 'lighting_dark' in filename:
                        challenging_conditions['dark'][class_name] += 1
                    elif 'lighting_bright' in filename:
                        challenging_conditions['bright'][class_name] += 1
                    elif 'lighting_contrast' in filename:
                        challenging_conditions['low_contrast'][class_name] += 1
                    elif 'perspective' in filename:
                        challenging_conditions['perspective'][class_name] += 1
                    elif 'noise' in filename:
                        challenging_conditions['noise'][class_name] += 1
                    elif 'blur' in filename:
                        challenging_conditions['blur'][class_name] += 1
                    elif 'combined' in filename:
                        challenging_conditions['combined'][class_name] += 1
                    else:
                        challenging_conditions['other'][class_name] += 1
                else:
                    challenging_conditions['normal'][class_name] += 1
    
    # Print summary
    print(f"{'Class':<20} | {'Count':<10} | {'Percentage':<10}")
    print("-" * 45)
    
    total_images = sum(class_counts.values())
    for class_name, count in sorted(class_counts.items()):
        percentage = (count / total_images * 100) if total_images > 0 else 0
        print(f"{class_name:<20} | {count:<10} | {percentage:<10.2f}%")
        
    print("\nTotal test images:", total_images)
    
    # Print challenging conditions summary
    print("\nChallenging Conditions Breakdown:")
    for condition, counts in challenging_conditions.items():
        if sum(counts.values()) > 0:
            print(f"\n{condition.capitalize()}:")
            for class_name, count in sorted(counts.items()):
                print(f"  {class_name}: {count}")
    
    return class_counts, challenging_conditions, total_images

def apply_lighting_condition(img, condition):
    """Apply a specific lighting condition to an image."""
    if condition == "normal":
        return img
    elif condition == "dark":
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(0.4)
    elif condition == "bright":
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(1.6)
    elif condition == "uneven":
        # Create a gradient overlay
        width, height = img.size
        gradient = Image.new('L', (width, height))
        for y in range(height):
            for x in range(width):
                gradient.putpixel((x, y), int(255 * (0.5 + 0.5 * (x / width))))
        
        # Apply the gradient
        r, g, b = img.split()
        r = r.point(lambda i: i * 0.6)
        output = Image.merge('RGB', (r, g, b))
        return output
    elif condition == "low_contrast":
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(0.5)
    elif condition == "shadowed":
        # Add directional shadow
        width, height = img.size
        shadow = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(shadow)
        # Draw a gradient shadow in one corner
        for i in range(width // 3):
            opacity = 100 + i
            draw.rectangle((0, 0, width // 3 - i, height // 3 - i), fill=opacity)
        
        # Apply the shadow
        img = img.copy()
        img.putalpha(shadow)
        background = Image.new('RGB', img.size, (30, 30, 30))
        background.paste(img, (0, 0), img)
        return background.convert('RGB')
    elif condition == "backlit":
        # Simulate backlit effect
        width, height = img.size
        backlight = Image.new('RGB', (width, height), (240, 240, 240))
        img = ImageEnhance.Brightness(img).enhance(0.7)
        img = ImageEnhance.Contrast(img).enhance(1.2)
        
        # Create a mask for the pizza (simple threshold-based)
        mask = img.convert('L')
        mask = mask.point(lambda x: 0 if x < 100 else 255)
        
        # Blend the images
        result = Image.composite(img, backlight, mask)
        return result
    else:
        return img
        
def apply_perspective_variation(img, variation):
    """Apply a perspective variation to an image"""
    width, height = img.size
    if variation == "normal":
        return img
    elif variation == "angle_side":
        # Side angle perspective transform
        coeffs = (1.2, 0.1, -0.1 * width,
                 0.05, 1.0, 0,
                 0.0005, 0, 1)
        return img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    elif variation == "angle_top":
        # Top-down view
        coeffs = (1.0, 0, 0,
                 0.2, 0.8, 0.1 * height,
                 0, 0.001, 1)
        return img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    elif variation == "angle_close":
        # Close-up effect (zoom on center)
        crop_width = int(width * 0.7)
        crop_height = int(height * 0.7)
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        
        return img.crop((left, top, right, bottom)).resize((width, height), Image.LANCZOS)
    elif variation == "angle_diagonal":
        # Diagonal angle 
        coeffs = (1.0, 0.15, -20,
                 0.15, 1.0, -20,
                 0.0005, 0.0005, 1)
        return img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    else:
        return img
        
def apply_quality_variation(img, variation):
    """Apply quality/texture variations to an image"""
    if variation == "normal":
        return img
    elif variation == "noisy":
        # Add noise
        img = img.copy()
        noise = np.random.normal(0, 25, img.size + (3,))
        img_array = np.array(img)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    elif variation == "blurry":
        # Add gaussian blur
        return img.filter(ImageFilter.GaussianBlur(radius=2.5))
    elif variation == "jpeg_artifact":
        # Add JPEG compression artifacts
        temp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_jpeg_artifact.jpg")
        img.save(temp_file, quality=15)
        result = Image.open(temp_file)
        try:
            os.remove(temp_file)
        except:
            pass
        return result
    elif variation == "motion_blur":
        # Add motion blur
        return img.filter(ImageFilter.GaussianBlur(radius=0))  # Placeholder - would need custom kernel for true motion blur
    else:
        return img

def verify_no_duplicates(source_path, target_path):
    """
    Verify that the source image is not a duplicate of the target image
    Uses perceptual hash to allow for small variations
    """
    source_hash = compute_image_hash(source_path)
    target_hash = compute_image_hash(target_path)
    
    if source_hash is None or target_hash is None:
        return False
    
    # Calculate Hamming distance between hashes (number of differing bits)
    hamming_distance = sum(c1 != c2 for c1, c2 in zip(source_hash, target_hash))
    
    # If distance is small, images are likely similar
    return hamming_distance > 20  # Threshold can be adjusted

def collect_existing_images(directory, extensions=('.jpg', '.jpeg', '.png')):
    """Collect all images in a directory (recursively)"""
    image_paths = []
    
    if not os.path.exists(directory):
        return image_paths
        
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(root, file))
                
    return image_paths

def check_for_leakage(test_images, train_val_images):
    """
    Check for potential data leakage between test and train/validation sets
    Returns a list of potentially duplicated images in the test set
    """
    potential_duplicates = []
    
    print(f"Checking {len(test_images)} test images against {len(train_val_images)} train/val images")
    
    # Compute hashes for train/val images (can be time-consuming)
    train_val_hashes = {}
    for img_path in train_val_images:
        train_val_hashes[img_path] = compute_image_hash(img_path)
    
    # Check each test image against train/val hashes
    for test_img in test_images:
        test_hash = compute_image_hash(test_img)
        if test_hash is None:
            continue
            
        for train_img, train_hash in train_val_hashes.items():
            if train_hash is None:
                continue
                
            # Calculate Hamming distance
            hamming_distance = sum(c1 != c2 for c1, c2 in zip(test_hash, train_hash))
            
            # If distance is small, images are potentially duplicated
            if hamming_distance < 10:  # Threshold can be adjusted
                potential_duplicates.append((test_img, train_img, hamming_distance))
                
    return potential_duplicates
    
def create_combined_variation(img, light_condition, perspective, quality):
    """
    Apply multiple variations to an image
    """
    img = apply_lighting_condition(img, light_condition)
    img = apply_perspective_variation(img, perspective)
    img = apply_quality_variation(img, quality)
    return img

def create_advanced_test_set(class_counts, challenging_conditions, target_count=100):
    """
    Create an advanced test set with challenging conditions combining
    lighting, perspective and quality variations
    """
    # Create output directory for new test images
    os.makedirs(TEST_NEW_DIR, exist_ok=True)
    
    # Create subdirectories for each class
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(TEST_NEW_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    # Determine images per class based on target count
    total_current = sum(class_counts.values())
    if total_current > 0:
        # Calculate how many more images we need per class
        images_per_class = {cls: max(5, int(target_count * (count / total_current))) 
                           for cls, count in class_counts.items()}
    else:
        # If no images yet, distribute evenly
        images_per_class = {cls: int(target_count / len(CLASS_NAMES)) for cls in CLASS_NAMES}
    
    # Get existing training and validation images to avoid duplicates
    train_val_images = collect_existing_images(TRAIN_DIR) + collect_existing_images(VALIDATION_DIR)
    
    # Generate challenging images for each class
    generated_count = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for class_name, target in images_per_class.items():
        print(f"\nGenerating challenging test images for class '{class_name}' (target: {target})")
        
        # Get source images from test, augmented_pizza, and raw directories
        source_images = []
        
        # First try test directory
        class_test_dir = os.path.join(TEST_DIR, class_name)
        if os.path.exists(class_test_dir):
            for file in os.listdir(class_test_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    source_images.append(os.path.join(class_test_dir, file))
        
        # If we need more, look in augmented directory
        if len(source_images) < 5:
            class_aug_dir = os.path.join(AUGMENTED_DIR, class_name)
            if os.path.exists(class_aug_dir):
                for file in os.listdir(class_aug_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        source_images.append(os.path.join(class_aug_dir, file))
        
        # If still need more, look in raw directory
        if len(source_images) < 5:
            class_raw_dir = os.path.join(RAW_DIR, class_name)
            if os.path.exists(class_raw_dir):
                for file in os.listdir(class_raw_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        source_images.append(os.path.join(class_raw_dir, file))
        
        if not source_images:
            print(f"Warning: No source images found for class '{class_name}'")
            continue
            
        print(f"Found {len(source_images)} source images for class '{class_name}'")
        
        # Generate the challenging images
        class_generated = 0
        max_attempts = target * 2  # Allow for some failed attempts
        attempts = 0
        
        while class_generated < target and attempts < max_attempts:
            attempts += 1
            
            # Randomly select a source image
            source_path = random.choice(source_images)
            
            try:
                # Randomly select challenging conditions
                light_condition = random.choice(LIGHTING_CONDITIONS)
                perspective = random.choice(PERSPECTIVE_VARIATIONS)
                quality = random.choice(QUALITY_VARIATIONS)
                
                # Make sure at least one condition is challenging
                while light_condition == "normal" and perspective == "normal" and quality == "normal":
                    light_condition = random.choice(LIGHTING_CONDITIONS)
                    perspective = random.choice(PERSPECTIVE_VARIATIONS)
                    quality = random.choice(QUALITY_VARIATIONS)
                
                # Load and transform the image
                try:
                    img = Image.open(source_path)
                    transformed_img = create_combined_variation(img, light_condition, perspective, quality)
                    
                    # Create a unique filename
                    conditions = f"{light_condition}_{perspective}_{quality}"
                    filename = f"test_{class_name}_challenging_{conditions}_{timestamp}_{class_generated:03d}.jpg"
                    target_path = os.path.join(TEST_NEW_DIR, class_name, filename)
                    
                    # Save the image
                    transformed_img.save(target_path)
                    
                    # Check if this image is too similar to training or validation images
                    duplicated = False
                    for train_val_img in random.sample(train_val_images, min(50, len(train_val_images))):
                        if not verify_no_duplicates(target_path, train_val_img):
                            duplicated = True
                            os.remove(target_path)
                            print(f"Removed potential duplicate: {filename}")
                            break
                    
                    if not duplicated:
                        print(f"Generated: {filename}")
                        class_generated += 1
                
                except Exception as e:
                    print(f"Error processing image {source_path}: {e}")
            
            except Exception as e:
                print(f"Error in generation process: {e}")
                
        print(f"Generated {class_generated} challenging images for class '{class_name}'")
        generated_count += class_generated
    
    print(f"\nTest set extension complete! Generated {generated_count} new challenging test images.")
    return generated_count

def copy_to_test_directory(source_dir=TEST_NEW_DIR, target_dir=TEST_DIR):
    """
    Copy the generated test images to the main test directory
    while maintaining class structure
    """
    copied_count = 0
    
    for class_name in CLASS_NAMES:
        source_class_dir = os.path.join(source_dir, class_name)
        target_class_dir = os.path.join(target_dir, class_name)
        
        if not os.path.exists(source_class_dir):
            continue
            
        # Ensure target directory exists
        os.makedirs(target_class_dir, exist_ok=True)
        
        # Copy all images
        for filename in os.listdir(source_class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                source_path = os.path.join(source_class_dir, filename)
                target_path = os.path.join(target_class_dir, filename)
                
                # Copy the file
                shutil.copy2(source_path, target_path)
                copied_count += 1
    
    return copied_count

def analyze_test_distribution():
    """
    Analyze and visualize the class distribution in the test set
    Similar to analyze_class_distribution.py but focused on test data
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Count images in each class
    class_counts = {}
    total_images = 0
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(TEST_DIR, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[class_name] = count
            total_images += count
    
    # Calculate percentages
    percentages = {}
    for class_name, count in class_counts.items():
        percentages[class_name] = round((count / total_images) * 100, 2) if total_images > 0 else 0
    
    # Create visualization
    # Bar chart
    plt.figure(figsize=(12, 8))
    
    class_names = list(class_counts.keys())
    counts = list(class_counts.values())
    
    bars = plt.bar(class_names, counts)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom')
    
    # Add percentage labels inside bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height/2,
                 f'{percentages[class_names[i]]}%', ha='center', va='center', 
                 color='white', fontweight='bold')
    
    plt.title('Test Dataset - Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save chart
    plt.savefig(os.path.join(OUTPUT_DIR, "test_class_distribution.png"))
    
    # Create pie chart
    plt.figure(figsize=(10, 10))
    plt.pie(counts, labels=class_names, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Test Dataset - Class Distribution (%)')
    plt.tight_layout()
    
    # Save pie chart
    plt.savefig(os.path.join(OUTPUT_DIR, "test_class_distribution_pie.png"))
    
    # Save data to JSON
    results = {
        "class_counts": class_counts,
        "percentages": percentages,
        "total": total_images
    }
    
    with open(os.path.join(OUTPUT_DIR, "test_class_distribution.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTest data distribution analysis complete.")
    print(f"Results saved to {OUTPUT_DIR}")
    
    return results

def main():
    """Main function to handle the task"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Extend the test set with challenging images.')
    parser.add_argument('--target', type=int, default=100, 
                        help='Target number of total new test images to generate')
    parser.add_argument('--integrate', action='store_true',
                        help='Copy the generated images to the main test directory')
    parser.add_argument('--check-leakage', action='store_true',
                        help='Check for potential data leakage between test and train/validation sets')
    args = parser.parse_args()
    
    # Step 1: Analyze existing test set
    print("Step 1: Analyzing existing test set...")
    class_counts, challenging_conditions, total_images = analyze_test_set()
    
    # Save analysis to JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    analysis = {
        "class_counts": dict(class_counts),
        "challenging_conditions": {k: dict(v) for k, v in challenging_conditions.items()},
        "total_images": total_images
    }
    
    with open(os.path.join(OUTPUT_DIR, "test_set_analysis_before.json"), "w") as f:
        json.dump(analysis, f, indent=2)
    
    # Step 2: Generate advanced test images
    print("\nStep 2: Generating advanced test images...")
    generated_count = create_advanced_test_set(class_counts, challenging_conditions, args.target)
    
    # Step 3: Check for data leakage if requested
    if args.check_leakage:
        print("\nStep 3: Checking for potential data leakage...")
        test_images = collect_existing_images(TEST_NEW_DIR)
        train_val_images = collect_existing_images(TRAIN_DIR) + collect_existing_images(VALIDATION_DIR)
        
        potential_duplicates = check_for_leakage(test_images, train_val_images)
        
        if potential_duplicates:
            print(f"\nFound {len(potential_duplicates)} potentially duplicated images:")
            for test_img, train_img, distance in potential_duplicates:
                print(f"Test: {os.path.basename(test_img)} - Train/Val: {os.path.basename(train_img)} - Distance: {distance}")
                
            # Save potential duplicates to a report
            with open(os.path.join(OUTPUT_DIR, "potential_duplicates.json"), "w") as f:
                json.dump([{
                    "test_image": test_img,
                    "train_val_image": train_img,
                    "distance": distance
                } for test_img, train_img, distance in potential_duplicates], f, indent=2)
        else:
            print("No potential duplicates found between test and train/validation sets.")
    
    # Step 4: Integrate generated images into the test set if requested
    if args.integrate:
        print("\nStep 4: Integrating new images into main test directory...")
        copied_count = copy_to_test_directory()
        print(f"Copied {copied_count} images to main test directory.")
    
    # Step 5: Analyze final test distribution
    print("\nStep 5: Analyzing final test distribution...")
    final_distribution = analyze_test_distribution()
    
    # Print summary
    print("\n=== Test Set Extension Summary ===")
    print(f"Original test set: {total_images} images")
    print(f"Generated: {generated_count} new challenging test images")
    
    if args.integrate:
        print(f"Final test set: {final_distribution['total']} images")
    
    print(f"\nAll results and visualizations saved to {OUTPUT_DIR}")
    return 0

if __name__ == "__main__":
    main()
