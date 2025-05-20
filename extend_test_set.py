#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to analyze the test dataset and generate challenging test scenarios
"""

import os
import json
import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import re

# Project directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEST_DIR = os.path.join(DATA_DIR, "test")
TEST_EXT_DIR = os.path.join(DATA_DIR, "test_extension")
CLASS_DEF_FILE = os.path.join(DATA_DIR, "class_definitions.json")
AUGMENTED_DIR = os.path.join(DATA_DIR, "augmented_pizza")
RAW_DIR = os.path.join(DATA_DIR, "raw")

# Define lighting conditions
LIGHTING_CONDITIONS = [
    "normal",           # Normal lighting
    "dark",             # Dark conditions
    "bright",           # Overexposed
    "uneven",           # Uneven lighting
    "low_contrast"      # Low contrast
]

# Load class definitions
with open(CLASS_DEF_FILE, 'r') as f:
    CLASS_DEFINITIONS = json.load(f)

CLASS_NAMES = list(CLASS_DEFINITIONS.keys())

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
        return enhancer.enhance(0.5)
    elif condition == "bright":
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(1.5)
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
        return enhancer.enhance(0.6)
    else:
        return img

def create_extension_set(class_counts, challenging_conditions):
    """Create an extended test set with challenging conditions."""
    # Make sure the extension directory exists with subdirectories for each lighting condition
    os.makedirs(TEST_EXT_DIR, exist_ok=True)
    for condition in LIGHTING_CONDITIONS:
        condition_dir = os.path.join(TEST_EXT_DIR, condition)
        os.makedirs(condition_dir, exist_ok=True)
        
        # Create class subdirectories
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(condition_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
    
    # Determine how many images to generate per class and condition
    target_per_class_condition = 20  # Target number of images per class per condition
    
    # For each class
    for class_name in CLASS_NAMES:
        source_dir = os.path.join(TEST_DIR, class_name)
        if not os.path.exists(source_dir):
            print(f"Warning: Source directory not found: {source_dir}")
            continue
            
        # Get all source images for this class
        source_images = []
        for filename in os.listdir(source_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                source_images.append(os.path.join(source_dir, filename))
        
        if not source_images:
            print(f"Warning: No source images found for class {class_name}")
            continue
            
        # If we don't have enough images, look in other directories
        if len(source_images) < 5:
            # Try the augmented directory
            aug_class_dir = os.path.join(AUGMENTED_DIR, class_name)
            if os.path.exists(aug_class_dir):
                for filename in os.listdir(aug_class_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        source_images.append(os.path.join(aug_class_dir, filename))
            
            # Try the raw directory
            raw_class_dir = os.path.join(RAW_DIR, class_name)
            if os.path.exists(raw_class_dir):
                for filename in os.listdir(raw_class_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        source_images.append(os.path.join(raw_class_dir, filename))
        
        print(f"Found {len(source_images)} source images for class {class_name}")
        
        # For each lighting condition
        for condition in LIGHTING_CONDITIONS:
            target_dir = os.path.join(TEST_EXT_DIR, condition, class_name)
            
            # Check how many images we already have for this condition
            existing_images = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            num_to_generate = max(0, target_per_class_condition - len(existing_images))
            
            print(f"Generating {num_to_generate} images for {class_name} under {condition} conditions")
            
            # Generate the required number of images
            for i in range(num_to_generate):
                # Randomly select a source image
                source_path = random.choice(source_images)
                try:
                    # Open the image and apply the lighting condition
                    img = Image.open(source_path)
                    img = apply_lighting_condition(img, condition)
                    
                    # Save the modified image
                    filename = f"test_{class_name}_{condition}_{i:03d}.jpg"
                    target_path = os.path.join(target_dir, filename)
                    img.save(target_path)
                    print(f"Generated {target_path}")
                except Exception as e:
                    print(f"Error processing {source_path}: {e}")
    
    # Count the newly generated images
    new_images_count = 0
    for condition in LIGHTING_CONDITIONS:
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(TEST_EXT_DIR, condition, class_name)
            if os.path.exists(class_dir):
                new_images_count += len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"\nGenerated a total of {new_images_count} new test images")
    
    return new_images_count

if __name__ == "__main__":
    print("Analyzing existing test set...")
    class_counts, challenging_conditions, total_images = analyze_test_set()
    
    # Save analysis to JSON
    analysis = {
        "class_counts": dict(class_counts),
        "challenging_conditions": {k: dict(v) for k, v in challenging_conditions.items()},
        "total_images": total_images
    }
    
    os.makedirs(os.path.join(PROJECT_ROOT, "output"), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, "output", "test_set_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)
    
    print("\nCreating extended test set with challenging conditions...")
    new_images_count = create_extension_set(class_counts, challenging_conditions)
    
    print("\nTest set extension complete!")
    print(f"- Original test set: {total_images} images")
    print(f"- New extended test set: {new_images_count} images")
    print(f"- Extended test set location: {TEST_EXT_DIR}")
    print(f"- Analysis saved to: {os.path.join(PROJECT_ROOT, 'output', 'test_set_analysis.json')}")
