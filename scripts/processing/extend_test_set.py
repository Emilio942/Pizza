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
