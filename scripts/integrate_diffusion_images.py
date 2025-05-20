#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diffusion Model Integration Script for Pizza Dataset

This script integrates diffusion model-generated images into the existing
dataset structure, with a focus on underrepresented classes and hard-to-capture
scenarios like specific burn patterns and lighting conditions.

Usage:
    python scripts/integrate_diffusion_images.py --analyze-dataset
    python scripts/integrate_diffusion_images.py --generate --preset small_diverse --model sd-food --image_size 512
    python scripts/integrate_diffusion_images.py --organize-images
    python scripts/integrate_diffusion_images.py --all
"""

import os
import sys
import argparse
import logging
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import subprocess

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('diffusion_integration.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
SYNTHETIC_DIR = Path("data/synthetic")
DATA_DIR = Path("data")
PIZZA_CLASSES = None  # Will load from data/class_definitions.json

def load_class_definitions():
    """Load class definitions from the central JSON file"""
    global PIZZA_CLASSES
    
    try:
        with open(DATA_DIR / "class_definitions.json", "r") as f:
            PIZZA_CLASSES = json.load(f)
        logger.info(f"Loaded {len(PIZZA_CLASSES)} pizza classes from class definitions")
        return True
    except Exception as e:
        logger.error(f"Failed to load class definitions: {str(e)}")
        return False

def analyze_dataset():
    """
    Analyze current dataset distribution to identify underrepresented classes
    """
    if not load_class_definitions():
        return
    
    logger.info("Analyzing dataset distribution...")
    
    # Count images per class in the dataset
    class_counts = {cls: 0 for cls in PIZZA_CLASSES}
    
    # Check primary dataset locations
    for class_name in PIZZA_CLASSES:
        # Check regular data directories
        data_paths = [
            DATA_DIR / "processed" / class_name,
            DATA_DIR / "raw" / class_name,
            DATA_DIR / "augmented" / class_name
        ]
        
        for path in data_paths:
            if path.exists():
                # Count PNG and JPG files
                files = list(path.glob("*.jpg")) + list(path.glob("*.png"))
                class_counts[class_name] += len(files)
    
    # Calculate statistics
    total_images = sum(class_counts.values())
    avg_images = total_images / len(class_counts) if class_counts else 0
    
    # Identify underrepresented classes (less than 80% of the average)
    underrepresented = {
        cls: count for cls, count in class_counts.items() 
        if count < 0.8 * avg_images
    }
    
    # Sort classes by count
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1])
    
    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_images": total_images,
        "average_per_class": avg_images,
        "class_distribution": class_counts,
        "underrepresented_classes": underrepresented,
        "recommended_generation": {
            cls: max(int(avg_images * 0.8) - count, 10) 
            for cls, count in underrepresented.items()
        }
    }
    
    # Save report
    os.makedirs("output/data_analysis", exist_ok=True)
    with open("output/data_analysis/diffusion_class_distribution.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\nDataset Analysis Summary:")
    print(f"Total images: {total_images}")
    print(f"Average images per class: {avg_images:.1f}")
    print("\nClass distribution:")
    for cls, count in sorted_counts:
        print(f"  {cls}: {count} images ({count/total_images*100:.1f}%)")
    
    print("\nUnderrepresented classes:")
    for cls, count in sorted(underrepresented.items(), key=lambda x: x[1]):
        shortfall = int(avg_images * 0.8) - count
        print(f"  {cls}: {count} images (shortfall: {shortfall})")
    
    print("\nRecommended generation targets:")
    for cls, target in sorted(report["recommended_generation"].items(), key=lambda x: -x[1]):
        print(f"  {cls}: {target} additional images")
    
    return report

def run_diffusion_generator(preset=None, model="sd-food", image_size=512, batch_size=1, 
                           output_dir=None, custom_classes=None, custom_counts=None):
    """
    Run the memory-optimized generator with specified parameters
    """
    logger.info(f"Running diffusion generator with preset: {preset}, model: {model}")
    
    # Ensure memory_optimized_generator.py is executable
    os.chmod("memory_optimized_generator.py", 0o755)
    
    # Build command
    cmd = ["./memory_optimized_generator.py"]
    
    # Add preset if specified
    if preset:
        cmd.append(f"--preset={preset}")
    
    # Add model
    cmd.append(f"--model={model}")
    
    # Add image size
    cmd.append(f"--image_size={image_size}")
    
    # Add batch size
    cmd.append(f"--batch_size={batch_size}")
    
    # Add memory optimizations for NVIDIA RTX 3060
    cmd.append("--offload_to_cpu")
    cmd.append("--expand_segments")
    
    # Add output directory
    if output_dir:
        cmd.append(f"--output_dir={output_dir}")
    else:
        # Use timestamp for unique directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = SYNTHETIC_DIR / f"diffusion_run_{timestamp}"
        cmd.append(f"--output_dir={output_path}")
    
    # Execute command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        logger.info(f"Generator completed with return code: {result.returncode}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Generator failed with error: {str(e)}")
        return False

def organize_generated_images():
    """
    Organize generated images into class directories
    """
    if not load_class_definitions():
        return
    
    logger.info("Organizing generated images into class directories...")
    
    # Find the most recent generation directory
    generation_dirs = [d for d in SYNTHETIC_DIR.glob("pizza_dataset_*") if d.is_dir()]
    
    if not generation_dirs:
        logger.error("No generation directories found")
        return False
    
    # Sort by modification time (newest first)
    latest_dir = sorted(generation_dirs, key=lambda d: d.stat().st_mtime, reverse=True)[0]
    logger.info(f"Processing latest generation directory: {latest_dir}")
    
    # Load generation config
    config_file = latest_dir / "generation_config.json"
    if not config_file.exists():
        logger.error(f"Config file not found in {latest_dir}")
        return False
    
    with open(config_file, "r") as f:
        config = json.load(f)
    
    # Process images based on metadata
    images_processed = 0
    
    # Create class directories if they don't exist
    for class_name in PIZZA_CLASSES:
        (SYNTHETIC_DIR / class_name).mkdir(exist_ok=True)
    
    # Process each image in the generation directory
    for image_path in latest_dir.glob("*.jpg"):
        # Check for accompanying metadata JSON
        metadata_path = image_path.with_suffix(".json")
        
        if not metadata_path.exists():
            logger.warning(f"No metadata found for {image_path}, skipping")
            continue
        
        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Get the primary class from metadata
        if "pizza_class" in metadata:
            primary_class = metadata["pizza_class"]
        elif "dominant_class" in metadata:
            primary_class = metadata["dominant_class"]
        elif "stage" in metadata:
            primary_class = metadata["stage"]
        else:
            logger.warning(f"No class found in metadata for {image_path}")
            continue
        
        # Ensure the class is valid
        if primary_class not in PIZZA_CLASSES:
            logger.warning(f"Unknown class '{primary_class}' for {image_path}")
            continue
        
        # Copy to the appropriate class directory
        target_dir = SYNTHETIC_DIR / primary_class
        target_dir.mkdir(exist_ok=True)
        
        # Generate a unique filename
        new_filename = f"synthetic_{primary_class}_{image_path.stem}_{random.randint(1000, 9999)}.jpg"
        target_path = target_dir / new_filename
        
        # Copy image and metadata
        shutil.copy2(image_path, target_path)
        shutil.copy2(metadata_path, target_path.with_suffix(".json"))
        
        images_processed += 1
    
    logger.info(f"Processed {images_processed} images from {latest_dir}")
    
    # Update stats file
    stats_path = SYNTHETIC_DIR / "generation_stats.json"
    stats = {}
    
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)
    
    # Count images in each class directory
    class_counts = {}
    for class_name in PIZZA_CLASSES:
        class_dir = SYNTHETIC_DIR / class_name
        if class_dir.exists():
            # Count JPG files
            files = list(class_dir.glob("*.jpg"))
            class_counts[class_name] = len(files)
    
    # Update stats
    stats.update({
        "last_updated": datetime.now().isoformat(),
        "last_generation": str(latest_dir),
        "total_generated": sum(class_counts.values()),
        "class_distribution": class_counts
    })
    
    # Save updated stats
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Updated generation stats in {stats_path}")
    return True

def integrate_into_dataset():
    """
    Integrate organized images into the main dataset
    """
    if not load_class_definitions():
        return
    
    logger.info("Integrating diffusion-generated images into main dataset...")
    
    # Source directories for synthetic images
    source_dirs = {cls: SYNTHETIC_DIR / cls for cls in PIZZA_CLASSES if (SYNTHETIC_DIR / cls).exists()}
    
    # Create target directories if they don't exist
    for class_name in PIZZA_CLASSES:
        (DATA_DIR / "synthetic" / class_name).mkdir(parents=True, exist_ok=True)
    
    # Copy images to main dataset with appropriate metadata
    images_integrated = 0
    
    for class_name, source_dir in source_dirs.items():
        # Get all JPG files in the source directory
        image_files = list(source_dir.glob("*.jpg"))
        
        if not image_files:
            logger.info(f"No images found in {source_dir}")
            continue
        
        # Target directory
        target_dir = DATA_DIR / "synthetic" / class_name
        
        for image_path in image_files:
            # Check if metadata exists
            metadata_path = image_path.with_suffix(".json")
            
            # Define target paths
            target_image = target_dir / image_path.name
            target_metadata = target_dir / metadata_path.name
            
            # Skip if already exists
            if target_image.exists():
                continue
            
            # Copy image
            shutil.copy2(image_path, target_image)
            
            # Copy metadata if exists
            if metadata_path.exists():
                shutil.copy2(metadata_path, target_metadata)
            
            images_integrated += 1
    
    logger.info(f"Integrated {images_integrated} images into the main dataset")
    return images_integrated

def create_quality_control_report():
    """
    Create an HTML report for manual quality checking of generated images
    """
    if not load_class_definitions():
        return
    
    logger.info("Creating quality control report...")
    
    # Create directory for report
    report_dir = Path("output/data_analysis/quality_control")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Start HTML content
    html_content = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <title>Diffusion-Generated Images Quality Control</title>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 20px; }",
        "        h1, h2 { color: #333; }",
        "        .image-grid { display: flex; flex-wrap: wrap; gap: 10px; }",
        "        .image-item { margin: 5px; text-align: center; }",
        "        .image-item img { width: 150px; height: 150px; object-fit: cover; border: 1px solid #ddd; }",
        "        .image-caption { font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 150px; }",
        "        .class-section { margin-bottom: 30px; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <h1>Diffusion-Generated Images Quality Control</h1>",
        f"    <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>",
        "    <p>Review the images below for quality issues before adding to the main dataset.</p>"
    ]
    
    # Add images by class
    for class_name in PIZZA_CLASSES:
        class_dir = SYNTHETIC_DIR / class_name
        
        if not class_dir.exists() or not list(class_dir.glob("*.jpg")):
            continue
        
        # Add class section
        html_content.extend([
            f"    <div class='class-section'>",
            f"        <h2>{class_name.capitalize()} - {PIZZA_CLASSES[class_name]['description']}</h2>",
            f"        <div class='image-grid'>"
        ])
        
        # Add images (up to 50 per class)
        image_files = list(class_dir.glob("*.jpg"))[:50]
        
        for img_path in image_files:
            # Create relative path to image
            rel_path = os.path.relpath(img_path, report_dir)
            
            # Create symlink in report directory to avoid copying
            link_path = report_dir / img_path.name
            if not link_path.exists():
                os.symlink(os.path.abspath(img_path), link_path)
            
            html_content.extend([
                f"            <div class='image-item'>",
                f"                <img src='{img_path.name}' alt='{img_path.name}' />",
                f"                <div class='image-caption'>{img_path.name}</div>",
                f"            </div>"
            ])
        
        html_content.extend([
            "        </div>",
            "    </div>"
        ])
    
    # Close HTML
    html_content.extend([
        "</body>",
        "</html>"
    ])
    
    # Write HTML report
    report_path = report_dir / "quality_control.html"
    with open(report_path, "w") as f:
        f.write("\n".join(html_content))
    
    logger.info(f"Created quality control report at {report_path}")
    return str(report_path)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Diffusion Model Integration for Pizza Dataset")
    
    # Action options
    parser.add_argument("--analyze-dataset", action="store_true",
                        help="Analyze current dataset distribution")
    parser.add_argument("--generate", action="store_true",
                        help="Generate new images using diffusion model")
    parser.add_argument("--organize-images", action="store_true",
                        help="Organize generated images into class directories")
    parser.add_argument("--integrate", action="store_true",
                        help="Integrate organized images into the main dataset")
    parser.add_argument("--quality-report", action="store_true",
                        help="Create quality control report for manual checking")
    parser.add_argument("--all", action="store_true",
                        help="Run all steps in sequence")
                        
    # Generation options
    parser.add_argument("--preset", type=str, choices=["small_diverse", "training_focused", 
                                                     "progression_heavy", "burn_patterns"],
                        help="Generation preset to use")
    parser.add_argument("--model", type=str, default="sd-food",
                        choices=["sdxl", "sdxl-turbo", "sd-food", "kandinsky"],
                        help="Diffusion model to use")
    parser.add_argument("--image-size", type=int, default=512,
                        help="Size of generated images")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for generation")
    
    args = parser.parse_args()
    
    # Load class definitions
    load_class_definitions()
    
    # Run selected actions
    if args.all or args.analyze_dataset:
        analyze_dataset()
    
    if args.all or args.generate:
        preset = args.preset or "small_diverse"
        run_diffusion_generator(
            preset=preset,
            model=args.model,
            image_size=args.image_size,
            batch_size=args.batch_size
        )
    
    if args.all or args.organize_images:
        organize_generated_images()
    
    if args.all or args.integrate:
        integrate_into_dataset()
    
    if args.all or args.quality_report:
        report_path = create_quality_control_report()
        print(f"\nQuality control report created at {report_path}")
        print("Please review this report before integrating images into the main dataset.")

if __name__ == "__main__":
    main()
