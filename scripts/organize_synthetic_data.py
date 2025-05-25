#!/usr/bin/env python3
"""
Organize filtered synthetic data into class subdirectories to match the expected structure.
"""

import os
import json
import shutil
from pathlib import Path

def organize_synthetic_data():
    """Organize synthetic data into class subdirectories."""
    
    base_dir = Path("/home/emilio/Documents/ai/pizza")
    filtered_dir = base_dir / "data" / "synthetic_filtered"
    
    # Check if the evaluation data has class information
    evaluation_file = base_dir / "output" / "diffusion_analysis" / "full_synthetic_evaluation_20250524_025813.json"
    
    if not evaluation_file.exists():
        print(f"Evaluation file not found: {evaluation_file}")
        return
    
    print("Loading evaluation data...")
    with open(evaluation_file) as f:
        evaluation_data = json.load(f)
    
    # Create class directories
    class_dirs = ["basic", "burnt", "mixed", "combined", "progression", "segment"]
    for class_name in class_dirs:
        class_dir = filtered_dir / class_name
        class_dir.mkdir(exist_ok=True)
    
    # Map images to classes based on naming convention or evaluation data
    # Most synthetic images seem to follow the pattern synthetic_{class_id}_*
    class_mapping = {
        "0": "basic",     # Basic pizza stage
        "4": "basic",     # Basic variations
        "8": "burnt",     # Burnt stage
        "12": "burnt",    # Burnt variations
        "16": "mixed",    # Mixed toppings
        "20": "mixed",    # Mixed variations
        "24": "combined", # Combined stages
        "28": "combined", # Combined variations
        "30": "progression", # Progression sequences
        "32": "progression", # Progression variations
        "35": "segment",  # Segment analysis
        "36": "segment",  # Segment variations
        "40": "basic",    # Additional basic
        "45": "basic",    # Additional basic variations
    }
    
    moved_count = 0
    unmapped_count = 0
    
    print("Organizing synthetic images...")
    
    # Process all images in the filtered directory
    for image_file in filtered_dir.glob("*.jpg"):
        if image_file.is_file():
            # Extract class identifier from filename
            filename = image_file.name
            
            # Try to map based on filename pattern
            class_assigned = None
            
            # Check for patterns like synthetic_{class_id}_*
            if filename.startswith("synthetic_"):
                parts = filename.replace("synthetic_", "").split("_")
                if len(parts) > 0:
                    class_id = parts[0]
                    if class_id in class_mapping:
                        class_assigned = class_mapping[class_id]
            
            # If no mapping found, assign to basic as default
            if class_assigned is None:
                class_assigned = "basic"
                unmapped_count += 1
            
            # Move image to appropriate class directory
            destination = filtered_dir / class_assigned / filename
            try:
                shutil.move(str(image_file), str(destination))
                moved_count += 1
                if moved_count % 100 == 0:
                    print(f"Moved {moved_count} images...")
            except Exception as e:
                print(f"Error moving {filename}: {e}")
    
    print(f"\nOrganization complete:")
    print(f"  Total images moved: {moved_count}")
    print(f"  Unmapped images (assigned to basic): {unmapped_count}")
    
    # Count images per class
    for class_name in class_dirs:
        class_dir = filtered_dir / class_name
        image_count = len(list(class_dir.glob("*.jpg")))
        print(f"  {class_name}: {image_count} images")

if __name__ == "__main__":
    organize_synthetic_data()
