#!/usr/bin/env python3
"""
Remove Test Images Leakage (DATEN-5.1)

This script removes images from the training/source directories that are also in the test set
to prevent data leakage.
"""

import json
import shutil
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent
    
    # Load the leakage report
    report_path = project_root / "output" / "data_analysis" / "class_distribution_test.json"
    
    try:
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        leaked_files = report_data.get('leakage_check', {}).get('leaked_files_basenames', [])
        
        if not leaked_files:
            print("No leaked files found in the report. Nothing to do.")
            return
        
        print(f"Found {len(leaked_files)} leaked files to remove from training/source directories.")
        
        # Source directories to check
        source_dirs = [
            project_root / "augmented_pizza",
            project_root / "augmented_pizza_legacy",
            project_root / "data" / "raw",
            project_root / "data" / "processed",
            project_root / "data" / "classified",
            project_root / "data" / "augmented",
            project_root / "data" / "train",
            project_root / "data" / "validation"
        ]
        
        # Create a backup directory
        backup_dir = project_root / "data" / "leakage_backup"
        backup_dir.mkdir(exist_ok=True, parents=True)
        print(f"Created backup directory: {backup_dir}")
        
        # Find and move leaked files
        files_moved = 0
        for source_dir in source_dirs:
            if not source_dir.is_dir():
                continue
            
            print(f"Checking directory: {source_dir}")
            
            for leaked_basename in leaked_files:
                # Find the file recursively
                for ext in ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG']:
                    for file_path in source_dir.rglob(f"*{leaked_basename}"):
                        if file_path.is_file():
                            # Create backup with directory structure
                            rel_path = file_path.relative_to(source_dir)
                            backup_path = backup_dir / f"{source_dir.name}_{rel_path}"
                            backup_path.parent.mkdir(exist_ok=True, parents=True)
                            
                            # Move the file to backup
                            shutil.copy2(file_path, backup_path)
                            file_path.unlink()
                            files_moved += 1
                            print(f"  Moved: {file_path} -> {backup_path}")
        
        print(f"\nCompleted: Moved {files_moved} leaked files to backup directory")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
