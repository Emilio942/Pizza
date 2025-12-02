#!/usr/bin/env python3
"""
Aufgabe 2.1: Vorbereitung des gesamten Pizza-Verifier-Datensatzes

This module combines positive and negative examples with special weighting for 
food-safety-critical cases and prepares train/validation/test splits.
"""

import json
import random
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

@dataclass
class DatasetSplit:
    """Container for dataset split with metadata."""
    samples: List[Dict[str, Any]]
    split_name: str
    total_count: int
    positive_count: int
    negative_count: int
    food_safety_critical_count: int
    class_distribution: Dict[str, int]
    quality_statistics: Dict[str, float]

class PizzaVerifierDatasetPreparator:
    """Prepares the complete pizza verifier dataset with proper splits and weighting."""
    
    def __init__(self, project_root: str = "/home/emilio/Documents/ai/pizza"):
        self.project_root = Path(project_root)
        self.verification_data_dir = self.project_root / "output" / "verification_data"
        
        # Input files
        self.positive_examples_file = self.verification_data_dir / "pizza_positive_examples_comprehensive.json"
        self.hard_negatives_file = self.verification_data_dir / "pizza_hard_negatives_comprehensive.json"
        
        # Output directory
        self.output_dir = self.verification_data_dir / "prepared_dataset"
        self.output_dir.mkdir(exist_ok=True)
        
    def load_dataset(self, file_path: Path) -> Tuple[List[Dict], Dict]:
        """Load dataset from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data.get('samples', []), data.get('dataset_info', {})
        except FileNotFoundError:
            print(f"Warning: File not found: {file_path}")
            return [], {}
    
    def apply_food_safety_weighting(self, samples: List[Dict]) -> List[Dict]:
        """Apply special weighting for food safety critical cases."""
        
        weighted_samples = []
        
        for sample in samples:
            # Create weighted copy
            weighted_sample = sample.copy()
            
            # Check if food safety critical
            is_food_safety = sample.get('food_safety_critical', False)
            
            # Apply weighting through adjusted quality score and sample duplication
            if is_food_safety:
                # Lower quality score for critical errors (make them more important)
                if sample.get('quality_score', 0.5) > 0.5:
                    # This is a positive example marked as food safety critical (unusual)
                    weighted_sample['quality_score'] = max(sample['quality_score'] * 0.9, 0.0)
                else:
                    # This is a negative example that's food safety critical
                    weighted_sample['quality_score'] = min(sample['quality_score'] * 0.8, 1.0)
                
                # Add weight metadata
                weighted_sample['sample_weight'] = 2.0  # Double weight for food safety cases
                weighted_sample['priority'] = 'high'
                
                # Duplicate critical food safety samples for emphasis
                weighted_samples.append(weighted_sample)
                
                # Add second copy with slight variation for robust learning
                duplicate_sample = weighted_sample.copy()
                duplicate_sample['sample_weight'] = 1.5
                duplicate_sample['duplicate_id'] = 1
                weighted_samples.append(duplicate_sample)
            else:
                # Normal weighting
                weighted_sample['sample_weight'] = 1.0
                weighted_sample['priority'] = 'normal'
                weighted_samples.append(weighted_sample)
        
        return weighted_samples
    
    def balance_class_distribution(self, samples: List[Dict]) -> List[Dict]:
        """Balance class distribution while respecting existing pizza project distribution."""
        
        # Group samples by prediction class
        class_groups = {}
        for sample in samples:
            pred_class = sample.get('model_prediction', 'unknown')
            if pred_class not in class_groups:
                class_groups[pred_class] = []
            class_groups[pred_class].append(sample)
        
        print(f"   Original class distribution: {[(k, len(v)) for k, v in class_groups.items()]}")
        
        # Find target count per class (based on minimum class size)
        class_sizes = [len(samples) for samples in class_groups.values()]
        target_per_class = max(min(class_sizes), 10)  # At least 10 samples per class
        
        # Oversample classes with fewer samples
        balanced_samples = []
        for class_name, class_samples in class_groups.items():
            if len(class_samples) >= target_per_class:
                # Take random sample if too many
                selected_samples = random.sample(class_samples, target_per_class)
            else:
                # Oversample by repeating samples
                selected_samples = class_samples.copy()
                while len(selected_samples) < target_per_class:
                    additional_samples = random.sample(class_samples, 
                                                     min(len(class_samples), target_per_class - len(selected_samples)))
                    for sample in additional_samples:
                        # Mark as oversampled
                        oversampled = sample.copy()
                        oversampled['oversampled'] = True
                        selected_samples.append(oversampled)
            
            balanced_samples.extend(selected_samples)
        
        print(f"   Balanced class distribution: {target_per_class} samples per class")
        return balanced_samples
    
    def shuffle_dataset(self, samples: List[Dict]) -> List[Dict]:
        """Shuffle dataset with stratified sampling to maintain distribution."""
        
        # Group by positive/negative and food safety status
        positive_samples = [s for s in samples if s.get('quality_score', 0) >= 0.5]
        negative_samples = [s for s in samples if s.get('quality_score', 0) < 0.5]
        
        # Shuffle each group separately
        random.shuffle(positive_samples)
        random.shuffle(negative_samples)
        
        # Interleave to maintain balance
        shuffled_samples = []
        max_len = max(len(positive_samples), len(negative_samples))
        
        for i in range(max_len):
            if i < len(positive_samples):
                shuffled_samples.append(positive_samples[i])
            if i < len(negative_samples):
                shuffled_samples.append(negative_samples[i])
        
        # Final shuffle
        random.shuffle(shuffled_samples)
        
        return shuffled_samples
    
    def create_train_val_test_splits(self, samples: List[Dict]) -> Tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
        """Create 80% train, 10% validation, 10% test splits."""
        
        total_count = len(samples)
        train_size = int(0.8 * total_count)
        val_size = int(0.1 * total_count)
        test_size = total_count - train_size - val_size
        
        # Ensure we have enough samples for each split
        if val_size < 5:
            val_size = min(5, total_count // 5)
        if test_size < 5:
            test_size = min(5, total_count // 5)
        train_size = total_count - val_size - test_size
        
        print(f"   Split sizes: train={train_size}, val={val_size}, test={test_size}")
        
        # Create splits
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]
        
        # Create split objects with statistics
        def create_split_stats(split_samples: List[Dict], split_name: str) -> DatasetSplit:
            positive_count = len([s for s in split_samples if s.get('quality_score', 0) >= 0.5])
            negative_count = len(split_samples) - positive_count
            food_safety_count = len([s for s in split_samples if s.get('food_safety_critical', False)])
            
            # Class distribution
            class_dist = {}
            for sample in split_samples:
                pred_class = sample.get('model_prediction', 'unknown')
                class_dist[pred_class] = class_dist.get(pred_class, 0) + 1
            
            # Quality statistics
            quality_scores = [s.get('quality_score', 0.5) for s in split_samples]
            quality_stats = {
                'mean': round(np.mean(quality_scores), 4),
                'std': round(np.std(quality_scores), 4),
                'min': round(min(quality_scores), 4),
                'max': round(max(quality_scores), 4)
            }
            
            return DatasetSplit(
                samples=split_samples,
                split_name=split_name,
                total_count=len(split_samples),
                positive_count=positive_count,
                negative_count=negative_count,
                food_safety_critical_count=food_safety_count,
                class_distribution=class_dist,
                quality_statistics=quality_stats
            )
        
        train_split = create_split_stats(train_samples, "train")
        val_split = create_split_stats(val_samples, "validation")
        test_split = create_split_stats(test_samples, "test")
        
        return train_split, val_split, test_split
    
    def save_dataset_split(self, split: DatasetSplit) -> str:
        """Save a dataset split to JSON file."""
        
        output_file = self.output_dir / f"pizza_verifier_{split.split_name}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                "split_info": {
                    "split_name": split.split_name,
                    "total_samples": split.total_count,
                    "positive_samples": split.positive_count,
                    "negative_samples": split.negative_count,
                    "food_safety_critical": split.food_safety_critical_count,
                    "class_distribution": split.class_distribution,
                    "quality_statistics": split.quality_statistics,
                    "positive_ratio": round(split.positive_count / split.total_count, 4),
                    "food_safety_ratio": round(split.food_safety_critical_count / split.total_count, 4)
                },
                "samples": split.samples
            }, f, indent=2)
        
        return str(output_file)
    
    def save_dataset_metadata(self, train_split: DatasetSplit, val_split: DatasetSplit, test_split: DatasetSplit) -> str:
        """Save overall dataset metadata."""
        
        output_file = self.output_dir / "dataset_metadata.json"
        
        total_samples = train_split.total_count + val_split.total_count + test_split.total_count
        total_positive = train_split.positive_count + val_split.positive_count + test_split.positive_count
        total_negative = train_split.negative_count + val_split.negative_count + test_split.negative_count
        total_food_safety = train_split.food_safety_critical_count + val_split.food_safety_critical_count + test_split.food_safety_critical_count
        
        with open(output_file, 'w') as f:
            json.dump({
                "dataset_info": {
                    "task": "Aufgabe 2.1: Vorbereitung des gesamten Pizza-Verifier-Datensatzes",
                    "description": "Combined pizza verifier dataset with food safety weighting and train/val/test splits",
                    "total_samples": total_samples,
                    "positive_samples": total_positive,
                    "negative_samples": total_negative,
                    "food_safety_critical": total_food_safety,
                    "splits": {
                        "train": {
                            "samples": train_split.total_count,
                            "percentage": round(100 * train_split.total_count / total_samples, 1)
                        },
                        "validation": {
                            "samples": val_split.total_count,
                            "percentage": round(100 * val_split.total_count / total_samples, 1)
                        },
                        "test": {
                            "samples": test_split.total_count,
                            "percentage": round(100 * test_split.total_count / total_samples, 1)
                        }
                    },
                    "quality_distribution": {
                        "positive_ratio": round(total_positive / total_samples, 4),
                        "negative_ratio": round(total_negative / total_samples, 4),
                        "food_safety_ratio": round(total_food_safety / total_samples, 4)
                    }
                },
                "split_details": {
                    "train": {
                        "total_count": train_split.total_count,
                        "class_distribution": train_split.class_distribution,
                        "quality_statistics": train_split.quality_statistics
                    },
                    "validation": {
                        "total_count": val_split.total_count,
                        "class_distribution": val_split.class_distribution,
                        "quality_statistics": val_split.quality_statistics
                    },
                    "test": {
                        "total_count": test_split.total_count,
                        "class_distribution": test_split.class_distribution,
                        "quality_statistics": test_split.quality_statistics
                    }
                }
            }, f, indent=2)
        
        return str(output_file)
    
    def prepare_complete_dataset(self) -> Tuple[str, str, str, str]:
        """
        Main method to prepare the complete pizza verifier dataset.
        
        Returns:
            Tuple of (train_file, val_file, test_file, metadata_file) paths
        """
        print("🚀 Starting Aufgabe 2.1: Vorbereitung des gesamten Pizza-Verifier-Datensatzes")
        
        # 1. Load positive and negative examples
        print("📂 Loading positive and negative examples...")
        positive_samples, positive_info = self.load_dataset(self.positive_examples_file)
        negative_samples, negative_info = self.load_dataset(self.hard_negatives_file)
        
        print(f"   Loaded {len(positive_samples)} positive examples")
        print(f"   Loaded {len(negative_samples)} negative examples")
        
        # 2. Combine datasets
        print("🔗 Combining datasets...")
        all_samples = positive_samples + negative_samples
        
        # 3. Apply food safety weighting
        print("⚖️  Applying food safety weighting...")
        weighted_samples = self.apply_food_safety_weighting(all_samples)
        food_safety_count = len([s for s in weighted_samples if s.get('food_safety_critical', False)])
        print(f"   Applied special weighting to {food_safety_count} food safety critical samples")
        
        # 4. Balance class distribution
        print("⚖️  Balancing class distribution...")
        balanced_samples = self.balance_class_distribution(weighted_samples)
        
        # 5. Shuffle dataset
        print("🔀 Shuffling dataset...")
        shuffled_samples = self.shuffle_dataset(balanced_samples)
        
        # 6. Create train/validation/test splits
        print("✂️  Creating train/validation/test splits...")
        train_split, val_split, test_split = self.create_train_val_test_splits(shuffled_samples)
        
        # 7. Save all splits
        print("💾 Saving dataset splits...")
        train_file = self.save_dataset_split(train_split)
        val_file = self.save_dataset_split(val_split)
        test_file = self.save_dataset_split(test_split)
        metadata_file = self.save_dataset_metadata(train_split, val_split, test_split)
        
        # 8. Print summary
        print("📊 Dataset preparation complete!")
        print(f"   Total samples: {len(shuffled_samples)}")
        print(f"   Train: {train_split.total_count} samples ({100*train_split.total_count/len(shuffled_samples):.1f}%)")
        print(f"   Validation: {val_split.total_count} samples ({100*val_split.total_count/len(shuffled_samples):.1f}%)")
        print(f"   Test: {test_split.total_count} samples ({100*test_split.total_count/len(shuffled_samples):.1f}%)")
        print(f"   Food safety critical: {food_safety_count} samples")
        
        return train_file, val_file, test_file, metadata_file

def main():
    """Main execution for Aufgabe 2.1"""
    random.seed(42)  # For reproducible splits
    
    preparator = PizzaVerifierDatasetPreparator()
    
    train_file, val_file, test_file, metadata_file = preparator.prepare_complete_dataset()
    
    print(f"✅ Aufgabe 2.1 completed successfully!")
    print(f"📄 Output files:")
    print(f"   Train: {train_file}")
    print(f"   Validation: {val_file}")
    print(f"   Test: {test_file}")
    print(f"   Metadata: {metadata_file}")
    print(f"🔄 Next task: Aufgabe 2.2 - Implementierung und Training des Pizza-Verifier-Modells")

if __name__ == "__main__":
    main()
