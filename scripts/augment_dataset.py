#!/usr/bin/env python3
"""
Enhanced Pizza Dataset Augmentation Script

This script applies a wide range of augmentation techniques to the pizza dataset
to increase training data variety and improve model robustness for pizza-related
machine learning tasks.

Usage:
    python augment_dataset.py [options]

Options:
    --input-dir: Directory containing the original pizza images (default: data/classified)
    --output-dir: Directory to save augmented images (default: data/augmented)
    --num-per-image: Number of augmented images to generate per original image (default: 5)
    --aug-types: Types of augmentations to apply (comma-separated)
                 Options: basic,burning,mixup,cutmix,progression,segment,lighting,perspective,all (default: all)
    --use-gpu: Use GPU for augmentation if available (default: True)
    --seed: Random seed for reproducibility
    --target-size: Target image size in pixels (default: 224)
    --batch-size: Batch size for efficient processing (default: 16)
    --save-stats: Save augmentation statistics to JSON file (default: False)

Example:
    python augment_dataset.py --input-dir data/raw --output-dir data/augmented --num-per-image 10 --aug-types basic,burning
"""

import os
import sys
import argparse
import random
import time
import json
import gc
import warnings
from pathlib import Path
from contextlib import contextmanager
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from tqdm import tqdm

# Handle optional SciPy dependency
try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not installed - some filters will use alternative implementations.")

# Handle PyTorch for advanced augmentations
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from torchvision.transforms import functional as TVF
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed - advanced augmentations will be disabled.")

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ================ UTILITIES AND HELPER FUNCTIONS ================

@contextmanager
def open_image(path):
    """Context manager for safely opening and closing images"""
    try:
        img = Image.open(path).convert('RGB')
        yield img
    except Exception as e:
        print(f"Error opening image {path}: {e}")
        raise
    finally:
        if 'img' in locals() and img is not None:
            img.close()

def select_device(use_gpu=True):
    """Select appropriate device for computation"""
    if not TORCH_AVAILABLE:
        return None
    
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif use_gpu and hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Metal Performance Shaders")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device

def get_optimal_batch_size(suggested_batch_size, device):
    """Determine optimal batch size based on available memory"""
    if not TORCH_AVAILABLE:
        return 8  # Default for non-PyTorch mode
        
    if device is None or device.type == 'cpu':
        # For CPU: Use a conservative value
        return min(8, suggested_batch_size)
    
    if device.type == 'cuda':
        # Conservative estimate of available GPU memory
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            already_used = torch.cuda.memory_allocated(0)
            available = total_memory - already_used
            
            # Assume a typical image takes ~2MB in memory
            estimated_image_size = 2 * 1024 * 1024  # 2MB in bytes
            
            # Leave 20% headroom
            safe_memory = available * 0.8
            max_images = int(safe_memory / estimated_image_size)
            
            # Limit batch size to a reasonable range
            optimal_batch_size = max(4, min(max_images, suggested_batch_size))
            
            # Round down to nearest power of two
            optimal_batch_size = 2 ** int(np.log2(optimal_batch_size))
            
            print(f"Optimal batch size for GPU: {optimal_batch_size}")
            return optimal_batch_size
        except Exception as e:
            print(f"Error determining optimal batch size: {e}")
            return 8  # Default fallback
    
    # Default fallback for other devices
    return 8

def validate_and_prepare_paths(args):
    """Validate and prepare input/output paths"""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Check if input directory exists
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Create output directory and subdirectories
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If specific augmentation types are requested, create those subdirectories
    subdirs = {}
    
    # Create subdirectories for each augmentation type if needed
    aug_types = args.aug_types.split(',')
    if 'all' in aug_types:
        aug_types = ['basic', 'burning', 'mixup', 'cutmix', 'progression', 'segment', 'lighting', 'perspective']
    
    for aug_type in aug_types:
        subdirs[aug_type] = output_dir / aug_type
        subdirs[aug_type].mkdir(exist_ok=True)
    
    return input_dir, output_dir, subdirs

def get_image_files(input_dir):
    """Collect all valid image files in the input directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    # Search for files with image extensions
    for file in input_dir.glob('**/*'):
        if file.is_file() and file.suffix.lower() in image_extensions:
            try:
                # Briefly open the image to validate it
                with Image.open(file) as img:
                    img.verify()  # Validate the image
                image_files.append(file)
            except Exception as e:
                print(f"Warning: Cannot process image {file}: {e}")
    
    if not image_files:
        raise ValueError(f"No valid images found in directory: {input_dir}")
    
    return image_files

def show_images(images, titles=None, cols=5, figsize=(15, 10), save_path=None):
    """Display multiple images in a grid with optional saving"""
    if not images:
        print("No images to display")
        return
    
    rows = len(images) // cols + (1 if len(images) % cols != 0 else 0)
    plt.figure(figsize=figsize)
    
    for i, img in enumerate(images):
        if i >= rows * cols:
            break
            
        plt.subplot(rows, cols, i + 1)
        
        if TORCH_AVAILABLE and isinstance(img, torch.Tensor):
            # Convert tensor to NumPy for display
            img = img.cpu().detach()
            if img.device != torch.device('cpu'):
                img = img.cpu()
            img = img.permute(1, 2, 0).numpy()
            # Normalize to [0,1] if needed
            img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.axis('off')
        if titles is not None and i < len(titles):
            plt.title(titles[i])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Image saved to: {save_path}")
    
    plt.show()

def save_augmented_images(images, output_dir, base_filename, batch_size=16, metadata=None):
    """Save augmented images in batches with consistent size"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    TARGET_SIZE = 224  # Fixed target size for all images
    
    # Process images in batches
    saved_count = 0
    for batch_idx in range(0, len(images), batch_size):
        batch_end = min(batch_idx + batch_size, len(images))
        batch = images[batch_idx:batch_end]
        
        for i, img in enumerate(batch):
            idx = batch_idx + i
            output_path = output_dir / f"{base_filename}_{idx:04d}.jpg"
            
            try:
                if TORCH_AVAILABLE and isinstance(img, torch.Tensor):
                    # Ensure image is on CPU
                    if img.device != torch.device('cpu'):
                        img = img.cpu()
                    img = img.detach()
                    
                    # Normalize to [0,1] if needed
                    if img.min() < 0 or img.max() > 1:
                        img = torch.clamp(img, 0, 1)
                    
                    # Ensure uniform size
                    current_size = img.shape[1:3]
                    if current_size[0] != TARGET_SIZE or current_size[1] != TARGET_SIZE:
                        img = F.interpolate(img.unsqueeze(0), size=(TARGET_SIZE, TARGET_SIZE), 
                                           mode='bilinear', align_corners=False).squeeze(0)
                    
                    # Convert tensor to PIL for saving
                    img = TVF.to_pil_image(img)
                elif isinstance(img, np.ndarray):
                    # Handle NumPy arrays
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    img = Image.fromarray(img.astype(np.uint8))
                    
                    # Ensure uniform size
                    if img.width != TARGET_SIZE or img.height != TARGET_SIZE:
                        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
                elif isinstance(img, Image.Image):
                    # Ensure uniform size for PIL images
                    if img.width != TARGET_SIZE or img.height != TARGET_SIZE:
                        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
                
                # Save with quality settings
                img.save(output_path, quality=92, optimize=True)
                saved_count += 1
                
                # Save metadata if provided
                if metadata:
                    meta_path = output_path.with_suffix('.json')
                    with open(meta_path, 'w') as f:
                        json.dump(metadata, f)
            except Exception as e:
                print(f"Error saving image {output_path}: {e}")
        
        # Explicitly free memory
        del batch
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return saved_count

class AugmentationStats:
    """Class for tracking and saving augmentation statistics"""
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.start_time = time.time()
        self.stats = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": 0,
            "input_images": 0,
            "output_images": 0,
            "augmentation_types": {}
        }
    
    def update(self, augmentation_type, count, metadata=None):
        """Update statistics for an augmentation type"""
        if augmentation_type not in self.stats["augmentation_types"]:
            self.stats["augmentation_types"][augmentation_type] = {
                "count": 0,
                "metadata": {}
            }
        
        self.stats["augmentation_types"][augmentation_type]["count"] += count
        self.stats["output_images"] += count
        
        if metadata:
            for key, value in metadata.items():
                self.stats["augmentation_types"][augmentation_type]["metadata"][key] = value
    
    def set_input_count(self, count):
        """Set the number of input images"""
        self.stats["input_images"] = count
    
    def save(self):
        """Save statistics to a JSON file"""
        self.stats["duration_seconds"] = time.time() - self.start_time
        
        # Calculate useful metrics
        if self.stats["input_images"] > 0:
            self.stats["images_per_original"] = self.stats["output_images"] / self.stats["input_images"]
        
        # Save as JSON
        stats_file = self.output_dir / "augmentation_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"Statistics saved to {stats_file}")

# ================ BASIC AUGMENTATION FUNCTION ================

def apply_basic_augmentation(img, device=None):
    """Apply basic augmentation to a single image (fallback implementation)"""
    if TORCH_AVAILABLE and isinstance(img, torch.Tensor):
        # Convert tensor to PIL image for basic operations
        if device and img.device != device:
            img = img.to(device)
        pil_img = TVF.to_pil_image(img.cpu())
        
        # Create a list of basic transforms
        transforms_list = []
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            rotated = pil_img.rotate(angle, resample=Image.BICUBIC)
            transforms_list.append(rotated)
        
        # Random flip
        if random.random() > 0.5:
            flipped = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            transforms_list.append(flipped)
        
        # Random crop
        if random.random() > 0.5:
            width, height = pil_img.size
            crop_size = random.uniform(0.8, 0.95)
            crop_width, crop_height = int(width * crop_size), int(height * crop_size)
            left = random.randint(0, width - crop_width)
            top = random.randint(0, height - crop_height)
            cropped = pil_img.crop((left, top, left + crop_width, top + crop_height))
            cropped = cropped.resize((width, height), Image.BICUBIC)
            transforms_list.append(cropped)
        
        # Random brightness/contrast/saturation adjustment
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(pil_img)
            brightened = enhancer.enhance(random.uniform(0.8, 1.2))
            transforms_list.append(brightened)
        
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(pil_img)
            contrasted = enhancer.enhance(random.uniform(0.8, 1.2))
            transforms_list.append(contrasted)
        
        if random.random() > 0.5:
            enhancer = ImageEnhance.Color(pil_img)
            colored = enhancer.enhance(random.uniform(0.8, 1.2))
            transforms_list.append(colored)
        
        # If no transforms were applied, return the original image
        if not transforms_list:
            return img
        
        # Pick a random transformed image
        result_pil = random.choice(transforms_list)
        
        # Convert back to tensor
        result = TVF.to_tensor(result_pil)
        if device:
            result = result.to(device)
        
        return result
    
    elif isinstance(img, Image.Image):
        # Apply transformations to PIL image
        original_size = img.size
        transforms_list = []
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            rotated = img.rotate(angle, resample=Image.BICUBIC)
            transforms_list.append(rotated)
        
        # Random flip
        if random.random() > 0.5:
            flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            transforms_list.append(flipped)
        
        # Random crop
        if random.random() > 0.5:
            width, height = img.size
            crop_size = random.uniform(0.8, 0.95)
            crop_width, crop_height = int(width * crop_size), int(height * crop_size)
            left = random.randint(0, width - crop_width)
            top = random.randint(0, height - crop_height)
            cropped = img.crop((left, top, left + crop_width, top + crop_height))
            cropped = cropped.resize(original_size, Image.BICUBIC)
            transforms_list.append(cropped)
        
        # Apply brightness/contrast/saturation adjustment
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(img)
            brightened = enhancer.enhance(random.uniform(0.8, 1.2))
            transforms_list.append(brightened)
        
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(img)
            contrasted = enhancer.enhance(random.uniform(0.8, 1.2))
            transforms_list.append(contrasted)
        
        if random.random() > 0.5:
            enhancer = ImageEnhance.Color(img)
            colored = enhancer.enhance(random.uniform(0.8, 1.2))
            transforms_list.append(colored)
        
        # If no transforms were applied, return the original image
        if not transforms_list:
            return img
        
        # Pick a random transformed image
        return random.choice(transforms_list)
    
    else:
        # Unsupported input type
        print("Warning: Unsupported image type for basic augmentation")
        return img

# ================ DATASET AND DATALOADER CLASSES ================

# Import advanced augmentation modules conditionally to handle potential missing dependencies
try:
    from scripts.augment_classes import (
        PizzaBurningEffect, OvenEffect, PizzaSegmentEffect,
        DirectionalLightEffect, CLAHEEffect, ExposureVariationEffect, 
        PerspectiveTransformEffect, create_shadow_mask
    )
    from scripts.augment_functions import (
        apply_burning_augmentation, 
        apply_mixup, apply_cutmix, 
        apply_burning_progression, apply_segment_augmentation,
        apply_lighting_augmentation, apply_perspective_augmentation,
        apply_combined_light_perspective_augmentation
    )
    AUGMENTATION_MODULES_AVAILABLE = True
except ImportError:
    AUGMENTATION_MODULES_AVAILABLE = False

class PizzaAugmentationDataset(Dataset):
    """Enhanced dataset class for pizza image augmentation"""
    
    def __init__(self, image_paths, transform=None, device=None, target_size=(224, 224), cache_size=0):
        self.image_paths = image_paths
        self.transform = transform
        self.device = device
        self.target_size = target_size
        
        # Optional image cache for frequently used images
        self.cache_size = min(cache_size, len(image_paths)) if cache_size > 0 else 0
        self.cache = {}
        
        if self.cache_size > 0:
            print(f"Initializing image cache with {self.cache_size} images...")
            for i in range(self.cache_size):
                self._load_and_cache(i)
    
    def _load_and_cache(self, idx):
        """Load an image and store it in the cache"""
        if idx in self.cache:
            return
        
        path = self.image_paths[idx]
        try:
            with open_image(path) as img:
                # Resize to target size for consistency
                img = img.resize(self.target_size, Image.BICUBIC)
                # Store as tensor directly in cache
                tensor = TVF.to_tensor(img)
                if self.device:
                    tensor = tensor.to(self.device)
                self.cache[idx] = tensor
        except Exception as e:
            print(f"Error caching image {path}: {e}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Try to load from cache first
        if idx in self.cache:
            image = self.cache[idx]
        else:
            # If not in cache, load from file
            try:
                with open_image(self.image_paths[idx]) as img:
                    # Resize to target size for consistency
                    img = img.resize(self.target_size, Image.BICUBIC)
                    image = TVF.to_tensor(img)
                    if self.device:
                        image = image.to(self.device)
            except Exception as e:
                # Error handling - return a black image
                print(f"Error loading image {self.image_paths[idx]}: {e}")
                image = torch.zeros(3, *self.target_size)
                if self.device:
                    image = image.to(self.device)
        
        # Apply transformation if available
        if self.transform:
            try:
                # If the image is a tensor and the transform expects PIL
                if isinstance(image, torch.Tensor) and not isinstance(self.transform, transforms.Compose):
                    # Convert to PIL for non-tensor transform
                    pil_image = TVF.to_pil_image(image.cpu() if self.device else image)
                    image = self.transform(pil_image)
                    if self.device and isinstance(image, torch.Tensor):
                        image = image.to(self.device)
                else:
                    # Direct transformation
                    image = self.transform(image)
            except Exception as e:
                print(f"Error transforming image {self.image_paths[idx]}: {e}")
        
        return image

# ================ MAIN SCRIPT EXECUTION ================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Enhanced Pizza Dataset Augmentation Script")
    
    parser.add_argument("--input-dir", type=str, default="data/classified",
                        help="Directory containing the original pizza images")
    parser.add_argument("--output-dir", type=str, default="data/augmented",
                        help="Directory to save augmented images")
    parser.add_argument("--num-per-image", type=int, default=5,
                        help="Number of augmented images to generate per original image")
    parser.add_argument("--aug-types", type=str, default="all",
                        help="Types of augmentations to apply (comma-separated): basic,burning,mixup,cutmix,progression,segment,lighting,perspective,all")
    parser.add_argument("--use-gpu", action="store_true", default=True,
                        help="Use GPU for augmentation if available")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--target-size", type=int, default=224,
                        help="Target image size in pixels")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for efficient processing")
    parser.add_argument("--save-stats", action="store_true", default=False,
                        help="Save augmentation statistics to JSON file")
    parser.add_argument("--preview", action="store_true", default=False,
                        help="Preview a sample of augmented images")
    
    return parser.parse_args()

def main():
    """Main function to run the augmentation script"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed)
    
    # Check if augmentation modules are available
    if not TORCH_AVAILABLE:
        print("WARNING: PyTorch not available. Only basic PIL-based augmentations will be used.")
    
    if not AUGMENTATION_MODULES_AVAILABLE:
        print("WARNING: Augmentation modules not found. Some augmentation types will be disabled.")
    
    # Setup device for computations
    device = select_device(args.use_gpu) if TORCH_AVAILABLE else None
    
    # Determine optimal batch size
    batch_size = get_optimal_batch_size(args.batch_size, device)
    
    # Validate and prepare input/output paths
    try:
        input_dir, output_dir, subdirs = validate_and_prepare_paths(args)
    except Exception as e:
        print(f"Error setting up directories: {e}")
        return 1
    
    # Get list of image files to process
    try:
        image_files = get_image_files(input_dir)
        print(f"Found {len(image_files)} valid images in {input_dir}")
    except Exception as e:
        print(f"Error finding image files: {e}")
        return 1
    
    # Initialize statistics tracking if enabled
    stats = AugmentationStats(output_dir) if args.save_stats else None
    if stats:
        stats.set_input_count(len(image_files))
    
    # Create dataset for input images
    pizza_dataset = PizzaAugmentationDataset(
        image_files,
        device=device,
        target_size=(args.target_size, args.target_size),
        cache_size=min(100, len(image_files))  # Cache up to 100 images for efficiency
    )
    
    # Create dataloader for batch processing
    data_loader = DataLoader(
        pizza_dataset,
        batch_size=batch_size,
        shuffle=False,  # Process in order
        num_workers=0,  # No workers for simplicity and to avoid CUDA issues
        pin_memory=False  # Disable pin_memory to avoid issues with GPU tensors
    )
    
    # Parse augmentation types
    aug_types = args.aug_types.split(',')
    if 'all' in aug_types:
        aug_types = ['basic', 'burning', 'mixup', 'cutmix', 'progression', 'segment', 'lighting', 'perspective']
    
    # Prepare augmentation functions
    augmentation_functions = {}
    for aug_type in aug_types:
        if aug_type == 'basic':
            augmentation_functions[aug_type] = apply_basic_augmentation
        elif aug_type == 'burning' and AUGMENTATION_MODULES_AVAILABLE:
            augmentation_functions[aug_type] = apply_burning_augmentation
        elif aug_type == 'mixup' and AUGMENTATION_MODULES_AVAILABLE and TORCH_AVAILABLE:
            augmentation_functions[aug_type] = apply_mixup
        elif aug_type == 'cutmix' and AUGMENTATION_MODULES_AVAILABLE and TORCH_AVAILABLE:
            augmentation_functions[aug_type] = apply_cutmix
        elif aug_type == 'progression' and AUGMENTATION_MODULES_AVAILABLE and TORCH_AVAILABLE:
            augmentation_functions[aug_type] = apply_burning_progression
        elif aug_type == 'segment' and AUGMENTATION_MODULES_AVAILABLE and TORCH_AVAILABLE:
            # Code for the segment augmentation type
            pass
        elif aug_type == 'lighting' and AUGMENTATION_MODULES_AVAILABLE and TORCH_AVAILABLE:
            augmentation_functions[aug_type] = lambda img, d=device: apply_lighting_augmentation(img, d)
        elif aug_type == 'perspective' and AUGMENTATION_MODULES_AVAILABLE and TORCH_AVAILABLE:
            augmentation_functions[aug_type] = lambda img, d=device: apply_perspective_augmentation(img, d)
    
    # Summary of augmentation plan
    available_types = list(augmentation_functions.keys())
    print(f"Will apply the following augmentation types: {', '.join(available_types)}")
    print(f"Generating {args.num_per_image} augmented versions per original image")
    print(f"Target size: {args.target_size}x{args.target_size} pixels")
    
    # Process each batch of images
    total_augmented = 0
    preview_images = [] if args.preview else None
    preview_titles = [] if args.preview else None
    
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing batches")):
        
        # Get the file paths for this batch
        batch_start_idx = batch_idx * batch_size
        batch_end_idx = min(batch_start_idx + batch_size, len(image_files))
        batch_paths = image_files[batch_start_idx:batch_end_idx]
        
        # Process each original image in the batch
        for i, (img, img_path) in enumerate(zip(batch, batch_paths)):
            # Determine output base filename
            rel_path = img_path.relative_to(input_dir) if img_path.is_relative_to(input_dir) else img_path.name
            base_filename = str(rel_path).replace("/", "_").replace("\\", "_").replace(" ", "_")
            base_filename = f"{base_filename.split('.')[0]}"
            
            # For each augmentation type
            for aug_type, aug_func in augmentation_functions.items():
                augmented_images = []
                
                try:
                    # Generate multiple augmented versions
                    for j in range(args.num_per_image):
                        # Apply the augmentation function
                        if aug_type in ['mixup', 'cutmix']:
                            # These need multiple images - get a random one from the dataset
                            random_idx = random.randrange(len(pizza_dataset))
                            img2 = pizza_dataset[random_idx]
                            augmented = aug_func(img, img2, device=device)
                        elif aug_type == 'progression':
                            # Progression needs multiple steps
                            num_steps = min(args.num_per_image, 10)
                            step_idx = j % num_steps
                            progression_factor = step_idx / (num_steps - 1) if num_steps > 1 else 0.5
                            augmented = aug_func(img, progression_factor=progression_factor, device=device)
                        else:
                            # Regular augmentation
                            augmented = aug_func(img, device=device)
                        
                        augmented_images.append(augmented)
                        
                        # Collect a sample for preview
                        if args.preview and j == 0 and len(preview_images) < 10:
                            preview_images.append(augmented)
                            preview_titles.append(f"{aug_type}")
                
                except Exception as e:
                    print(f"Error applying {aug_type} augmentation to {img_path}: {e}")
                    continue
                
                # Save the augmented images
                output_subdir = subdirs[aug_type]
                save_count = save_augmented_images(
                    augmented_images, 
                    output_subdir, 
                    base_filename, 
                    batch_size=batch_size
                )
                
                # Update statistics if enabled
                if stats:
                    stats.update(aug_type, save_count, {
                        "batch_idx": batch_idx,
                        "img_idx": i,
                        "aug_type": aug_type
                    })
                
                total_augmented += save_count
                
                # Cleanup
                del augmented_images
                gc.collect()
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Print progress update
            if (batch_idx * batch_size + i + 1) % 10 == 0:
                print(f"Processed {batch_idx * batch_size + i + 1}/{len(image_files)} images, generated {total_augmented} augmented images")
    
    # Show preview if requested
    if args.preview and preview_images:
        print("Showing preview of augmented images...")
        show_images(
            preview_images[:10],  # Limit to 10 images
            titles=preview_titles[:10],
            cols=min(5, len(preview_images)),
            save_path=output_dir / "augmentation_preview.png"
        )
    
    # Save final statistics
    if stats:
        stats.save()
    
    print(f"Augmentation complete! Generated {total_augmented} images from {len(image_files)} original images.")
    print(f"Output saved to: {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())