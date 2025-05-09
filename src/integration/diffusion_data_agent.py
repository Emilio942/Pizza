#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diffusion Model Data Reliability Agent

This agent automates and improves the reliability of data generation, processing,
and quality assessment for diffusion models used in the pizza detection project.

It provides the following capabilities:
1. Batch generation control
2. Metadata tracking
3. Automated quality checking
4. Metric calculation
5. Data filtering
6. Resource management
7. Reporting and logging
"""

import os
import gc
import time
import json
import random
import argparse
import logging
import numpy as np
import torch
from PIL import Image, ImageStat
from datetime import datetime
from pathlib import Path
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm
import threading
import queue
import concurrent.futures

# Try importing optional dependencies
try:
    from cleanfid import fid
    HAS_CLEANFID = True
except ImportError:
    HAS_CLEANFID = False
    print("Warning: cleanfid not installed. FID metrics will be unavailable.")

# Configuration class
class AgentConfig:
    """Configuration for the Diffusion Data Agent"""
    
    def __init__(self, config_dict=None):
        # Default configuration
        self.batch_size = 16
        self.min_disk_space_gb = 5
        self.quality_threshold = 0.7
        self.image_size = (256, 256)
        self.save_format = 'jpg'
        self.save_quality = 90
        self.max_gpu_memory_usage = 0.9
        self.min_brightness = 20
        self.max_brightness = 240
        self.min_contrast = 0.1
        self.min_file_size_kb = 5
        self.blur_threshold = 50
        self.log_level = "INFO"
        self.metadata_schema = {
            "prompt": "",
            "seed": 0,
            "sampler": "",
            "guidance_scale": 0.0,
            "model_version": "",
            "generation_timestamp": "",
            "quality_score": 0.0,
            "batch_id": ""
        }
        
        # Override with provided configuration
        if config_dict:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    @classmethod
    def from_file(cls, filepath):
        """Load configuration from a JSON file"""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            return cls(config_dict)
        except Exception as e:
            print(f"Error loading configuration from {filepath}: {e}")
            return cls()
    
    def save(self, filepath):
        """Save configuration to a JSON file"""
        config_dict = {k: v for k, v in self.__dict__.items()}
        try:
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving configuration to {filepath}: {e}")
            return False


class DiffusionDataAgent:
    """Agent for managing diffusion model data reliability"""
    
    def __init__(self, config=None):
        """Initialize the agent with configuration"""
        self.config = config if config else AgentConfig()
        self._setup_logging()
        
        # Initialize counters and tracking
        self.stats = {
            "total_generated": 0,
            "total_filtered_out": 0,
            "batches_processed": 0,
            "quality_scores": [],
            "errors": [],
            "start_time": time.time()
        }
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Task queue for background processing
        self.task_queue = queue.Queue()
        self.processing_thread = None
        self.stop_processing = False
    
    def _setup_logging(self):
        """Set up logging for the agent"""
        log_level = getattr(logging, self.config.log_level)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("diffusion_agent.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("DiffusionDataAgent")
    
    def check_resources(self):
        """Check if sufficient resources are available"""
        # Check disk space
        disk_stats = os.statvfs('.')
        free_space_gb = (disk_stats.f_frsize * disk_stats.f_bavail) / (1024**3)
        if free_space_gb < self.config.min_disk_space_gb:
            self.logger.warning(f"Low disk space: {free_space_gb:.2f}GB remaining")
            return False
        
        # Check GPU memory if using CUDA
        if torch.cuda.is_available():
            used_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if used_memory > self.config.max_gpu_memory_usage:
                self.logger.warning(f"High GPU memory usage: {used_memory:.2%}")
                return False
        
        return True
    
    def generate_batch(self, generator_function, batch_id, batch_params, *args, **kwargs):
        """Generate a batch of images using the provided generator function"""
        if not self.check_resources():
            self.logger.error("Insufficient resources. Aborting batch generation.")
            return []
        
        self.logger.info(f"Starting batch generation: {batch_id}")
        try:
            # Call the generator function
            batch_images = generator_function(*args, **kwargs)
            
            # Record batch metadata
            batch_metadata = {
                "batch_id": batch_id,
                "timestamp": datetime.now().isoformat(),
                "parameters": batch_params,
                "count": len(batch_images) if isinstance(batch_images, list) else 1
            }
            
            self.stats["batches_processed"] += 1
            self.stats["total_generated"] += batch_metadata["count"]
            
            # Log batch completion
            self.logger.info(f"Completed batch {batch_id}: generated {batch_metadata['count']} images")
            
            # Save batch metadata
            self._save_batch_metadata(batch_id, batch_metadata)
            
            return batch_images
            
        except Exception as e:
            self.logger.error(f"Error generating batch {batch_id}: {str(e)}")
            self.stats["errors"].append({"batch_id": batch_id, "error": str(e), "timestamp": datetime.now().isoformat()})
            return []
    
    def _save_batch_metadata(self, batch_id, metadata):
        """Save batch metadata to disk"""
        os.makedirs("metadata", exist_ok=True)
        with open(f"metadata/batch_{batch_id}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_image_metadata(self, image_path, metadata):
        """Save image metadata alongside image"""
        metadata_path = os.path.splitext(image_path)[0] + ".json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def check_image_quality(self, img, metadata=None):
        """
        Perform basic quality checks on an image
        Returns (is_good, score, reasons)
        """
        reasons = []
        
        # Convert to PIL if tensor
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        
        # Check image dimensions
        if img.width < 32 or img.height < 32:
            reasons.append("too_small")
        
        # Check if image is completely black or white
        stats = ImageStat.Stat(img)
        mean_brightness = sum(stats.mean) / len(stats.mean)
        if mean_brightness < self.config.min_brightness:
            reasons.append("too_dark")
        if mean_brightness > self.config.max_brightness:
            reasons.append("too_bright")
        
        # Check contrast
        if stats.stddev and len(stats.stddev) > 0:
            contrast = sum(stats.stddev) / len(stats.stddev)
            if contrast < self.config.min_contrast * 255:
                reasons.append("low_contrast")
        
        # Calculate quality score (higher is better)
        # Simple formula - can be replaced with more sophisticated model
        quality_score = 1.0
        if reasons:
            # Each issue reduces score
            quality_score -= len(reasons) * 0.3
            quality_score = max(0.0, quality_score)
        
        is_good = quality_score >= self.config.quality_threshold
        
        # Update metadata if provided
        if metadata is not None:
            metadata["quality_score"] = quality_score
            metadata["quality_issues"] = reasons
        
        return is_good, quality_score, reasons
    
    def filter_batch(self, images, metadata_list=None):
        """
        Filter a batch of images based on quality checks
        Returns (good_images, good_metadata, filtered_count)
        """
        good_images = []
        good_metadata = [] if metadata_list else None
        filtered_count = 0
        
        for i, img in enumerate(images):
            metadata = metadata_list[i] if metadata_list else None
            is_good, score, reasons = self.check_image_quality(img, metadata)
            
            if is_good:
                good_images.append(img)
                if metadata_list:
                    good_metadata.append(metadata)
            else:
                filtered_count += 1
                self.logger.debug(f"Filtered image {i} with score {score:.2f}, issues: {', '.join(reasons)}")
        
        self.stats["total_filtered_out"] += filtered_count
        self.logger.info(f"Filtered {filtered_count} images from batch")
        
        return good_images, good_metadata, filtered_count
    
    def process_and_save_image(self, img, output_dir, filename, metadata=None):
        """Process and save an individual image with its metadata"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to PIL if it's a tensor
        if isinstance(img, torch.Tensor):
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = TF.to_pil_image(img)
            else:
                self.logger.error(f"Invalid tensor shape for image: {img.shape}")
                return False
        
        # Check quality
        is_good, score, reasons = self.check_image_quality(img, metadata)
        
        # Determine target directory based on quality
        target_dir = os.path.join(output_dir, "good" if is_good else "filtered")
        os.makedirs(target_dir, exist_ok=True)
        
        # Save image
        save_path = os.path.join(target_dir, filename)
        try:
            img.save(save_path, format=self.config.save_format.upper(), quality=self.config.save_quality)
            
            # Save metadata if provided
            if metadata:
                self._save_image_metadata(save_path, metadata)
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving image {filename}: {str(e)}")
            return False
    
    def batch_save_images(self, images, output_dir, batch_id, metadata_list=None):
        """Save a batch of images with sequential naming"""
        if not images:
            return 0
        
        os.makedirs(output_dir, exist_ok=True)
        saved_count = 0
        
        for i, img in enumerate(images):
            metadata = metadata_list[i] if metadata_list else None
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"diffusion_{batch_id}_{timestamp}_{i:04d}.{self.config.save_format}"
            
            if self.process_and_save_image(img, output_dir, filename, metadata):
                saved_count += 1
        
        self.logger.info(f"Saved {saved_count} images from batch {batch_id}")
        return saved_count
    
    def calculate_batch_metrics(self, images, batch_id):
        """Calculate quality metrics for a batch of images"""
        if not images:
            return {}
        
        metrics = {
            "batch_id": batch_id,
            "image_count": len(images),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Calculate basic statistics
        quality_scores = []
        brightness_values = []
        
        for img in images:
            # Convert to PIL if it's a tensor
            if isinstance(img, torch.Tensor):
                img = TF.to_pil_image(img)
            
            # Get quality score
            _, score, _ = self.check_image_quality(img)
            quality_scores.append(score)
            
            # Get brightness
            stats = ImageStat.Stat(img)
            brightness = sum(stats.mean) / len(stats.mean)
            brightness_values.append(brightness)
        
        # Add statistics to metrics
        metrics["quality_mean"] = np.mean(quality_scores) if quality_scores else 0
        metrics["quality_std"] = np.std(quality_scores) if quality_scores else 0
        metrics["quality_min"] = min(quality_scores) if quality_scores else 0
        metrics["quality_max"] = max(quality_scores) if quality_scores else 0
        metrics["brightness_mean"] = np.mean(brightness_values) if brightness_values else 0
        
        # Calculate FID if cleanfid is available
        if HAS_CLEANFID and hasattr(self, 'reference_dir') and self.reference_dir:
            try:
                # Save batch to temporary directory
                temp_dir = os.path.join(output_dir, f"temp_fid_{batch_id}")
                os.makedirs(temp_dir, exist_ok=True)
                
                for i, img in enumerate(images):
                    if isinstance(img, torch.Tensor):
                        img = TF.to_pil_image(img)
                    img.save(os.path.join(temp_dir, f"temp_{i:04d}.png"))
                
                # Calculate FID
                metrics["fid"] = fid.compute_fid(self.reference_dir, temp_dir)
                
                # Clean up
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)
                
            except Exception as e:
                self.logger.error(f"Error calculating FID: {str(e)}")
        
        # Save metrics
        self._save_batch_metrics(batch_id, metrics)
        
        return metrics
    
    def _save_batch_metrics(self, batch_id, metrics):
        """Save batch metrics to disk"""
        os.makedirs("metrics", exist_ok=True)
        with open(f"metrics/batch_{batch_id}_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def start_background_processing(self):
        """Start background processing thread"""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.stop_processing = False
            self.processing_thread = threading.Thread(target=self._background_processor)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            self.logger.info("Started background processing thread")
    
    def stop_background_processing(self):
        """Stop background processing thread"""
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_processing = True
            self.processing_thread.join(timeout=10)
            self.logger.info("Stopped background processing thread")
    
    def _background_processor(self):
        """Background thread to process tasks from queue"""
        while not self.stop_processing:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:
                    continue
                
                task_type, args, kwargs = task
                
                if task_type == "filter":
                    images, metadata, output_dir, batch_id = args
                    good_images, good_metadata, _ = self.filter_batch(images, metadata)
                    self.batch_save_images(good_images, output_dir, batch_id, good_metadata)
                
                elif task_type == "calculate_metrics":
                    images, batch_id = args
                    self.calculate_batch_metrics(images, batch_id)
                
                self.task_queue.task_done()
                
            except queue.Empty:
                pass
            except Exception as e:
                self.logger.error(f"Error in background processing: {str(e)}")
    
    def queue_batch_processing(self, images, output_dir, batch_id, metadata_list=None):
        """Queue a batch for background processing (filtering and saving)"""
        self.task_queue.put(("filter", (images, metadata_list, output_dir, batch_id), {}))
    
    def queue_metrics_calculation(self, images, batch_id):
        """Queue metrics calculation for a batch"""
        self.task_queue.put(("calculate_metrics", (images, batch_id), {}))
    
    def prepare_for_downstream(self, images, target_size=None, normalize=False):
        """Prepare filtered images for downstream tasks"""
        if not images:
            return []
        
        prepared_images = []
        target_size = target_size or self.config.image_size
        
        transform_list = []
        if target_size:
            transform_list.append(transforms.Resize(target_size, antialias=True))
        
        if normalize:
            transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                     std=[0.229, 0.224, 0.225]))
        
        if transform_list:
            processor = transforms.Compose(transform_list)
            
            for img in images:
                if not isinstance(img, torch.Tensor):
                    img = transforms.ToTensor()(img)
                prepared_images.append(processor(img))
        else:
            # Just ensure they're all tensors
            for img in images:
                if not isinstance(img, torch.Tensor):
                    prepared_images.append(transforms.ToTensor()(img))
                else:
                    prepared_images.append(img)
        
        return prepared_images
    
    def generate_report(self, output_file=None):
        """Generate a report on processing statistics"""
        # Calculate run time
        run_time = time.time() - self.stats["start_time"]
        run_time_str = f"{run_time // 3600:.0f}h {(run_time % 3600) // 60:.0f}m {run_time % 60:.0f}s"
        
        # Calculate pass rate
        total = self.stats["total_generated"]
        filtered = self.stats["total_filtered_out"]
        pass_rate = ((total - filtered) / total) * 100 if total > 0 else 0
        
        # Format report
        report = {
            "timestamp": datetime.now().isoformat(),
            "run_time": run_time_str,
            "total_images_generated": total,
            "images_filtered_out": filtered,
            "quality_pass_rate": f"{pass_rate:.2f}%",
            "batches_processed": self.stats["batches_processed"],
            "errors": len(self.stats["errors"]),
        }
        
        # Add quality score stats if available
        if self.stats["quality_scores"]:
            report["avg_quality_score"] = np.mean(self.stats["quality_scores"])
        
        # Print summary
        summary = "\n".join([f"{k}: {v}" for k, v in report.items()])
        self.logger.info(f"Generation Report:\n{summary}")
        
        # Save report if output file provided
        if output_file:
            try:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2)
                self.logger.info(f"Report saved to {output_file}")
            except Exception as e:
                self.logger.error(f"Error saving report: {str(e)}")
        
        return report
    
    def run_batch_job(self, generator_function, output_dir, total_count, batch_size=None, 
                     batch_params=None, metadata_template=None):
        """
        Run a complete batch job to generate a specified number of images
        
        Args:
            generator_function: Function that generates images
            output_dir: Directory to save outputs
            total_count: Total number of images to generate
            batch_size: Size of each batch (default: uses config batch_size)
            batch_params: Parameters to pass to generator function
            metadata_template: Template for image metadata
        """
        batch_size = batch_size or self.config.batch_size
        batch_params = batch_params or {}
        metadata_template = metadata_template or self.config.metadata_schema.copy()
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "good"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "filtered"), exist_ok=True)
        
        # Start background processing if not already running
        self.start_background_processing()
        
        # Calculate number of batches
        num_batches = (total_count + batch_size - 1) // batch_size
        
        # Generate batches
        total_generated = 0
        batch_time_start = time.time()
        
        for batch_idx in range(num_batches):
            batch_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{batch_idx}"
            
            # Adjust last batch size if needed
            current_batch_size = min(batch_size, total_count - total_generated)
            if current_batch_size <= 0:
                break
            
            # Update batch parameters
            batch_params_copy = batch_params.copy()
            batch_params_copy["batch_size"] = current_batch_size
            
            # Generate batch
            self.logger.info(f"Generating batch {batch_idx+1}/{num_batches} with size {current_batch_size}")
            images = self.generate_batch(generator_function, batch_id, batch_params_copy, **batch_params_copy)
            
            if not images:
                self.logger.warning(f"Batch {batch_id} produced no images, continuing...")
                continue
            
            # Prepare metadata for each image
            metadata_list = []
            for i in range(len(images)):
                img_metadata = metadata_template.copy()
                img_metadata["batch_id"] = batch_id
                img_metadata["generation_timestamp"] = datetime.now().isoformat()
                # Additional metadata could be added here (e.g., from generator)
                metadata_list.append(img_metadata)
            
            # Queue batch for processing
            self.queue_batch_processing(images, output_dir, batch_id, metadata_list)
            
            # Also queue metrics calculation
            self.queue_metrics_calculation(images, batch_id)
            
            # Update counters
            total_generated += len(images)
            
            # Free memory
            del images
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Log progress
            if batch_idx > 0 and batch_idx % 5 == 0:
                elapsed = time.time() - batch_time_start
                images_per_sec = total_generated / elapsed if elapsed > 0 else 0
                est_remaining = (total_count - total_generated) / images_per_sec if images_per_sec > 0 else 0
                self.logger.info(f"Progress: {total_generated}/{total_count} images "
                               f"({images_per_sec:.2f} img/s, est. {est_remaining/60:.1f}m remaining)")
        
        # Wait for all tasks to complete
        self.task_queue.join()
        
        # Generate report
        report_file = os.path.join(output_dir, "generation_report.json")
        report = self.generate_report(report_file)
        
        # Return summary
        return report

# Helper functions for image quality assessment

def detect_blur(img, threshold=100):
    """
    Detect if an image is blurry using the Laplacian operator
    Returns True if image is blurry, False otherwise
    """
    if isinstance(img, torch.Tensor):
        img = TF.to_pil_image(img)
    
    try:
        import cv2
        import numpy as np
        
        # Convert to grayscale and then to numpy array
        img_gray = img.convert('L')
        img_np = np.array(img_gray)
        
        # Calculate variance of Laplacian (measure of focus)
        laplacian_var = cv2.Laplacian(img_np, cv2.CV_64F).var()
        
        # Return True if the variance is below threshold (blurry)
        return laplacian_var < threshold, laplacian_var
        
    except ImportError:
        # Fallback if OpenCV not available
        return False, 0

def check_image_size(filepath, min_size_kb=5):
    """Check if an image file meets minimum size requirements"""
    try:
        file_size_kb = os.path.getsize(filepath) / 1024
        return file_size_kb >= min_size_kb
    except Exception:
        return False

# Main function
def main():
    parser = argparse.ArgumentParser(description='Diffusion Model Data Reliability Agent')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='./diffusion_output', help='Output directory')
    parser.add_argument('--total', type=int, default=100, help='Total number of images to generate')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--report', type=str, help='Path to save report')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        config = AgentConfig.from_file(args.config)
    else:
        config = AgentConfig()
    
    # Override config with command line arguments
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # Create agent
    agent = DiffusionDataAgent(config)
    
    # Example usage (this would be replaced with actual diffusion model code)
    def dummy_generator(batch_size, **kwargs):
        """Dummy function that simulates a diffusion model generator"""
        print(f"Generating batch of {batch_size} images...")
        # Simulate generation delay
        time.sleep(2)
        # Return random tensors as fake images
        return [torch.rand(3, 256, 256) for _ in range(batch_size)]
    
    # Run batch job
    agent.run_batch_job(
        generator_function=dummy_generator,
        output_dir=args.output_dir,
        total_count=args.total,
        batch_params={"additional_param": "value"}
    )
    
    # Generate final report
    report_file = args.report if args.report else os.path.join(args.output_dir, "final_report.json")
    agent.generate_report(report_file)
    
    # Stop background processing
    agent.stop_background_processing()

if __name__ == "__main__":
    main()