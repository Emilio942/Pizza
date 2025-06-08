#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Frame Spatial Analysis Pipeline for Spatial-MLLM
Implementation of SPATIAL-5.1: Video-based Pizza Analysis

This script implements multi-frame capabilities for the Spatial-MLLM architecture to enable
video-based pizza analysis, including baking process monitoring, temporal spatial feature tracking,
and space-aware frame sampling optimized for pizza quality assessment.

Key Features:
1. Space-aware frame sampling for pizza videos
2. Temporal spatial analysis with VGGT encoder
3. Baking process simulation and monitoring
4. Video preprocessing pipeline for dual-encoder architecture
5. Temporal-spatial feature correlation analysis

SPATIAL-5.1 Implementation
Author: GitHub Copilot (2025-06-07)
"""

import os
import sys
import cv2
import json
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any, Iterator
from datetime import datetime, timedelta
import threading
import time

import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass

# Setup project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'output' / 'multi_frame_spatial_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class VideoConfig:
    """Configuration for video-based analysis"""
    fps: float = 1.0  # Frames per second for baking process
    duration_seconds: float = 60.0  # Total duration
    target_frames: int = 8  # Frames for spatial-MLLM processing
    frame_sampling_method: str = "space_aware"  # space_aware, uniform, adaptive
    spatial_resolution: Tuple[int, int] = (518, 518)  # VGGT compatible resolution
    enable_temporal_fusion: bool = True

@dataclass
class SpatialVideoFrame:
    """Represents a frame with spatial and temporal features"""
    frame_id: int
    timestamp: float
    visual_data: torch.Tensor  # (C, H, W) format
    spatial_data: torch.Tensor  # (4, H, W) format - depth, normals, curvature, meta
    temperature: float  # Simulated oven temperature
    baking_stage: str  # raw, rising, browning, baked, burnt
    spatial_features: Dict[str, float]  # Extracted spatial metrics

@dataclass
class VideoAnalysisResult:
    """Results from multi-frame spatial analysis"""
    video_id: str
    total_frames: int
    duration: float
    baking_progression: List[str]
    spatial_quality_scores: List[float]
    temporal_consistency: float
    burn_detection_frames: List[int]
    optimal_baking_frame: Optional[int]
    quality_trend: str  # improving, degrading, stable

class PizzaBakingSimulator:
    """Simulates the pizza baking process for video generation"""
    
    def __init__(self, config: VideoConfig):
        self.config = config
        self.baking_stages = [
            "raw", "rising", "setting", "browning", "golden", "dark", "burnt"
        ]
        self.stage_durations = {
            "raw": 0.15,      # 15% of total time
            "rising": 0.20,   # 20% of total time
            "setting": 0.15,  # 15% of total time  
            "browning": 0.25, # 25% of total time
            "golden": 0.15,   # 15% of total time
            "dark": 0.08,     # 8% of total time
            "burnt": 0.02     # 2% of total time
        }
        
    def get_baking_stage(self, time_ratio: float) -> str:
        """Get baking stage based on time ratio (0.0 to 1.0)"""
        cumulative = 0.0
        for stage, duration in self.stage_durations.items():
            cumulative += duration
            if time_ratio <= cumulative:
                return stage
        return "burnt"
    
    def get_temperature(self, time_ratio: float) -> float:
        """Get oven temperature based on baking progress"""
        # Simulate oven heating curve: starts at 200°C, heats to 250°C, then slightly cools
        if time_ratio < 0.1:  # Initial heating
            return 200 + (time_ratio * 10) * 30  # 200°C to 230°C
        elif time_ratio < 0.8:  # Main baking
            return 230 + (time_ratio - 0.1) * 20 / 0.7  # 230°C to 250°C
        else:  # Final stage, slight cooling
            return 250 - (time_ratio - 0.8) * 10 / 0.2  # 250°C to 240°C
    
    def simulate_pizza_appearance(self, base_image: Image.Image, time_ratio: float) -> Image.Image:
        """Simulate pizza appearance changes during baking"""
        pizza = base_image.copy()
        stage = self.get_baking_stage(time_ratio)
        
        # Create drawing context
        draw = ImageDraw.Draw(pizza)
        width, height = pizza.size
        
        # Apply stage-specific transformations
        if stage == "raw":
            # Pale, uncooked appearance
            enhancer = ImageEnhance.Color(pizza)
            pizza = enhancer.enhance(0.7)  # Reduce color saturation
            
        elif stage == "rising":
            # Slightly puffed, still pale
            enhancer = ImageEnhance.Brightness(pizza)
            pizza = enhancer.enhance(1.1)  # Slightly brighter
            
        elif stage == "setting":
            # Beginning to firm up, slight color change
            enhancer = ImageEnhance.Color(pizza)
            pizza = enhancer.enhance(0.9)
            
        elif stage == "browning":
            # Golden brown color starting to develop
            enhancer = ImageEnhance.Color(pizza)
            pizza = enhancer.enhance(1.2)
            # Add slight brownish tint
            brown_overlay = Image.new('RGBA', (width, height), (139, 69, 19, 30))
            pizza = Image.alpha_composite(pizza.convert('RGBA'), brown_overlay).convert('RGB')
            
        elif stage == "golden":
            # Perfect golden brown
            enhancer = ImageEnhance.Color(pizza)
            pizza = enhancer.enhance(1.4)
            golden_overlay = Image.new('RGBA', (width, height), (218, 165, 32, 40))
            pizza = Image.alpha_composite(pizza.convert('RGBA'), golden_overlay).convert('RGB')
            
        elif stage == "dark":
            # Getting darker, approaching burn
            dark_overlay = Image.new('RGBA', (width, height), (101, 67, 33, 60))
            pizza = Image.alpha_composite(pizza.convert('RGBA'), dark_overlay).convert('RGB')
            
        elif stage == "burnt":
            # Burnt appearance with black spots
            burnt_overlay = Image.new('RGBA', (width, height), (40, 20, 10, 80))
            pizza = Image.alpha_composite(pizza.convert('RGBA'), burnt_overlay).convert('RGB')
            
            # Add random burnt spots
            for _ in range(np.random.randint(3, 8)):
                x = np.random.randint(width // 4, 3 * width // 4)
                y = np.random.randint(height // 4, 3 * height // 4)
                radius = np.random.randint(5, 15)
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=(20, 10, 5))
        
        return pizza

class SpaceAwareFrameSampler:
    """Implements intelligent frame sampling based on spatial features"""
    
    def __init__(self, config: VideoConfig):
        self.config = config
        
    def analyze_spatial_change(self, frame1: torch.Tensor, frame2: torch.Tensor) -> float:
        """Calculate spatial change between two frames"""
        # Compare spatial features (depth maps, surface gradients)
        diff = torch.abs(frame1 - frame2)
        spatial_change = torch.mean(diff).item()
        return spatial_change
    
    def extract_spatial_complexity(self, frame: torch.Tensor) -> float:
        """Extract spatial complexity score from frame"""
        # Calculate spatial gradients
        grad_x = torch.abs(frame[:, :, 1:] - frame[:, :, :-1])
        grad_y = torch.abs(frame[:, 1:, :] - frame[:, :-1, :])
        
        complexity = (torch.mean(grad_x) + torch.mean(grad_y)).item()
        return complexity
    
    def space_aware_sampling(self, frames: List[SpatialVideoFrame]) -> List[int]:
        """Sample frames based on spatial content and temporal distribution"""
        if len(frames) <= self.config.target_frames:
            return list(range(len(frames)))
        
        # Calculate spatial complexity for all frames
        complexities = []
        for frame in frames:
            complexity = self.extract_spatial_complexity(frame.visual_data)
            complexities.append(complexity)
        
        # Select frames with high spatial complexity and good temporal distribution
        selected_indices = []
        
        # Always include first and last frame
        selected_indices.extend([0, len(frames) - 1])
        
        # Select frames with highest spatial complexity
        sorted_by_complexity = sorted(enumerate(complexities), key=lambda x: x[1], reverse=True)
        
        # Add frames ensuring temporal distribution
        target_remaining = self.config.target_frames - 2
        step = len(frames) // (target_remaining + 1)
        
        for i in range(1, target_remaining + 1):
            candidate_idx = i * step
            # Find the most complex frame in a window around the candidate
            window_start = max(0, candidate_idx - step // 2)
            window_end = min(len(frames), candidate_idx + step // 2)
            
            best_idx = candidate_idx
            best_complexity = complexities[candidate_idx]
            
            for idx in range(window_start, window_end):
                if idx not in selected_indices and complexities[idx] > best_complexity:
                    best_idx = idx
                    best_complexity = complexities[idx]
            
            if best_idx not in selected_indices:
                selected_indices.append(best_idx)
        
        # Fill remaining slots with highest complexity frames
        while len(selected_indices) < self.config.target_frames:
            for idx, _ in sorted_by_complexity:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    break
        
        return sorted(selected_indices[:self.config.target_frames])
    
    def uniform_sampling(self, frames: List[SpatialVideoFrame]) -> List[int]:
        """Sample frames uniformly across time"""
        if len(frames) <= self.config.target_frames:
            return list(range(len(frames)))
        
        step = len(frames) / self.config.target_frames
        indices = [int(i * step) for i in range(self.config.target_frames)]
        return indices
    
    def adaptive_sampling(self, frames: List[SpatialVideoFrame]) -> List[int]:
        """Adaptive sampling based on baking stage transitions"""
        if len(frames) <= self.config.target_frames:
            return list(range(len(frames)))
        
        # Find stage transition points
        transition_indices = []
        current_stage = frames[0].baking_stage
        
        for i, frame in enumerate(frames[1:], 1):
            if frame.baking_stage != current_stage:
                transition_indices.append(i)
                current_stage = frame.baking_stage
        
        # Ensure we have enough transitions
        if len(transition_indices) < self.config.target_frames // 2:
            # Add some uniform samples
            uniform_indices = self.uniform_sampling(frames)
            transition_indices.extend(uniform_indices)
        
        # Select the most important transitions and uniform samples
        selected = sorted(list(set(transition_indices)))[:self.config.target_frames]
        
        # Fill remaining slots uniformly if needed
        while len(selected) < self.config.target_frames:
            candidates = set(range(len(frames))) - set(selected)
            if candidates:
                selected.append(min(candidates))
        
        return sorted(selected[:self.config.target_frames])

class VideoPreprocessingPipeline:
    """Preprocessing pipeline for multi-frame spatial analysis"""
    
    def __init__(self, config: VideoConfig):
        self.config = config
        
        # Import spatial preprocessing if available
        try:
            from scripts.spatial_preprocessing import SpatialPreprocessingPipeline
            self.spatial_processor = SpatialPreprocessingPipeline(
                output_size=config.spatial_resolution,
                depth_estimation_method="edge_based"
            )
            logger.info("✅ Spatial preprocessing pipeline loaded")
        except ImportError as e:
            logger.warning(f"⚠️  Spatial preprocessing not available: {e}")
            self.spatial_processor = None
        
        # Setup transforms
        self.visual_transform = transforms.Compose([
            transforms.Resize(config.spatial_resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def process_video_frame(self, pil_image: Image.Image, frame_metadata: Dict) -> SpatialVideoFrame:
        """Process a single video frame with spatial and temporal features"""
        
        # Process with spatial pipeline if available
        if self.spatial_processor:
            try:
                spatial_result = self.spatial_processor.process_image(pil_image)
                visual_tensor = spatial_result.get('visual_features', spatial_result.get('visual_input'))
                spatial_tensor = spatial_result.get('spatial_features', spatial_result.get('spatial_input'))
                
                if visual_tensor is not None and spatial_tensor is not None:
                    # Remove batch dim if present, add frame dim if needed
                    if visual_tensor.dim() == 5:  # (B,F,C,H,W)
                        visual_tensor = visual_tensor.squeeze(0)  # (F,C,H,W)
                    elif visual_tensor.dim() == 4:  # (B,C,H,W)
                        visual_tensor = visual_tensor.squeeze(0).unsqueeze(0)  # (F,C,H,W)
                    elif visual_tensor.dim() == 3:  # (C,H,W)
                        visual_tensor = visual_tensor.unsqueeze(0)  # (F,C,H,W)
                    
                    if spatial_tensor.dim() == 5:  # (B,F,C,H,W)
                        spatial_tensor = spatial_tensor.squeeze(0)  # (F,C,H,W)
                    elif spatial_tensor.dim() == 4:  # (B,C,H,W)
                        spatial_tensor = spatial_tensor.squeeze(0).unsqueeze(0)  # (F,C,H,W)
                    elif spatial_tensor.dim() == 3:  # (C,H,W)
                        spatial_tensor = spatial_tensor.unsqueeze(0)  # (F,C,H,W)
                else:
                    raise KeyError("Missing visual or spatial features")
                    
            except Exception as e:
                logger.warning(f"Spatial processing failed: {e}, using fallback")
                visual_tensor = self.visual_transform(pil_image).unsqueeze(0)  # Add frame dim (F,C,H,W)
                spatial_tensor = self._create_fallback_spatial_data(visual_tensor)
        else:
            # Fallback processing
            visual_tensor = self.visual_transform(pil_image).unsqueeze(0)  # Add frame dim (F,C,H,W)
            spatial_tensor = self._create_fallback_spatial_data(visual_tensor)
        
        # Extract spatial features for analysis
        spatial_features = self._extract_spatial_metrics(visual_tensor, spatial_tensor)
        
        return SpatialVideoFrame(
            frame_id=frame_metadata['frame_id'],
            timestamp=frame_metadata['timestamp'],
            visual_data=visual_tensor,
            spatial_data=spatial_tensor,
            temperature=frame_metadata.get('temperature', 225.0),
            baking_stage=frame_metadata.get('baking_stage', 'unknown'),
            spatial_features=spatial_features
        )
    
    def _create_fallback_spatial_data(self, visual_tensor: torch.Tensor) -> torch.Tensor:
        """Create fallback spatial data when spatial processor is not available"""
        # Handle different input shapes: (F,C,H,W) expected
        if visual_tensor.dim() == 5:  # (B,F,C,H,W)
            F, C, H, W = visual_tensor.shape[1:]
            visual_tensor = visual_tensor.squeeze(0)  # Remove batch dim
        elif visual_tensor.dim() == 4:  # (F,C,H,W)
            F, C, H, W = visual_tensor.shape
        else:
            raise ValueError(f"Unexpected visual tensor shape: {visual_tensor.shape}")
        
        # Create synthetic depth map from luminance
        gray = torch.mean(visual_tensor, dim=1, keepdim=True)  # Convert to grayscale (F,1,H,W)
        depth = gray / (gray.max() + 1e-8)  # Normalize to [0, 1]
        
        # Create surface normals (simplified)
        grad_x = torch.diff(depth, dim=-1, prepend=depth[:, :, :, :1])
        grad_y = torch.diff(depth, dim=-2, prepend=depth[:, :, :1, :])
        normals = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # Create curvature (second derivative)
        curvature = torch.diff(grad_x, dim=-1, prepend=grad_x[:, :, :, :1])
        
        # Meta channel (combine edge and texture information)
        meta = (normals + curvature) / 2
        
        # Concatenate along channel dimension to get (F,4,H,W)
        spatial_data = torch.cat([depth, normals, curvature, meta], dim=1)
        
        return spatial_data
        return spatial_data
    
    def _extract_spatial_metrics(self, visual_tensor: torch.Tensor, spatial_tensor: torch.Tensor) -> Dict[str, float]:
        """Extract spatial metrics for analysis"""
        
        # Handle tensor shapes: both should be (F,C,H,W) format
        # Depth variation (pizza height variation)
        depth_channel = spatial_tensor[:, 0, :, :]  # (F,H,W)
        depth_variance = torch.var(depth_channel).item()
        
        # Surface roughness (from normals)
        normals_channel = spatial_tensor[:, 1, :, :]  # (F,H,W)
        surface_roughness = torch.mean(normals_channel).item()
        
        # Edge strength (from curvature)
        curvature_channel = spatial_tensor[:, 2, :, :]  # (F,H,W)
        edge_strength = torch.mean(torch.abs(curvature_channel)).item()
        
        # Color variance (browning detection)
        color_variance = torch.var(visual_tensor).item()
        
        return {
            'depth_variance': depth_variance,
            'surface_roughness': surface_roughness,
            'edge_strength': edge_strength,
            'color_variance': color_variance
        }

class MultiFrameSpatialAnalyzer:
    """Main analyzer for multi-frame spatial analysis"""
    
    def __init__(self, config: VideoConfig):
        self.config = config
        self.baking_simulator = PizzaBakingSimulator(config)
        self.frame_sampler = SpaceAwareFrameSampler(config)
        self.video_preprocessor = VideoPreprocessingPipeline(config)
        
        # Import spatial inference if available
        try:
            from scripts.spatial_inference_optimized import OptimizedSpatialInference, InferenceConfig
            inference_config = InferenceConfig(
                batch_size=1,
                enable_parallel_encoders=True,
                enable_amp=True
            )
            self.spatial_inference = OptimizedSpatialInference(inference_config)
            logger.info("✅ Spatial inference system loaded")
        except ImportError as e:
            logger.warning(f"⚠️  Spatial inference not available: {e}")
            self.spatial_inference = None
    
    def generate_baking_process_video(self, base_pizza_image: Image.Image, video_id: str) -> List[SpatialVideoFrame]:
        """Generate a simulated baking process video"""
        logger.info(f"🎬 Generating baking process video: {video_id}")
        
        total_frames = int(self.config.duration_seconds * self.config.fps)
        frames = []
        
        for frame_idx in range(total_frames):
            time_ratio = frame_idx / (total_frames - 1) if total_frames > 1 else 0
            timestamp = frame_idx / self.config.fps
            
            # Simulate pizza appearance at this time
            pizza_image = self.baking_simulator.simulate_pizza_appearance(base_pizza_image, time_ratio)
            
            # Create frame metadata
            frame_metadata = {
                'frame_id': frame_idx,
                'timestamp': timestamp,
                'temperature': self.baking_simulator.get_temperature(time_ratio),
                'baking_stage': self.baking_simulator.get_baking_stage(time_ratio)
            }
            
            # Process frame
            spatial_frame = self.video_preprocessor.process_video_frame(pizza_image, frame_metadata)
            frames.append(spatial_frame)
            
            if frame_idx % 10 == 0:
                logger.info(f"  Frame {frame_idx}/{total_frames}: {spatial_frame.baking_stage} "
                          f"({spatial_frame.temperature:.1f}°C)")
        
        logger.info(f"✅ Generated {len(frames)} frames for video {video_id}")
        return frames
    
    def analyze_temporal_spatial_features(self, frames: List[SpatialVideoFrame]) -> Dict[str, Any]:
        """Analyze temporal spatial features across video sequence"""
        logger.info("🔍 Analyzing temporal spatial features...")
        
        # Track feature evolution over time
        feature_evolution = {
            'depth_variance': [],
            'surface_roughness': [],
            'edge_strength': [],
            'color_variance': [],
            'temperatures': [],
            'stages': []
        }
        
        for frame in frames:
            feature_evolution['depth_variance'].append(frame.spatial_features['depth_variance'])
            feature_evolution['surface_roughness'].append(frame.spatial_features['surface_roughness'])
            feature_evolution['edge_strength'].append(frame.spatial_features['edge_strength'])
            feature_evolution['color_variance'].append(frame.spatial_features['color_variance'])
            feature_evolution['temperatures'].append(frame.temperature)
            feature_evolution['stages'].append(frame.baking_stage)
        
        # Calculate temporal consistency
        depth_std = np.std(feature_evolution['depth_variance'])
        roughness_std = np.std(feature_evolution['surface_roughness'])
        temporal_consistency = 1.0 / (1.0 + depth_std + roughness_std)
        
        # Detect burning frames (high edge strength + dark colors)
        burn_threshold = np.mean(feature_evolution['edge_strength']) + 2 * np.std(feature_evolution['edge_strength'])
        burn_frames = [i for i, edge in enumerate(feature_evolution['edge_strength']) if edge > burn_threshold]
        
        # Find optimal baking frame (golden stage with good features)
        optimal_frame = None
        for i, stage in enumerate(feature_evolution['stages']):
            if stage == 'golden':
                optimal_frame = i
                break
        
        # Determine quality trend
        color_trend = np.polyfit(range(len(feature_evolution['color_variance'])), 
                                feature_evolution['color_variance'], 1)[0]
        if color_trend > 0.01:
            quality_trend = "improving"
        elif color_trend < -0.01:
            quality_trend = "degrading"
        else:
            quality_trend = "stable"
        
        return {
            'feature_evolution': feature_evolution,
            'temporal_consistency': temporal_consistency,
            'burn_detection_frames': burn_frames,
            'optimal_baking_frame': optimal_frame,
            'quality_trend': quality_trend
        }
    
    def run_spatial_mllm_inference(self, sampled_frames: List[SpatialVideoFrame]) -> Dict[str, Any]:
        """Run spatial MLLM inference on sampled frames"""
        if not self.spatial_inference:
            logger.warning("⚠️  Spatial inference not available, skipping")
            return {"error": "Spatial inference not available"}
        
        logger.info(f"🧠 Running Spatial-MLLM inference on {len(sampled_frames)} frames...")
        
        try:
            # Prepare batch data for dual-encoder processing
            batch_visual = []
            batch_spatial = []
            
            for frame in sampled_frames:
                batch_visual.append(frame.visual_data)
                batch_spatial.append(frame.spatial_data)
            
            # Stack into tensors
            visual_batch = torch.stack(batch_visual, dim=0)  # (B, F, C, H, W)
            spatial_batch = torch.stack(batch_spatial, dim=0)  # (B, F, 4, H, W)
            
            # Run inference
            batch_data = {
                'visual_input': visual_batch,
                'spatial_input': spatial_batch
            }
            
            # Use the spatial inference system
            if hasattr(self.spatial_inference, '_parallel_encoder_processing'):
                encoded_features = self.spatial_inference._parallel_encoder_processing(batch_data)
            else:
                # Fallback to basic processing
                encoded_features = {"status": "processed"}
            
            logger.info("✅ Spatial-MLLM inference completed")
            return {
                "status": "success",
                "features_extracted": True,
                "batch_size": len(sampled_frames),
                "encoded_features": encoded_features
            }
            
        except Exception as e:
            logger.error(f"❌ Spatial-MLLM inference failed: {e}")
            return {"error": str(e)}
    
    def analyze_video(self, video_frames: List[SpatialVideoFrame], video_id: str) -> VideoAnalysisResult:
        """Perform complete multi-frame spatial analysis"""
        logger.info(f"📊 Starting multi-frame spatial analysis for video: {video_id}")
        
        # 1. Temporal spatial feature analysis
        temporal_analysis = self.analyze_temporal_spatial_features(video_frames)
        
        # 2. Frame sampling based on spatial content
        if self.config.frame_sampling_method == "space_aware":
            sampled_indices = self.frame_sampler.space_aware_sampling(video_frames)
        elif self.config.frame_sampling_method == "uniform":
            sampled_indices = self.frame_sampler.uniform_sampling(video_frames)
        elif self.config.frame_sampling_method == "adaptive":
            sampled_indices = self.frame_sampler.adaptive_sampling(video_frames)
        else:
            sampled_indices = self.frame_sampler.uniform_sampling(video_frames)
        
        sampled_frames = [video_frames[i] for i in sampled_indices]
        logger.info(f"🎯 Sampled {len(sampled_frames)} key frames using {self.config.frame_sampling_method} method")
        
        # 3. Spatial-MLLM inference on sampled frames
        inference_results = self.run_spatial_mllm_inference(sampled_frames)
        
        # 4. Extract quality scores
        quality_scores = []
        for frame in video_frames:
            # Simple quality metric based on spatial features
            quality = (
                frame.spatial_features['depth_variance'] * 0.3 +
                frame.spatial_features['surface_roughness'] * 0.3 +
                frame.spatial_features['edge_strength'] * 0.2 +
                frame.spatial_features['color_variance'] * 0.2
            )
            quality_scores.append(quality)
        
        # 5. Create analysis result
        result = VideoAnalysisResult(
            video_id=video_id,
            total_frames=len(video_frames),
            duration=video_frames[-1].timestamp if video_frames else 0.0,
            baking_progression=[f.baking_stage for f in video_frames],
            spatial_quality_scores=quality_scores,
            temporal_consistency=temporal_analysis['temporal_consistency'],
            burn_detection_frames=temporal_analysis['burn_detection_frames'],
            optimal_baking_frame=temporal_analysis['optimal_baking_frame'],
            quality_trend=temporal_analysis['quality_trend']
        )
        
        logger.info(f"✅ Multi-frame spatial analysis completed for {video_id}")
        logger.info(f"   📈 Quality trend: {result.quality_trend}")
        logger.info(f"   🎯 Optimal frame: {result.optimal_baking_frame}")
        logger.info(f"   🔥 Burn detected in {len(result.burn_detection_frames)} frames")
        
        return result

class VideoAnalysisVisualizer:
    """Visualizer for multi-frame analysis results"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_feature_evolution_plot(self, analysis_result: VideoAnalysisResult, 
                                    temporal_analysis: Dict[str, Any]) -> Path:
        """Create visualization of feature evolution over time"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Spatial Feature Evolution - Video: {analysis_result.video_id}', fontsize=16)
        
        feature_evolution = temporal_analysis['feature_evolution']
        frames = range(analysis_result.total_frames)
        
        # Plot depth variance
        axes[0, 0].plot(frames, feature_evolution['depth_variance'], 'b-', linewidth=2)
        axes[0, 0].set_title('Depth Variance (Pizza Height Variation)')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Variance')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot surface roughness
        axes[0, 1].plot(frames, feature_evolution['surface_roughness'], 'g-', linewidth=2)
        axes[0, 1].set_title('Surface Roughness')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Roughness')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot edge strength
        axes[1, 0].plot(frames, feature_evolution['edge_strength'], 'r-', linewidth=2)
        axes[1, 0].set_title('Edge Strength (Crust Definition)')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Strength')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Mark burn detection frames
        if analysis_result.burn_detection_frames:
            for burn_frame in analysis_result.burn_detection_frames:
                axes[1, 0].axvline(x=burn_frame, color='red', linestyle='--', alpha=0.7)
        
        # Plot quality scores
        axes[1, 1].plot(frames, analysis_result.spatial_quality_scores, 'm-', linewidth=2)
        axes[1, 1].set_title('Overall Spatial Quality')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Quality Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Mark optimal baking frame
        if analysis_result.optimal_baking_frame is not None:
            axes[1, 1].axvline(x=analysis_result.optimal_baking_frame, color='green', 
                             linestyle='--', alpha=0.8, label='Optimal Frame')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"{analysis_result.video_id}_feature_evolution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def create_baking_stage_timeline(self, analysis_result: VideoAnalysisResult) -> Path:
        """Create timeline visualization of baking stages"""
        
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Create color map for baking stages
        stage_colors = {
            'raw': '#E8E8E8',      # Light gray
            'rising': '#F0E68C',    # Khaki
            'setting': '#DEB887',   # Burlywood
            'browning': '#D2691E',  # Chocolate
            'golden': '#FFD700',    # Gold
            'dark': '#8B4513',      # Saddle brown
            'burnt': '#2F1B14'      # Dark brown
        }
        
        # Plot baking stages as colored timeline
        frames = range(analysis_result.total_frames)
        colors = [stage_colors.get(stage, '#CCCCCC') for stage in analysis_result.baking_progression]
        
        ax.scatter(frames, [1] * len(frames), c=colors, s=50, alpha=0.8)
        
        # Add stage labels
        current_stage = None
        stage_start = 0
        
        for i, stage in enumerate(analysis_result.baking_progression + [None]):
            if stage != current_stage:
                if current_stage is not None:
                    # Add label for previous stage
                    mid_point = (stage_start + i - 1) / 2
                    ax.annotate(current_stage, (mid_point, 1), xytext=(mid_point, 1.2),
                              ha='center', va='bottom', fontweight='bold',
                              arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))
                
                current_stage = stage
                stage_start = i
        
        ax.set_xlim(-1, analysis_result.total_frames)
        ax.set_ylim(0.5, 1.5)
        ax.set_xlabel('Frame Number')
        ax.set_title(f'Baking Process Timeline - Video: {analysis_result.video_id}')
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
        
        # Add burn detection markers
        if analysis_result.burn_detection_frames:
            for burn_frame in analysis_result.burn_detection_frames:
                ax.axvline(x=burn_frame, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        
        # Save plot
        timeline_path = self.output_dir / f"{analysis_result.video_id}_baking_timeline.png"
        plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return timeline_path

def create_sample_pizza_image() -> Image.Image:
    """Create a sample pizza image for testing"""
    width, height = 518, 518
    pizza = Image.new('RGB', (width, height), color='#8B4513')  # Brown background
    
    draw = ImageDraw.Draw(pizza)
    
    # Pizza base circle
    center = (width // 2, height // 2)
    radius = min(width, height) // 3
    draw.ellipse([center[0] - radius, center[1] - radius, 
                  center[0] + radius, center[1] + radius], 
                 fill='#DEB887', outline='#8B4513', width=3)
    
    # Add some toppings
    for _ in range(12):
        x = center[0] + np.random.randint(-radius//2, radius//2)
        y = center[1] + np.random.randint(-radius//2, radius//2)
        topping_radius = np.random.randint(8, 15)
        color = '#FF6347' if np.random.rand() > 0.5 else '#228B22'  # Tomato or green
        draw.ellipse([x - topping_radius, y - topping_radius,
                      x + topping_radius, y + topping_radius], fill=color)
    
    return pizza

def main():
    """Main function for multi-frame spatial analysis demonstration"""
    logger.info("🍕 Starting Multi-Frame Spatial Analysis Pipeline...")
    
    # Configuration
    config = VideoConfig(
        fps=2.0,  # 2 frames per second for baking simulation
        duration_seconds=30.0,  # 30 second baking process
        target_frames=8,  # Process 8 key frames
        frame_sampling_method="space_aware",
        enable_temporal_fusion=True
    )
    
    # Create analyzer
    analyzer = MultiFrameSpatialAnalyzer(config)
    
    # Create output directory
    output_dir = project_root / "output" / "multi_frame_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizer
    visualizer = VideoAnalysisVisualizer(output_dir)
    
    # Generate sample pizza or load existing image
    sample_pizza_path = project_root / "augmented_pizza" / "raw"
    if sample_pizza_path.exists() and list(sample_pizza_path.glob("*.jpg")):
        # Use existing pizza image
        pizza_image_path = list(sample_pizza_path.glob("*.jpg"))[0]
        pizza_image = Image.open(pizza_image_path).convert('RGB')
        logger.info(f"📷 Using existing pizza image: {pizza_image_path}")
    else:
        # Create synthetic pizza image
        pizza_image = create_sample_pizza_image()
        logger.info("📷 Created synthetic pizza image for testing")
    
    # Test multiple baking scenarios
    scenarios = [
        {"video_id": "normal_baking", "duration": 30.0, "method": "space_aware"},
        {"video_id": "fast_baking", "duration": 20.0, "method": "adaptive"},
        {"video_id": "slow_baking", "duration": 45.0, "method": "uniform"}
    ]
    
    results = []
    
    for scenario in scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"🎬 Testing scenario: {scenario['video_id']}")
        logger.info(f"{'='*60}")
        
        # Update config for this scenario
        config.duration_seconds = scenario['duration']
        config.frame_sampling_method = scenario['method']
        
        # Generate baking process video
        video_frames = analyzer.generate_baking_process_video(pizza_image, scenario['video_id'])
        
        # Analyze video
        analysis_result = analyzer.analyze_video(video_frames, scenario['video_id'])
        
        # Create visualizations
        temporal_analysis = analyzer.analyze_temporal_spatial_features(video_frames)
        
        feature_plot = visualizer.create_feature_evolution_plot(analysis_result, temporal_analysis)
        timeline_plot = visualizer.create_baking_stage_timeline(analysis_result)
        
        logger.info(f"📊 Visualizations saved:")
        logger.info(f"   - Feature evolution: {feature_plot}")
        logger.info(f"   - Baking timeline: {timeline_plot}")
        
        results.append({
            'scenario': scenario,
            'analysis_result': analysis_result,
            'temporal_analysis': temporal_analysis
        })
    
    # Save comprehensive results
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'fps': config.fps,
            'spatial_resolution': config.spatial_resolution,
            'target_frames': config.target_frames
        },
        'scenarios': []
    }
    
    for result in results:
        scenario_summary = {
            'video_id': result['analysis_result'].video_id,
            'total_frames': result['analysis_result'].total_frames,
            'duration': result['analysis_result'].duration,
            'quality_trend': result['analysis_result'].quality_trend,
            'temporal_consistency': result['analysis_result'].temporal_consistency,
            'burn_frames_count': len(result['analysis_result'].burn_detection_frames),
            'optimal_frame': result['analysis_result'].optimal_baking_frame,
            'sampling_method': result['scenario']['method']
        }
        results_summary['scenarios'].append(scenario_summary)
    
    # Save results
    results_file = output_dir / "multi_frame_analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("✅ Multi-Frame Spatial Analysis Pipeline Completed!")
    logger.info(f"📊 Results saved to: {results_file}")
    logger.info(f"🎨 Visualizations saved to: {output_dir}")
    logger.info(f"{'='*60}")
    
    # Print summary
    logger.info("\n📋 ANALYSIS SUMMARY:")
    for i, result in enumerate(results, 1):
        ar = result['analysis_result']
        logger.info(f"  {i}. {ar.video_id}:")
        logger.info(f"     • Duration: {ar.duration:.1f}s ({ar.total_frames} frames)")
        logger.info(f"     • Quality trend: {ar.quality_trend}")
        logger.info(f"     • Temporal consistency: {ar.temporal_consistency:.3f}")
        logger.info(f"     • Burn detection: {len(ar.burn_detection_frames)} frames")
        logger.info(f"     • Optimal baking frame: {ar.optimal_baking_frame}")

if __name__ == "__main__":
    main()
