import os
import sys
import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
from PIL import Image
import logging
from pathlib import Path
import random
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pizza3DGenerator:
    """
    Generates synthetic 3D pizza images for training.
    Uses trimesh and pyrender to create 3D scenes with pizza models.
    """
    
    def __init__(self, output_dir="data/synthetic_3d", width=320, height=240):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        
        # Initialize renderer
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=width/height)
        self.renderer = pyrender.OffscreenRenderer(width, height)
        
        # Load textures
        self.textures = self._load_textures()
        
    def _load_textures(self):
        """Load pizza textures from existing 2D data."""
        texture_paths = []
        data_dir = Path("data")
        
        # Look for images in data directory
        if data_dir.exists():
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                texture_paths.extend(list(data_dir.rglob(ext)))
        
        if not texture_paths:
            logger.warning("No textures found in data directory. Using random colors.")
            return []
            
        logger.info(f"Found {len(texture_paths)} potential textures")
        return texture_paths

    def create_pizza_mesh(self, radius=1.0, height=0.1):
        """Create a simple 3D mesh representing a pizza."""
        # Create a cylinder for the pizza base
        pizza = trimesh.creation.cylinder(radius=radius, height=height)
        
        # Rotate to lie flat
        # trimesh cylinder is along Z axis, we want it flat on X-Z plane (Y up)
        # But pyrender Y is up.
        
        # Apply texture if available
        if self.textures:
            try:
                texture_path = random.choice(self.textures)
                im = Image.open(texture_path).convert('RGB')
                
                # Create UV coordinates (planar mapping from top)
                # Normalize vertices x, y to 0-1 range
                uv = pizza.vertices[:, :2] # Take X and Y
                uv = (uv / (2 * radius)) + 0.5
                
                material = pyrender.Material(
                    baseColorTexture=pyrender.Texture(source=im, source_channels='RGB'),
                    metallicFactor=0.0,
                    roughnessFactor=0.8
                )
            except Exception as e:
                logger.warning(f"Failed to apply texture: {e}")
                material = pyrender.Material(baseColorFactor=[0.8, 0.6, 0.4, 1.0])
        else:
            material = pyrender.Material(baseColorFactor=[0.8, 0.6, 0.4, 1.0])
            
        return pyrender.Mesh.from_trimesh(pizza, material=material)

    def generate_sample(self, filename_prefix="pizza_3d"):
        """Generate a single 3D rendered sample."""
        self.scene.clear()
        
        # Create pizza mesh
        mesh = self.create_pizza_mesh()
        mesh_node = self.scene.add(mesh)
        
        # Set up camera
        # Random camera position on a hemisphere
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0.1, math.pi / 3) # Don't go too low (under the pizza)
        dist = random.uniform(2.5, 4.0)
        
        x = dist * math.sin(phi) * math.cos(theta)
        y = dist * math.cos(phi) # Y is up
        z = dist * math.sin(phi) * math.sin(theta)
        
        camera_pose = np.array([
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Look at origin
        forward = -camera_pose[:3, 3]
        forward = forward / np.linalg.norm(forward)
        right = np.cross(np.array([0, 1, 0]), forward)
        if np.linalg.norm(right) < 0.001:
            right = np.array([1, 0, 0])
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        
        camera_pose[:3, 0] = right
        camera_pose[:3, 1] = up
        camera_pose[:3, 2] = -forward # Camera looks down -Z
        
        self.scene.add(self.camera, pose=camera_pose)
        
        # Add lighting
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
        self.scene.add(light, pose=camera_pose)
        
        # Render
        color, depth = self.renderer.render(self.scene)
        
        # Save image
        img = Image.fromarray(color)
        output_path = self.output_dir / f"{filename_prefix}_{int(time.time()*1000)}.png"
        img.save(output_path)
        
        return str(output_path)

    def generate_dataset(self, num_samples=10):
        """Generate a dataset of 3D rendered images."""
        logger.info(f"Generating {num_samples} 3D samples...")
        generated_files = []
        import time
        
        for i in range(num_samples):
            try:
                path = self.generate_sample(f"sample_{i}")
                generated_files.append(path)
            except Exception as e:
                logger.error(f"Failed to generate sample {i}: {e}")
                
        logger.info(f"Generated {len(generated_files)} samples in {self.output_dir}")
        return generated_files

if __name__ == "__main__":
    # Check if we have an X server or EGL (needed for pyrender)
    # On headless systems, this might require xvfb-run
    if not os.environ.get("PYOPENGL_PLATFORM"):
        os.environ["PYOPENGL_PLATFORM"] = "osmesa" # Try software rendering if available
        
    try:
        generator = Pizza3DGenerator()
        generator.generate_dataset(5)
    except Exception as e:
        logger.error(f"3D Generation failed: {e}")
        logger.info("Note: Pyrender requires OpenGL. On headless servers, install libosmesa6-dev and set PYOPENGL_PLATFORM=osmesa")
