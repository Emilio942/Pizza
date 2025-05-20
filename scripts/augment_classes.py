import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import functional as TVF
import random
import math
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw

class PizzaBurningEffect(nn.Module):
    """Optimized simulation of burning effects for pizza images"""
    
    def __init__(self, burn_intensity_min=0.2, burn_intensity_max=0.8, burn_pattern='random'):
        super().__init__()
        self.burn_intensity_min = burn_intensity_min
        self.burn_intensity_max = burn_intensity_max
        self.burn_pattern = burn_pattern  # 'random', 'edge', 'spot', 'streak'
        
    def _create_edge_burn_mask(self, h, w, device):
        """Create an edge burn mask"""
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing='ij'
        )
        
        # Distance from center
        dist = torch.sqrt(x_coords**2 + y_coords**2)
        
        # Random distortion for irregular edges
        noise = torch.randn(h, w, device=device) * 0.05
        dist = dist + noise
        
        # Exponential decay from edge to center with random threshold
        threshold = random.uniform(0.6, 0.8)
        edge_weight = torch.exp(3 * (dist - threshold))
        
        # Normalize between 0 and 1
        edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min())
        
        # Add random variations
        if random.random() < 0.3:
            # Asymmetric burning - more on one side
            side = random.choice(['left', 'right', 'top', 'bottom'])
            if side == 'left':
                side_mask = (x_coords < -0.3)
            elif side == 'right':
                side_mask = (x_coords > 0.3)
            elif side == 'top':
                side_mask = (y_coords < -0.3)
            else:  # bottom
                side_mask = (y_coords > 0.3)
            
            edge_weight = torch.where(side_mask, edge_weight * random.uniform(1.2, 1.5), edge_weight)
        
        return edge_weight
    
    def _create_spot_burn_mask(self, h, w, device):
        """Create a spot burn mask"""
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing='ij'
        )
        
        spots_mask = torch.zeros(h, w, device=device)
        num_spots = random.randint(3, 10)
        
        # Create spots with varying intensity and shape
        for _ in range(num_spots):
            # Random position, slightly biased towards the edge
            r = random.uniform(0.3, 1.0)  # Radius from center
            theta = random.uniform(0, 2*np.pi)  # Angle
            spot_x = r * np.cos(theta)
            spot_y = r * np.sin(theta)
            
            # Different shapes by scaling axes
            if random.random() < 0.3:
                # Elliptical instead of circular
                x_scale = random.uniform(0.5, 1.5)
                y_scale = random.uniform(0.5, 1.5)
                spot_dist = torch.sqrt(((x_coords - spot_x)/x_scale)**2 + 
                                       ((y_coords - spot_y)/y_scale)**2)
            else:
                # Circular
                spot_dist = torch.sqrt((x_coords - spot_x)**2 + (y_coords - spot_y)**2)
            
            # Parameters for the spot
            spot_radius = random.uniform(0.05, 0.25)
            spot_intensity = random.uniform(0.6, 1.0)
            spot_falloff = random.uniform(1.0, 3.0)
            
            # Generate spot with different profiles
            if random.random() < 0.5:
                # Exponential profile
                spot_mask = torch.exp(-spot_dist * spot_falloff / spot_radius) * spot_intensity
            else:
                # Quadratic profile for sharper edges
                normalized_dist = spot_dist / spot_radius
                spot_mask = torch.maximum(torch.zeros_like(normalized_dist), 
                                         (1 - normalized_dist**2)) * spot_intensity
            
            # Combine with existing mask (maximum for overlapping areas)
            spots_mask = torch.maximum(spots_mask, spot_mask)
        
        return spots_mask
    
    def _create_streak_burn_mask(self, h, w, device):
        """Create a streak burn mask"""
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing='ij'
        )
        
        streaks_mask = torch.zeros(h, w, device=device)
        num_streaks = random.randint(1, 4)
        
        for _ in range(num_streaks):
            # Define a line with random orientation
            theta = random.uniform(0, np.pi)  # Line angle
            rho = random.uniform(-0.8, 0.8)   # Distance from origin
            
            # Normal line: x*cos(theta) + y*sin(theta) = rho
            dist_to_line = torch.abs(x_coords * np.cos(theta) + y_coords * np.sin(theta) - rho)
            
            # Parameters for the streak
            streak_width = random.uniform(0.05, 0.15)
            streak_intensity = random.uniform(0.7, 1.0)
            
            # Generate streak with Gaussian profile
            streak_mask = torch.exp(-(dist_to_line**2) / (2 * streak_width**2)) * streak_intensity
            
            # Add slight variation to the line
            noise = torch.randn(h, w, device=device) * 0.03
            streak_mask = streak_mask * (1 + noise)
            
            # Combine with existing mask
            streaks_mask = torch.maximum(streaks_mask, streak_mask)
        
        return streaks_mask
    
    def _create_random_burn_mask(self, h, w, device):
        """Combine different burning patterns randomly"""
        pattern_weights = {
            'edge': random.uniform(0.3, 1.0) if random.random() < 0.8 else 0,
            'spot': random.uniform(0.5, 1.0),
            'streak': random.uniform(0.2, 0.8) if random.random() < 0.4 else 0
        }
        
        mask = torch.zeros(h, w, device=device)
        
        if pattern_weights['edge'] > 0:
            edge_mask = self._create_edge_burn_mask(h, w, device)
            mask = torch.maximum(mask, edge_mask * pattern_weights['edge'])
        
        if pattern_weights['spot'] > 0:
            spot_mask = self._create_spot_burn_mask(h, w, device)
            mask = torch.maximum(mask, spot_mask * pattern_weights['spot'])
        
        if pattern_weights['streak'] > 0:
            streak_mask = self._create_streak_burn_mask(h, w, device)
            mask = torch.maximum(mask, streak_mask * pattern_weights['streak'])
        
        # Normalize between 0 and 1
        if mask.max() > 0:
            mask = mask / mask.max()
        
        return mask
    
    def _create_burn_color(self, burn_level):
        """Create realistic burning colors based on intensity"""
        # Color variations from lightly browned to charred
        if burn_level < 0.2:  # Lightly browned
            return torch.tensor([0.85, 0.65, 0.45])
        elif burn_level < 0.5:  # Medium burnt
            return torch.tensor([0.65, 0.40, 0.25])
        elif burn_level < 0.8:  # Heavily burnt
            return torch.tensor([0.35, 0.20, 0.15])
        else:  # Charred
            return torch.tensor([0.15, 0.10, 0.10])
    
    def forward(self, img):
        # Ensure the image is a tensor and on the right device
        if not isinstance(img, torch.Tensor):
            img = TVF.to_tensor(img)
        
        device = img.device
        h, w = img.shape[1], img.shape[2]
        
        # Generate burn mask based on selected pattern
        if self.burn_pattern == 'edge':
            burn_mask_2d = self._create_edge_burn_mask(h, w, device)
        elif self.burn_pattern == 'spot':
            burn_mask_2d = self._create_spot_burn_mask(h, w, device)
        elif self.burn_pattern == 'streak':
            burn_mask_2d = self._create_streak_burn_mask(h, w, device)
        else:  # 'random' or default
            burn_mask_2d = self._create_random_burn_mask(h, w, device)
        
        # Random burn intensity
        burn_intensity = random.uniform(self.burn_intensity_min, self.burn_intensity_max)
        burn_mask_2d = burn_mask_2d * burn_intensity
        
        # Expand for all channels
        burn_mask = burn_mask_2d.unsqueeze(0).expand_as(img)
        
        # Create burning effect with different color levels
        result = img.clone()
        
        # Apply different burning colors by intensity
        burn_levels = [0.25, 0.5, 0.75, 1.0]
        
        for level in burn_levels:
            level_mask = (burn_mask_2d > level * 0.8) & (burn_mask_2d <= level)
            if level_mask.any():
                burn_color = self._create_burn_color(level).to(device)
                
                # Expand mask and color for all channels
                level_mask_3d = level_mask.unsqueeze(0).expand_as(img)
                burn_color_3d = burn_color.view(3, 1, 1).expand_as(img)
                
                # Blend original image with burn color
                blend_factor = torch.ones_like(img) * (level * burn_intensity)
                result = torch.where(level_mask_3d, 
                                   img * (1 - blend_factor) + burn_color_3d * blend_factor,
                                   result)
        
        # Heavy burning (charred)
        charred_mask = (burn_mask_2d > 0.8)
        if charred_mask.any():
            charred_color = torch.tensor([0.05, 0.05, 0.05]).to(device)
            charred_mask_3d = charred_mask.unsqueeze(0).expand_as(img)
            charred_color_3d = charred_color.view(3, 1, 1).expand_as(img)
            
            result = torch.where(charred_mask_3d, charred_color_3d, result)
        
        # Add subtle textures for burnt areas
        if random.random() < 0.7:
            texture_noise = torch.randn_like(burn_mask_2d) * 0.05
            texture_mask = (burn_mask_2d > 0.3).unsqueeze(0).expand_as(img)
            result = torch.where(texture_mask, result * (1 + texture_noise.unsqueeze(0)), result)
        
        # Clamp values to valid range
        result = torch.clamp(result, 0, 1)
        
        return result


class OvenEffect(nn.Module):
    """Enhanced simulation of oven effects with realistic details"""
    
    def __init__(self, effect_strength=1.0, scipy_available=False):
        super().__init__()
        self.effect_strength = effect_strength
        self.scipy_available = scipy_available
        
    def _apply_gaussian_blur(self, tensor, kernel_size, sigma, device):
        """Apply Gaussian blur to a tensor"""
        if self.scipy_available:
            # More efficient implementation with SciPy, but requires CPU transfer
            from scipy.ndimage import gaussian_filter
            tensor_np = tensor.squeeze(0).cpu().numpy()
            blurred_np = gaussian_filter(tensor_np, sigma=sigma)
            return torch.tensor(blurred_np, device=device).unsqueeze(0)
        else:
            # PyTorch-based alternative
            # Ensure kernel_size is odd
            kernel_size = max(3, int(kernel_size) // 2 * 2 + 1)
            return TVF.gaussian_blur(tensor, kernel_size=kernel_size, sigma=sigma)
    
    def forward(self, img):
        # Ensure the image is a tensor and on the right device
        if not isinstance(img, torch.Tensor):
            img = TVF.to_tensor(img)
        
        device = img.device
        h, w = img.shape[1], img.shape[2]
        
        # Choose effects based on strength and randomness
        effects = []
        if random.random() < 0.4 * self.effect_strength:
            effects.append('steam')
        if random.random() < 0.5 * self.effect_strength:
            effects.append('warmth')
        if random.random() < 0.3 * self.effect_strength:
            effects.append('shadow')
        if random.random() < 0.4 * self.effect_strength:
            effects.append('lighting')
        
        # If no effects selected, choose a random one
        if not effects and random.random() < 0.7:
            effects.append(random.choice(['steam', 'warmth', 'shadow', 'lighting']))
        
        # Only create a copy if effects will be applied
        result = img.clone() if effects else img
        
        # Steam effect - more realistic with gradients and motion blur
        if 'steam' in effects:
            steam_opacity = random.uniform(0.1, 0.3) * self.effect_strength
            
            # Generate enhanced steam mask
            # Base is a vertical gradient
            y_coords = torch.linspace(1.0, 0.0, h, device=device).view(-1, 1).expand(-1, w)
            
            # Add random variations
            noise_scale = random.uniform(0.3, 0.7)
            noise = torch.rand(h, w, device=device) * noise_scale
            
            # Combine with gradient for more realistic steam
            steam_base = y_coords * noise
            
            # Smoothing the mask
            sigma = random.uniform(5, 20)
            kernel_size = int(sigma * 3) // 2 * 2 + 1  # Odd number ~3*sigma
            steam_mask = self._apply_gaussian_blur(
                steam_base.unsqueeze(0), kernel_size, sigma, device
            ).squeeze(0) * steam_opacity
            
            # Expand for all channels
            steam_mask_3d = steam_mask.unsqueeze(0).expand_as(result)
            
            # Brighten where steam is, with slightly bluish tint for realism
            steam_color = torch.ones_like(result)
            steam_color[2] = steam_color[2] * 1.05  # Slightly more blue
            
            result = result * (1 - steam_mask_3d) + steam_color * steam_mask_3d
        
        # Warmth effect - gives the image a warmer color tone with subtle variations
        if 'warmth' in effects:
            warmth = random.uniform(0.05, 0.15) * self.effect_strength
            
            # Generate warmth gradient (more in center)
            y_coords, x_coords = torch.meshgrid(
                torch.linspace(-1, 1, h, device=device),
                torch.linspace(-1, 1, w, device=device),
                indexing='ij'
            )
            
            # Circular gradient from center
            dist_from_center = torch.sqrt(x_coords**2 + y_coords**2)
            warmth_gradient = torch.exp(-dist_from_center * 2) * 0.5 + 0.5
            
            # Apply warmth effect with gradient
            warmth_factor = warmth * warmth_gradient
            
            # Channel-wise adjustment
            result_channels = result.clone()
            # Increase red
            result_channels[0] = torch.clamp(result[0] * (1 + warmth_factor), 0, 1)
            # Slightly increase green
            result_channels[1] = torch.clamp(result[1] * (1 + warmth_factor * 0.3), 0, 1)
            # Decrease blue
            result_channels[2] = torch.clamp(result[2] * (1 - warmth_factor * 0.2), 0, 1)
            
            result = result_channels
        
        # Shadow effect with more realistic transitions
        if 'shadow' in effects:
            shadow_opacity = random.uniform(0.15, 0.4) * self.effect_strength
            
            # Generate coordinate grid
            y_coords, x_coords = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing='ij'
            )
            
            # Parameters for the shadow
            shadow_type = random.choice(['corner', 'side', 'spot'])
            
            if shadow_type == 'corner':
                # Shadow effect from a corner
                corner = random.choice(['tl', 'tr', 'bl', 'br'])
                if corner == 'tl':
                    corner_x, corner_y = 0, 0
                elif corner == 'tr':
                    corner_x, corner_y = w-1, 0
                elif corner == 'bl':
                    corner_x, corner_y = 0, h-1
                else:  # 'br'
                    corner_x, corner_y = w-1, h-1
                
                shadow_dist = torch.sqrt(((x_coords - corner_x)**2 + (y_coords - corner_y)**2).float())
                shadow_radius = random.uniform(0.7, 1.3) * max(h, w)
                shadow_mask = torch.exp(-shadow_dist / shadow_radius) * shadow_opacity
            
            elif shadow_type == 'side':
                # Shadow from a side
                side = random.choice(['left', 'right', 'top', 'bottom'])
                
                if side == 'left':
                    shadow_mask = torch.exp(-(x_coords.float()) / (w * 0.3)) * shadow_opacity
                elif side == 'right':
                    shadow_mask = torch.exp(-((w - 1 - x_coords).float()) / (w * 0.3)) * shadow_opacity
                elif side == 'top':
                    shadow_mask = torch.exp(-(y_coords.float()) / (h * 0.3)) * shadow_opacity
                else:  # 'bottom'
                    shadow_mask = torch.exp(-((h - 1 - y_coords).float()) / (h * 0.3)) * shadow_opacity
            
            else:  # 'spot'
                # Round shadow at random position
                shadow_x = random.randint(w//4, w*3//4)
                shadow_y = random.randint(h//4, h*3//4)
                shadow_radius = random.uniform(0.3, 0.5) * min(h, w)
                
                shadow_dist = torch.sqrt(((x_coords - shadow_x)**2 + (y_coords - shadow_y)**2).float())
                shadow_mask = (1 - torch.exp(-shadow_dist**2 / (2 * shadow_radius**2))) * shadow_opacity
            
            # Add slight blur to the shadow
            sigma = random.uniform(10, 30)
            kernel_size = int(sigma * 2) // 2 * 2 + 1
            shadow_mask = self._apply_gaussian_blur(
                shadow_mask.unsqueeze(0), kernel_size, sigma, device
            ).squeeze(0)
            
            # Expand for all channels
            shadow_mask_3d = shadow_mask.unsqueeze(0).expand_as(result)
            
            # Darken where shadow is
            result = result * (1 - shadow_mask_3d)
        
        # Lighting effect - simulates uneven lighting in the oven
        if 'lighting' in effects:
            light_intensity = random.uniform(0.1, 0.25) * self.effect_strength
            
            # Generate coordinate grid
            y_coords, x_coords = torch.meshgrid(
                torch.linspace(-1, 1, h, device=device),
                torch.linspace(-1, 1, w, device=device),
                indexing='ij'
            )
            
            # Parameters for the light
            light_x = random.uniform(-0.5, 0.5)
            light_y = random.uniform(-0.5, 0.5)
            light_radius = random.uniform(0.4, 0.8)
            
            # Calculate distance to light source
            light_dist = torch.sqrt((x_coords - light_x)**2 + (y_coords - light_y)**2)
            
            # Generate light mask with soft falloff
            light_mask = torch.exp(-light_dist**2 / (2 * light_radius**2)) * light_intensity
            
            # Expand for all channels
            light_mask_3d = light_mask.unsqueeze(0).expand_as(result)
            
            # Brighten where light is, with slightly yellowish tint
            light_color = torch.ones_like(result)
            light_color[0] = light_color[0] * 1.05  # Slightly more red
            light_color[1] = light_color[1] * 1.03  # Slightly more green
            
            result = result * (1 - light_mask_3d) + torch.minimum(
                light_color * result * (1 + light_mask_3d),
                torch.ones_like(result)
            )
        
        # Clamp values to valid range
        result = torch.clamp(result, 0, 1)
        
        return result


class PizzaSegmentEffect(nn.Module):
    """Modular effect system for individual pizza segments"""
    
    def __init__(self, device, burning_min=0.0, burning_max=0.9):
        super().__init__()
        self.device = device
        self.burning_min = burning_min
        self.burning_max = burning_max
        
        # Instantiate effect modules
        self.burning_effect = PizzaBurningEffect(
            burn_intensity_min=burning_min,
            burn_intensity_max=burning_max
        ).to(device)
        
        self.oven_effect = OvenEffect(
            effect_strength=random.uniform(0.5, 1.0)
        ).to(device)
    
    def _create_segment_mask(self, h, w, num_segments=8):
        """Generate mask for pizza segments"""
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, h, device=self.device),
            torch.linspace(-1, 1, w, device=self.device),
            indexing='ij'
        )
        
        # Calculate angle for all pixels (in radians)
        angles = torch.atan2(y_coords, x_coords)
        # Normalize angles to [0, 2π]
        angles = torch.where(angles < 0, angles + 2 * np.pi, angles)
        
        # Distance from center
        dist_from_center = torch.sqrt(x_coords**2 + y_coords**2)
        
        # Circle mask for pizza area
        pizza_radius = random.uniform(0.7, 0.9)
        pizza_mask = (dist_from_center <= pizza_radius).float()
        
        # Segment masks
        segment_size = 2 * np.pi / num_segments
        segment_masks = []
        
        for i in range(num_segments):
            start_angle = i * segment_size
            end_angle = (i + 1) * segment_size
            
            # Mask for current segment
            if end_angle <= 2 * np.pi:
                segment_mask = ((angles >= start_angle) & (angles < end_angle)).float()
            else:
                # Handle overflow
                segment_mask = ((angles >= start_angle) | (angles < (end_angle % (2 * np.pi)))).float()
            
            # Combine with pizza mask
            segment_mask = segment_mask * pizza_mask
            
            segment_masks.append(segment_mask)
        
        return segment_masks
    
    def forward(self, img):
        # Ensure the image is a tensor and on the right device
        if not isinstance(img, torch.Tensor):
            img = TVF.to_tensor(img).to(self.device)
        else:
            img = img.to(self.device)
        
        h, w = img.shape[1], img.shape[2]
        
        # Create random number of segments
        num_segments = random.randint(4, 8)
        segment_masks = self._create_segment_mask(h, w, num_segments)
        
        # Create different effect combinations for each segment
        result = img.clone()
        
        for i, segment_mask in enumerate(segment_masks):
            # Random effects per segment
            segment_img = img.clone()
            
            # Random burning intensity
            burn_intensity = random.uniform(0, 1)
            
            if burn_intensity > 0.1:  # Apply burning effects
                # Adjust burning intensity per segment
                self.burning_effect.burn_intensity_min = self.burning_min + burn_intensity * 0.3
                self.burning_effect.burn_intensity_max = min(self.burning_min + burn_intensity * 0.6, self.burning_max)
                
                # Choose pattern
                self.burning_effect.burn_pattern = random.choice(['random', 'edge', 'spot'])
                
                # Apply effect
                segment_img = self.burning_effect(segment_img)
            
            # Random application of oven effects
            if random.random() < 0.5:
                segment_img = self.oven_effect(segment_img)
            
            # Expand mask for all channels
            segment_mask_3d = segment_mask.unsqueeze(0).expand_as(result)
            
            # Combine with result
            result = torch.where(segment_mask_3d > 0, segment_img, result)
        
        return result


# ================ LIGHTING AND PERSPECTIVE AUGMENTATION CLASSES ================

class DirectionalLightEffect(nn.Module):
    """
    Simulates directional lighting effects on pizza images
    by adding highlights and shadows based on a light direction vector
    """
    
    def __init__(self, 
                 light_intensity_min=0.5,
                 light_intensity_max=1.5,
                 shadow_intensity_min=0.4, 
                 shadow_intensity_max=0.8,
                 light_position='random',
                 light_color=None,
                 specular_highlight_prob=0.3):
        super().__init__()
        self.light_intensity_min = light_intensity_min
        self.light_intensity_max = light_intensity_max
        self.shadow_intensity_min = shadow_intensity_min
        self.shadow_intensity_max = shadow_intensity_max
        self.light_position = light_position  # 'random', 'overhead', 'side', 'front', 'corner'
        self.light_color = light_color  # None for white light, or RGB tuple for colored lighting
        self.specular_highlight_prob = specular_highlight_prob
    
    def forward(self, img):
        """Apply directional lighting effect to pizza image"""
        if not torch.is_tensor(img):
            # Convert PIL to tensor
            img_tensor = TVF.to_tensor(img)
        else:
            img_tensor = img.clone()
        
        # Get image dimensions
        c, h, w = img_tensor.shape
        
        # Create light direction vector based on specified light position
        if self.light_position == 'random':
            light_pos = random.choice(['overhead', 'side', 'front', 'corner'])
        else:
            light_pos = self.light_position
            
        # Create light direction vector based on position
        if light_pos == 'overhead':
            # Light coming directly from above
            light_dir_x = 0
            light_dir_y = random.uniform(-0.2, 0.2)  # Slight variation
            light_dir_z = -1  # Pointing down
        elif light_pos == 'side':
            # Light coming from the side
            side = random.choice(['left', 'right'])
            light_dir_x = -1 if side == 'left' else 1
            light_dir_y = random.uniform(-0.3, 0.3)
            light_dir_z = random.uniform(-0.5, -0.2)  # Slightly downward
        elif light_pos == 'front':
            # Light coming from camera position
            light_dir_x = random.uniform(-0.2, 0.2)
            light_dir_y = random.uniform(-0.2, 0.2)
            light_dir_z = -1  # Pointing directly at pizza
        else:  # 'corner'
            # Light coming from one of the corners
            corner_x = random.choice([-1, 1])
            corner_y = random.choice([-1, 1])
            light_dir_x = corner_x * random.uniform(0.7, 1.0)
            light_dir_y = corner_y * random.uniform(0.7, 1.0)
            light_dir_z = random.uniform(-0.8, -0.4)
        
        # Normalize light direction vector
        norm = torch.sqrt(torch.tensor(light_dir_x**2 + light_dir_y**2 + light_dir_z**2))
        light_dir_x /= norm
        light_dir_y /= norm
        light_dir_z /= norm
        
        # Create coordinates for each pixel (relative to center)
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, h, device=img_tensor.device),
            torch.linspace(-1, 1, w, device=img_tensor.device),
            indexing='ij'
        )
        
        # Assume pizza is a slightly elevated surface (more realistic)
        # Create a simplified height map (assuming pizza is roughly circular)
        center_dist = torch.sqrt(x_coords**2 + y_coords**2)
        height_map = torch.where(
            center_dist < 0.8,  # Pizza region
            torch.cos(center_dist * math.pi / 2 * 0.8) * 0.1,  # Slight elevation for pizza
            torch.zeros_like(center_dist)  # Flat background
        )
        
        # Calculate surface normals (simplified)
        # For each pixel, calculate gradient in x and y direction
        gradient_x = torch.zeros_like(height_map)
        gradient_y = torch.zeros_like(height_map)
        
        # Approximate gradients using neighbor differences
        gradient_x[:, 1:-1] = (height_map[:, 2:] - height_map[:, :-2]) / 2
        gradient_y[1:-1, :] = (height_map[2:, :] - height_map[:-2, :]) / 2
        
        # Surface normal is (-gradient_x, -gradient_y, 1)
        # Normalized to unit length
        normal_z = torch.ones_like(height_map)
        normal_magnitude = torch.sqrt(gradient_x**2 + gradient_y**2 + normal_z**2)
        
        # Compute dot product between light direction and surface normal
        light_dot_normal = (-gradient_x * light_dir_x - gradient_y * light_dir_y + normal_z * (-light_dir_z)) / normal_magnitude
        
        # Ensure the dot product is in range [-1, 1]
        light_dot_normal = torch.clamp(light_dot_normal, -1, 1)
        
        # Create lighting mask from dot product
        # Higher values mean the surface is facing the light
        light_mask = (light_dot_normal + 1) / 2  # Scale to [0, 1]
        
        # Light intensity decreases with distance from light source
        # (simplified for a distant light source)
        light_intensity = random.uniform(self.light_intensity_min, self.light_intensity_max)
        shadow_intensity = random.uniform(self.shadow_intensity_min, self.shadow_intensity_max)
        
        # Apply light mask to each channel with potential color tint
        light_color = self.light_color
        if light_color is None:
            # Random subtle color temperature variation
            if random.random() < 0.5:
                # Warm light (yellowish)
                light_color = (1.0, random.uniform(0.9, 1.0), random.uniform(0.7, 0.9))
            else:
                # Cool light (bluish)
                light_color = (random.uniform(0.7, 0.9), random.uniform(0.9, 1.0), 1.0)
        
        # Apply lighting effect with potential color
        for i in range(min(c, 3)):  # Handle only RGB channels
            channel_light_intensity = light_intensity * light_color[i] if i < len(light_color) else light_intensity
            channel_shadow_intensity = shadow_intensity
            
            # Apply lighting: brighten areas facing the light, darken others
            lighting_effect = light_mask * (channel_light_intensity - 1) + (1 - light_mask) * (channel_shadow_intensity - 1) + 1
            img_tensor[i] = img_tensor[i] * lighting_effect
        
        # Add specular highlight with probability
        if random.random() < self.specular_highlight_prob:
            # Specular highlight appears where surface directly reflects light to viewer
            # Simplified: it's strongest where light_dot_normal is highest
            specular_mask = (light_mask > 0.7) * (torch.pow(light_mask, 8) * random.uniform(0.2, 0.5))
            
            # Add the highlight to all channels
            for i in range(min(c, 3)):
                img_tensor[i] = torch.clamp(img_tensor[i] + specular_mask, 0, 1)
        
        # Ensure values are in valid range
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        if not torch.is_tensor(img):
            return TVF.to_pil_image(img_tensor)
        
        return img_tensor


class CLAHEEffect(nn.Module):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE) implementation
    for improving local contrast in pizza images, especially in areas with
    poor exposure. This is important for enhancing texture details in
    pizza photographs taken under uneven lighting conditions.
    """
    
    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8), 
                 contrast_limit=(0.8, 1.5), detail_enhancement=0.3):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.contrast_limit = contrast_limit  # Min/max contrast adjustment
        self.detail_enhancement = detail_enhancement  # How much to enhance details
    
    def _apply_clahe_pil(self, img):
        """Apply CLAHE-like enhancement using PIL operations"""
        # Convert to LAB color space (using YCbCr as approximation available in PIL)
        img_ycbcr = img.convert('YCbCr')
        y, cb, cr = img_ycbcr.split()
        
        # Apply contrast enhancement to Y channel
        y_enhanced = ImageOps.equalize(y)
        
        # Apply additional local contrast enhancement
        y_contrast = ImageEnhance.Contrast(y_enhanced).enhance(
            random.uniform(self.contrast_limit[0], self.contrast_limit[1])
        )
        
        # Blend original and enhanced Y channel to control enhancement strength
        blend_factor = random.uniform(0.5, self.detail_enhancement + 0.5)
        y_blend = Image.blend(y, y_contrast, blend_factor)
        
        # Merge back with color channels
        img_enhanced = Image.merge('YCbCr', (y_blend, cb, cr))
        return img_enhanced.convert('RGB')
    
    def _apply_clahe_torch(self, img_tensor):
        """Apply CLAHE-like effect using torch operations"""
        # Clone to avoid modifying the original
        result = img_tensor.clone()
        
        # Get dimensions
        c, h, w = img_tensor.shape
        
        # Convert to YCbCr-like space for better separation of luminance
        # Approximate conversion from RGB to YCbCr
        if c >= 3:
            # Y (luminance) channel calculation
            y = 0.299 * img_tensor[0] + 0.587 * img_tensor[1] + 0.114 * img_tensor[2]
        else:
            # For grayscale, use the channel directly
            y = img_tensor[0]
        
        # Create grid of tiles
        tile_h, tile_w = self.tile_grid_size
        h_step, w_step = max(1, h // tile_h), max(1, w // tile_w)
        
        # Process each tile
        for i in range(tile_h):
            for j in range(tile_w):
                # Get current tile coordinates
                h_start, h_end = i * h_step, min((i + 1) * h_step, h)
                w_start, w_end = j * w_step, min((j + 1) * w_step, w)
                
                # Skip if tile is empty
                if h_end <= h_start or w_end <= w_start:
                    continue
                
                # Get current tile
                tile = y[h_start:h_end, w_start:w_end]
                
                # Calculate histogram
                hist = torch.histc(tile, bins=256, min=0, max=1)
                
                # Apply histogram clipping (CLAHE)
                if self.clip_limit > 0:
                    excess = torch.sum(torch.maximum(hist - self.clip_limit, torch.tensor(0.)))
                    clipped_hist = torch.minimum(hist, torch.tensor(self.clip_limit))
                    redistribution_value = excess / 256
                    clipped_hist += redistribution_value
                    hist = clipped_hist
                
                # Compute cumulative distribution function (CDF)
                cdf = torch.cumsum(hist, 0)
                
                # Normalize CDF to [0, 1]
                if cdf[-1] > 0:
                    cdf = cdf / cdf[-1]
                
                # Apply the transformation (lookup)
                # This is an approximation as exact histogram matching is complex
                
                # Scale original values through the CDF
                tile_min, tile_max = torch.min(tile), torch.max(tile)
                if tile_max > tile_min:
                    # Normalize to [0, 1] for lookup
                    normalized_tile = (tile - tile_min) / (tile_max - tile_min)
                    
                    # Map to histogram indices (0-255)
                    idx = (normalized_tile * 255).long().clamp(0, 255)
                    
                    # Apply CDF transformation
                    equalized_tile = torch.zeros_like(tile)
                    for k in range(256):
                        mask = (idx == k)
                        equalized_tile[mask] = cdf[k]
                    
                    # Adjust contrast based on contrast_limit
                    contrast_factor = random.uniform(self.contrast_limit[0], self.contrast_limit[1])
                    mean_val = torch.mean(equalized_tile)
                    equalized_tile = (equalized_tile - mean_val) * contrast_factor + mean_val
                    equalized_tile = torch.clamp(equalized_tile, 0, 1)
                    
                    # Update the luminance channel with equalized values
                    blend_factor = random.uniform(0.5, self.detail_enhancement + 0.5)
                    y[h_start:h_end, w_start:w_end] = torch.lerp(
                        tile, equalized_tile, blend_factor
                    )
        
        # Apply enhanced luminance while preserving color
        if c >= 3:
            # Original luminance
            orig_y = 0.299 * result[0] + 0.587 * result[1] + 0.114 * result[2]
            
            # Calculate luminance ratio for each pixel
            ratio = torch.ones_like(orig_y)
            mask = orig_y > 0.01  # Avoid division by very small values
            ratio[mask] = y[mask] / orig_y[mask]
            
            # Apply the ratio to each channel to preserve color relationships
            for i in range(3):
                result[i] = torch.clamp(result[i] * ratio, 0, 1)
        else:
            # For grayscale, directly use the enhanced luminance
            result[0] = y
        
        return result
    
    def forward(self, img):
        """Apply CLAHE effect to pizza image"""
        if torch.is_tensor(img):
            return self._apply_clahe_torch(img)
        else:
            return self._apply_clahe_pil(img)


class ExposureVariationEffect(nn.Module):
    """
    Simulate over-exposure and under-exposure conditions in pizza images,
    focusing on realistic lighting variations encountered in real-world settings
    such as restaurants, home kitchens, and outdoor environments.
    """
    
    def __init__(self, 
                 underexposure_prob=0.3,
                 overexposure_prob=0.3,
                 exposure_range=(0.3, 1.8),
                 vignette_prob=0.4,
                 color_temp_variation=True,
                 noise_prob=0.3):
        super().__init__()
        self.underexposure_prob = underexposure_prob
        self.overexposure_prob = overexposure_prob
        self.exposure_range = exposure_range
        self.vignette_prob = vignette_prob
        self.color_temp_variation = color_temp_variation  # Color temperature shifts
        self.noise_prob = noise_prob  # Add noise in low-light conditions
    
    def _apply_vignette(self, img_tensor):
        """Apply vignette effect (darkened corners)"""
        c, h, w = img_tensor.shape
        
        # Create center coordinates
        center_y, center_x = h / 2, w / 2
        y, x = torch.meshgrid(torch.arange(h, device=img_tensor.device), 
                             torch.arange(w, device=img_tensor.device), 
                             indexing='ij')
        
        # Calculate squared distance from center
        dist_squared = (x - center_x)**2 + (y - center_y)**2
        max_dist_squared = max(center_x**2, center_y**2) * 2
        
        # Create normalized vignette mask
        vignette_mask = 1 - torch.sqrt(dist_squared / max_dist_squared)
        
        # Add random variation to vignette shape (elliptical)
        if random.random() < 0.6:
            aspect_ratio = random.uniform(0.7, 1.3)
            x_scale = aspect_ratio if random.random() < 0.5 else 1.0
            y_scale = 1.0 if x_scale != 1.0 else aspect_ratio
            
            dist_squared = ((x - center_x) * x_scale)**2 + ((y - center_y) * y_scale)**2
            max_dist_squared = max((center_x * x_scale)**2, (center_y * y_scale)**2) * 2
            vignette_mask = 1 - torch.sqrt(dist_squared / max_dist_squared)
        
        # Adjust vignette intensity and smoothness
        intensity = random.uniform(0.6, 0.9)
        vignette_mask = torch.clamp(vignette_mask, 0, 1) ** random.uniform(0.5, 3.0)
        vignette_mask = intensity + (1 - intensity) * vignette_mask
        
        # Apply vignette mask to all channels
        for i in range(c):
            img_tensor[i] = img_tensor[i] * vignette_mask
        
        return img_tensor
    
    def _apply_color_temp(self, img_tensor, is_underexposed):
        """Apply color temperature shift based on lighting condition"""
        c, h, w = img_tensor.shape
        
        if c < 3:
            return img_tensor  # Only apply to RGB images
        
        if is_underexposed:
            # Cool/blue tint for underexposed (night/evening/indoor low light)
            blue_boost = random.uniform(1.0, 1.2)
            red_scale = random.uniform(0.8, 1.0)
            
            img_tensor[0] = img_tensor[0] * red_scale  # Reduce red
            img_tensor[2] = torch.clamp(img_tensor[2] * blue_boost, 0, 1)  # Increase blue
        else:
            # Warm/yellow tint for overexposed (bright sunlight)
            yellow_boost = random.uniform(1.0, 1.15)
            blue_scale = random.uniform(0.85, 1.0)
            
            img_tensor[0] = torch.clamp(img_tensor[0] * yellow_boost, 0, 1)  # Increase red
            img_tensor[1] = torch.clamp(img_tensor[1] * yellow_boost, 0, 1)  # Increase green
            img_tensor[2] = img_tensor[2] * blue_scale  # Reduce blue
        
        return img_tensor
    
    def _apply_noise(self, img_tensor, noise_level):
        """Apply realistic noise to simulate low-light conditions"""
        c, h, w = img_tensor.shape
        
        # Generate Gaussian noise with intensity based on the noise level
        noise = torch.randn(c, h, w, device=img_tensor.device) * noise_level
        
        # Apply noise with higher intensity in dark areas
        # (realistic camera behavior where dark areas have more noise)
        luminance = 0.299 * img_tensor[0] + 0.587 * img_tensor[1] + 0.114 * img_tensor[2] if c >= 3 else img_tensor[0]
        noise_mask = torch.exp(-4 * luminance)  # More noise in darker areas
        
        # Apply noise with mask
        for i in range(c):
            img_tensor[i] = img_tensor[i] + noise[i] * noise_mask
        
        # Ensure values are in valid range
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        return img_tensor
    
    def forward(self, img):
        """Apply exposure variation effect to pizza image"""
        if torch.is_tensor(img):
            img_tensor = img.clone()
        else:
            img_tensor = TVF.to_tensor(img)
        
        # Decide which effect to apply
        r = random.random()
        if r < self.underexposure_prob:
            # Under-exposure effect
            exposure_factor = random.uniform(
                self.exposure_range[0], 1.0)
            img_tensor = img_tensor * exposure_factor
            
            # Add color temperature variation (cooler/bluer for low light)
            if self.color_temp_variation and random.random() < 0.7:
                img_tensor = self._apply_color_temp(img_tensor, is_underexposed=True)
            
            # Add noise for low-light conditions
            if random.random() < self.noise_prob:
                noise_level = random.uniform(0.01, 0.05) * (1.0 - exposure_factor)
                img_tensor = self._apply_noise(img_tensor, noise_level)
                
        elif r < self.underexposure_prob + self.overexposure_prob:
            # Over-exposure effect
            exposure_factor = random.uniform(
                1.0, self.exposure_range[1])
            img_tensor = torch.clamp(img_tensor * exposure_factor, 0, 1)
            
            # Apply highlight recovery (similar to camera HDR)
            # Reduce contrast in bright areas to simulate highlight clipping
            if random.random() < 0.5:
                # Identify highlights (bright areas)
                highlights = img_tensor > 0.85
                
                # Compress highlights slightly to maintain some detail
                img_tensor = torch.where(
                    highlights,
                    0.85 + (img_tensor - 0.85) * 0.7,  # Compress values above 0.85
                    img_tensor
                )
            
            # Add color temperature variation (warmer/yellower for bright light)
            if self.color_temp_variation and random.random() < 0.7:
                img_tensor = self._apply_color_temp(img_tensor, is_underexposed=False)
        
        # Apply vignette with probability
        if random.random() < self.vignette_prob:
            img_tensor = self._apply_vignette(img_tensor)
        
        # Ensure values are in valid range
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        if torch.is_tensor(img):
            return img_tensor
        else:
            return TVF.to_pil_image(img_tensor)


class PerspectiveTransformEffect(nn.Module):
    """
    Advanced perspective transformations for pizza images,
    simulating various viewing angles and camera positions.
    Combines rotation, shearing, and perspective changes to create
    realistic variations in how the pizza might appear in photographs.
    """
    
    def __init__(self, 
                 rotation_range=(-30, 30),
                 shear_range=(-15, 15),
                 perspective_strength=(0.05, 0.3),
                 border_handling='reflect',
                 view_angle='random',
                 zoom_range=(0.9, 1.1)):
        super().__init__()
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.perspective_strength = perspective_strength
        self.border_handling = border_handling  # 'reflect', 'edge', 'black'
        self.view_angle = view_angle  # 'random', 'overhead', 'table', 'closeup', 'angle'
        self.zoom_range = zoom_range
    
    def forward(self, img):
        """Apply perspective transformation to pizza image"""
        if torch.is_tensor(img):
            # Convert to PIL for easier perspective transformation
            pil_img = TVF.to_pil_image(img.cpu() if img.device.type != 'cpu' else img)
        else:
            pil_img = img.copy()
        
        # Get image dimensions
        width, height = pil_img.size
        
        # Determine specific view angle configuration
        if self.view_angle == 'random':
            view = random.choice(['overhead', 'table', 'closeup', 'angle'])
        else:
            view = self.view_angle
        
        # Configure transformation parameters based on view angle
        if view == 'overhead':
            # Top-down view with slight variations
            rotation_range = (-10, 10)
            shear_range = (-5, 5)
            perspective_strength = (0.01, 0.1)
            zoom = random.uniform(0.95, 1.05)
        elif view == 'table':
            # Typical restaurant table-level view
            rotation_range = (-15, 15)
            shear_range = (-10, 10)
            perspective_strength = (0.1, 0.2)
            zoom = random.uniform(0.9, 1.0)
        elif view == 'closeup':
            # Close-up shot focusing on pizza details
            rotation_range = (-5, 5)
            shear_range = (-5, 5)
            perspective_strength = (0.05, 0.15)
            zoom = random.uniform(1.1, 1.2)  # Zoomed in
        else:  # 'angle'
            # More dramatic angled view
            rotation_range = (-25, 25)
            shear_range = (-12, 12)
            perspective_strength = (0.15, 0.25)
            zoom = random.uniform(0.85, 1.0)
        
        # Apply random zoom within configured range
        if zoom != 1.0:
            new_width, new_height = int(width * zoom), int(height * zoom)
            # Calculate crop
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = left + new_width
            bottom = top + new_height
            
            # Ensure crop is within bounds
            left = max(0, left)
            top = max(0, top)
            right = min(width, right)
            bottom = min(height, bottom)
            
            # Apply crop for zoom in, or pad for zoom out
            if zoom > 1.0:
                # Zoom in: crop
                pil_img = pil_img.crop((left, top, right, bottom))
                pil_img = pil_img.resize((width, height), Image.BICUBIC)
            else:
                # Zoom out: resize and pad
                small_img = pil_img.resize((new_width, new_height), Image.BICUBIC)
                new_img = Image.new(pil_img.mode, (width, height), (0, 0, 0))
                paste_left = (width - new_width) // 2
                paste_top = (height - new_height) // 2
                new_img.paste(small_img, (paste_left, paste_top))
                pil_img = new_img
        
        # Random rotation angle within configured range
        rotation_angle = random.uniform(rotation_range[0], rotation_range[1])
        
        # Random shear angles within configured range
        shear_x = random.uniform(shear_range[0], shear_range[1])
        shear_y = random.uniform(shear_range[0], shear_range[1])
        
        # Random perspective strength within configured range
        perspective_factor = random.uniform(
            perspective_strength[0], perspective_strength[1])
        
        # Compute perspective transform matrix
        # First apply rotation
        angle_rad = math.radians(rotation_angle)
        rotation_matrix = np.array([
            [math.cos(angle_rad), -math.sin(angle_rad), 0],
            [math.sin(angle_rad), math.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        
        # Apply shear
        shear_x_rad = math.radians(shear_x)
        shear_y_rad = math.radians(shear_y)
        shear_matrix = np.array([
            [1, math.tan(shear_x_rad), 0],
            [math.tan(shear_y_rad), 1, 0],
            [0, 0, 1]
        ])
        
        # Apply perspective distortion based on the view
        perspective_matrix = np.eye(3)
        
        # Scale perspective effect with image size
        effect_scale = min(width, height) * perspective_factor
        
        # Apply perspective transform based on view angle
        if view == 'overhead':
            # Minimal perspective for overhead shot
            corner_idx = random.randint(0, 3)
            offset = effect_scale * 0.5  # Reduced effect
        elif view == 'table':
            # More perspective effect at the further edge (top)
            corner_idx = random.randint(0, 1)  # Top corners
            offset = effect_scale
        elif view == 'closeup':
            # Random subtle perspective
            corner_idx = random.randint(0, 3)
            offset = effect_scale * 0.7
        else:  # 'angle'
            # Stronger perspective from one side
            corner_idx = random.randint(0, 3)
            offset = effect_scale * 1.2
        
        # Apply the perspective distortion to the selected corner
        if corner_idx == 0:  # Top-left
            perspective_matrix[0, 2] = random.uniform(0, offset)
            perspective_matrix[1, 2] = random.uniform(0, offset)
        elif corner_idx == 1:  # Top-right
            perspective_matrix[0, 2] = random.uniform(-offset, 0)
            perspective_matrix[1, 2] = random.uniform(0, offset)
        elif corner_idx == 2:  # Bottom-right
            perspective_matrix[0, 2] = random.uniform(-offset, 0)
            perspective_matrix[1, 2] = random.uniform(-offset, 0)
        else:  # Bottom-left
            perspective_matrix[0, 2] = random.uniform(0, offset)
            perspective_matrix[1, 2] = random.uniform(-offset, 0)
        
        # Compute combined transformation
        combined_matrix = rotation_matrix @ shear_matrix @ perspective_matrix
        
        # Convert to PIL's perspective transform format
        # (8 coefficients for projective transform)
        coeffs = (
            combined_matrix[0, 0], combined_matrix[0, 1], combined_matrix[0, 2],
            combined_matrix[1, 0], combined_matrix[1, 1], combined_matrix[1, 2],
            combined_matrix[2, 0], combined_matrix[2, 1]
        )
        
        # Apply transform
        transformed_img = pil_img.transform(
            (width, height),
            Image.PERSPECTIVE,
            coeffs,
            resample=Image.BICUBIC,
            fillcolor=(0, 0, 0) if self.border_handling == 'black' else None
        )
        
        # Handle borders if necessary
        if self.border_handling == 'reflect':
            # Apply reflection padding
            transformed_img = ImageOps.expand(transformed_img, border=10, fill=(0, 0, 0))
            transformed_img = transformed_img.crop((10, 10, width + 10, height + 10))
        elif self.border_handling == 'edge':
            # Use edge replication
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([0, 0, width, height], fill=255)
            transformed_mask = mask.transform(
                (width, height),
                Image.PERSPECTIVE,
                coeffs,
                resample=Image.BICUBIC,
                fillcolor=0
            )
            
            # Only keep the visible parts from transformed image
            combined = Image.new(transformed_img.mode, transformed_img.size, (0, 0, 0))
            combined.paste(transformed_img, (0, 0), transformed_mask)
            transformed_img = combined
        
        # Convert back to tensor if input was tensor
        if torch.is_tensor(img):
            transformed_tensor = TVF.to_tensor(transformed_img)
            if img.device.type != 'cpu':
                transformed_tensor = transformed_tensor.to(img.device)
            return transformed_tensor
        else:
            return transformed_img


# Utility function for shadow generation
def create_shadow_mask(img_size, num_shadows=3, shadow_dimension=0.5, blur_radius=20, 
                    shadow_type='random', direction=None):
    """
    Create a realistic shadow mask with random polygons or directional lighting shadows
    
    Args:
        img_size: Tuple of (width, height)
        num_shadows: Number of shadow polygons to generate
        shadow_dimension: Maximum size of shadow as fraction of image dimension
        blur_radius: Blur radius for shadow edges
        shadow_type: Type of shadow ('random', 'directional', 'edge', 'object')
        direction: Light direction for directional shadows ('top', 'bottom', 'left', 'right', None for random)
    
    Returns:
        PIL Image mask of shadows (grayscale)
    """
    width, height = img_size
    mask = Image.new('L', img_size, 255)
    draw = ImageDraw.Draw(mask)
    
    max_dim = max(width, height)
    
    # Choose shadow type if random
    if shadow_type == 'random':
        shadow_type = random.choice(['directional', 'edge', 'object'])
    
    if shadow_type == 'directional':
        # Directional shadows (like from window/overhead light)
        if direction is None:
            direction = random.choice(['top', 'bottom', 'left', 'right'])
        
        # Determine shadow parameters based on direction
        if direction == 'top':
            # Light coming from top
            start_y = 0
            end_y = int(height * shadow_dimension)
            start_x = random.randint(0, int(width * 0.3))
            end_x = random.randint(int(width * 0.7), width)
            
            # Create a gradient polygon for directional shadow
            shadow_val = random.randint(100, 180)
            points = [
                (start_x, start_y),
                (end_x, start_y),
                (width, end_y),
                (0, end_y)
            ]
            
        elif direction == 'bottom':
            # Light coming from bottom
            start_y = int(height * (1 - shadow_dimension))
            end_y = height
            start_x = random.randint(0, int(width * 0.3))
            end_x = random.randint(int(width * 0.7), width)
            
            shadow_val = random.randint(100, 180)
            points = [
                (start_x, end_y),
                (end_x, end_y),
                (width, start_y),
                (0, start_y)
            ]
            
        elif direction == 'left':
            # Light coming from left
            start_x = 0
            end_x = int(width * shadow_dimension)
            start_y = random.randint(0, int(height * 0.3))
            end_y = random.randint(int(height * 0.7), height)
            
            shadow_val = random.randint(100, 180)
            points = [
                (start_x, start_y),
                (end_x, 0),
                (end_x, height),
                (start_x, end_y)
            ]
            
        else:  # 'right'
            # Light coming from right
            start_x = int(width * (1 - shadow_dimension))
            end_x = width
            start_y = random.randint(0, int(height * 0.3))
            end_y = random.randint(int(height * 0.7), height)
            
            shadow_val = random.randint(100, 180)
            points = [
                (end_x, start_y),
                (start_x, 0),
                (start_x, height),
                (end_x, end_y)
            ]
        
        # Draw the directional shadow
        draw.polygon(points, fill=shadow_val)
        
    elif shadow_type == 'edge':
        # Shadows around edges of image (picture frame, vignette-like)
        shadow_width = int(max_dim * shadow_dimension * 0.5)
        shadow_val = random.randint(150, 200)
        
        # Choose which edges to shadow
        edges = []
        num_edges = random.randint(1, 4)
        possible_edges = ['top', 'bottom', 'left', 'right']
        random.shuffle(possible_edges)
        edges = possible_edges[:num_edges]
        
        for edge in edges:
            if edge == 'top':
                points = [
                    (0, 0),
                    (width, 0),
                    (width, shadow_width),
                    (0, shadow_width)
                ]
            elif edge == 'bottom':
                points = [
                    (0, height - shadow_width),
                    (width, height - shadow_width),
                    (width, height),
                    (0, height)
                ]
            elif edge == 'left':
                points = [
                    (0, 0),
                    (shadow_width, 0),
                    (shadow_width, height),
                    (0, height)
                ]
            else:  # 'right'
                points = [
                    (width - shadow_width, 0),
                    (width, 0),
                    (width, height),
                    (width - shadow_width, height)
                ]
            
            draw.polygon(points, fill=shadow_val)
    
    else:  # 'object' - shadows cast by objects
        for _ in range(num_shadows):
            # Random shadow intensity (0=black, 255=white)
            shadow_val = random.randint(100, 200)
            
            # Random number of points for polygon (3-6)
            num_points = random.randint(3, 6)
            
            # Generate random points for the shadow polygon
            # Anchor shadows to image edges for more realism
            points = []
            for _ in range(num_points):
                edge_anchor = random.random() < 0.5
                if edge_anchor:
                    # Anchor to an edge
                    edge = random.choice(['top', 'bottom', 'left', 'right'])
                    if edge == 'top':
                        points.append((random.randint(0, width), 0))
                    elif edge == 'bottom':
                        points.append((random.randint(0, width), height))
                    elif edge == 'left':
                        points.append((0, random.randint(0, height)))
                    else:  # 'right'
                        points.append((width, random.randint(0, height)))
                else:
                    # Random point within shadow dimension from edges
                    border = int(max_dim * shadow_dimension)
                    points.append((
                        random.randint(border, width - border),
                        random.randint(border, height - border)
                    ))
            
            # Draw the polygon shadow
            draw.polygon(points, fill=shadow_val)
    
    # Apply blur to soften shadow edges
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return mask
