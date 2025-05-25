#!/usr/bin/env python3
from PIL import Image, ImageEnhance
import numpy as np

# Create a sample image for each class
classes = ['basic', 'burnt', 'combined', 'mixed', 'progression', 'segment']

for cls in classes:
    # Create a random color image (320x240)
    img_array = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    
    # For visual distinction, tint the images based on class
    if cls == 'basic':
        # More blue tint for basic
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.5, 0, 255)
    elif cls == 'burnt':
        # More red/black for burnt
        img_array = np.clip(img_array * 0.5, 0, 255).astype(np.uint8)
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.8, 0, 255)
    elif cls == 'combined':
        # Mixed colors for combined
        pass
    elif cls == 'mixed':
        # Yellow tint for mixed
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.3, 0, 255)
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.3, 0, 255)
    elif cls == 'progression':
        # Green tint for progression
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.5, 0, 255)
    elif cls == 'segment':
        # Purple tint for segment
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.3, 0, 255)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.3, 0, 255)
    
    # Create PIL image and save
    img = Image.fromarray(img_array)
    img.save(f'/home/emilio/Documents/ai/pizza/augmented_pizza/{cls}/sample_{cls}.jpg')
    
    # Erweiterung: Generierung von Bildern mit schlechten Lichtverhältnissen und ungewöhnlichen Blickwinkeln
    # Simuliere schlechte Lichtverhältnisse
    dark_img = ImageEnhance.Brightness(img).enhance(0.3)  # Reduziere Helligkeit auf 30%
    dark_img.save(f'/home/emilio/Documents/ai/pizza/augmented_pizza/{cls}/sample_{cls}_dark.jpg')

    # Simuliere ungewöhnliche Blickwinkel
    rotated_img = img.rotate(45)  # Drehe das Bild um 45 Grad
    rotated_img.save(f'/home/emilio/Documents/ai/pizza/augmented_pizza/{cls}/sample_{cls}_rotated.jpg')

    print(f"Created sample image for class '{cls}'")
    print(f"Created additional images for class '{cls}' with variations")
