import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import torch.nn as nn
import torch.nn.functional as TF
import torchvision.transforms as transforms
from torchvision.transforms import functional as TVF
from torch.utils.data import Dataset, DataLoader
import random
import gc
from contextlib import contextmanager
from tqdm import tqdm

# Gerät für Berechnungen festlegen
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Verwende Gerät: {device}")

@contextmanager
def open_image(path):
    """Sicheres Öffnen und Schließen von Bildern"""
    img = Image.open(path).convert('RGB')
    try:
        yield img
    finally:
        img.close()

def show_images(images, titles=None, cols=5, figsize=(15, 10)):
    """Zeigt mehrere Bilder an"""
    rows = len(images) // cols + (1 if len(images) % cols != 0 else 0)
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if isinstance(img, torch.Tensor):
            img = img.cpu().detach()
            if img.device != torch.device('cpu'):
                img = img.cpu()
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.axis('off')
        if titles is not None and i < len(titles):
            plt.title(titles[i])
    plt.tight_layout()
    plt.show()

def save_images(images, output_dir, base_filename, batch_size=16):
    """Speichert Bilder in Batches"""
    os.makedirs(output_dir, exist_ok=True)
    
    for batch_idx in range(0, len(images), batch_size):
        batch_end = min(batch_idx + batch_size, len(images))
        batch = images[batch_idx:batch_end]
        
        for i, img in enumerate(batch):
            idx = batch_idx + i
            if isinstance(img, torch.Tensor):
                if img.device != torch.device('cpu'):
                    img = img.cpu()
                img = img.detach()
                img = TVF.to_pil_image(img)
            elif isinstance(img, np.ndarray):
                img = Image.fromarray((img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8))
            
            img.save(os.path.join(output_dir, f"{base_filename}_{idx}.jpg"))
        
        del batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class PizzaBurningEffect(nn.Module):
    """Simuliert Verbrennungseffekte"""
    
    def __init__(self, burn_intensity_min=0.2, burn_intensity_max=0.8):
        super().__init__()
        self.burn_intensity_min = burn_intensity_min
        self.burn_intensity_max = burn_intensity_max
        
    def forward(self, img):
        if not isinstance(img, torch.Tensor):
            img = TVF.to_tensor(img).to(device)
        else:
            img = img.to(device)
        
        h, w = img.shape[1], img.shape[2]
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing='ij'
        )
        
        dist = torch.sqrt(x_coords**2 + y_coords**2)
        edge_weight = torch.exp(2 * (dist - 0.7))
        edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min())
        
        burn_intensity = random.uniform(self.burn_intensity_min, self.burn_intensity_max)
        
        spots_mask = torch.zeros_like(dist, device=device)
        num_spots = random.randint(3, 8)
        
        for _ in range(num_spots):
            spot_x = random.uniform(-1, 1)
            spot_y = random.uniform(-1, 1)
            spot_radius = random.uniform(0.05, 0.2)
            spot_intensity = random.uniform(0.5, 1.0)
            
            spot_dist = torch.sqrt((x_coords - spot_x)**2 + (y_coords - spot_y)**2)
            spot_mask = torch.exp(-spot_dist / spot_radius) * spot_intensity
            spots_mask = torch.maximum(spots_mask, spot_mask)
        
        burn_mask = torch.maximum(edge_weight, spots_mask) * burn_intensity
        burn_mask = burn_mask.unsqueeze(0).expand_as(img)
        
        darkening = img.clone()
        darkening = torch.stack([
            darkening[0] * 0.8,
            darkening[1] * 0.5,
            darkening[2] * 0.5
        ])
        
        result = img * (1 - burn_mask) + darkening * burn_mask
        return result

def augment_pizza(input_dir, output_dir, num_images=20, batch_size=16):
    """Hauptfunktion für die Pizza-Augmentierung"""
    print("Starte Pizza-Augmentierung...")
    
    # Check if input_dir exists
    if not os.path.exists(input_dir):
        print(f"Eingabeverzeichnis {input_dir} existiert nicht!")
        return
    
    # Check if input_dir has subdirectories (class folders)
    class_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    if class_dirs:
        # If we have class directories, collect images from each
        image_files = []
        for class_dir in class_dirs:
            class_path = os.path.join(input_dir, class_dir)
            class_images = [os.path.join(class_path, f) for f in os.listdir(class_path)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_files.extend(class_images)
    else:
        # If no subdirectories, get images directly from input_dir
        image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("Keine Bilder gefunden!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Augmentierungseffekte
    burning_effect = PizzaBurningEffect().to(device)
    
    # Transformationen
    transform = transforms.Compose([
        transforms.RandomRotation(180),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
    ])
    
    # Verarbeite jedes Bild
    for img_idx, img_path in enumerate(tqdm(image_files, desc="Verarbeite Bilder")):
        with open_image(img_path) as img:
            # Basis-Augmentierung
            for i in range(num_images // 2):
                augmented = transform(img)
                augmented = augmented.to(device)
                save_images([augmented], output_dir, f"pizza_{img_idx}_basic_{i}", batch_size)
            
            # Verbrennungseffekte
            for i in range(num_images // 2):
                augmented = transform(img)
                augmented = augmented.to(device)
                burnt = burning_effect(augmented)
                save_images([burnt], output_dir, f"pizza_{img_idx}_burnt_{i}", batch_size)
    
    print(f"Fertig! Bilder wurden in {output_dir} gespeichert.")

if __name__ == "__main__":
    input_dir = "augmented_pizza"  # Updated input directory
    output_dir = "augmented_pizza_legacy"  # Ausgabeverzeichnis
    num_images = 20  # Bilder pro Original
    batch_size = 16  # Batch-Größe
    
    augment_pizza(input_dir, output_dir, num_images, batch_size) 