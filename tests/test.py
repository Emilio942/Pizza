import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as TF  # Alias for functional operations
import torchvision.transforms as transforms
from torchvision.transforms import functional as TVF # Alias for functional transforms
from torch.utils.data import Dataset, DataLoader
import random
import gc
from contextlib import contextmanager
from tqdm import tqdm
import time # For non-GPU timing

# Gerät für Berechnungen festlegen - GPU wenn verfügbar
try:
    # Check CUDA availability and choose device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Verwende Gerät: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available(): # Check for Apple Metal Performance Shaders
        device = torch.device('mps')
        print("Verwende Gerät: Apple MPS")
    else:
        device = torch.device('cpu')
        print("Verwende Gerät: CPU")
except Exception as e:
    print(f"Fehler bei der Geräteerkennung: {e}. Fallback auf CPU.")
    device = torch.device('cpu')


# Abhängigkeiten prüfen und sicher importieren
try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    print("WARNUNG: SciPy nicht installiert - Gauß-Filter für Dampf-Effekt verwendet PyTorch-Alternative.")
    SCIPY_AVAILABLE = False

@contextmanager
def open_image(path):
    """Kontext-Manager für sicheres Öffnen und Schließen von Bildern"""
    try:
        img = Image.open(path).convert('RGB')
        try:
            yield img
        finally:
            img.close()
    except FileNotFoundError:
        print(f"FEHLER: Bilddatei nicht gefunden: {path}")
        yield None
    except Exception as e:
        print(f"FEHLER beim Öffnen von Bild {path}: {e}")
        yield None


class PizzaAugmentationDataset(Dataset):
    """Dataset-Klasse für Pizza-Bildaugmentierung mit PyTorch"""

    def __init__(self, image_paths, labels=None, transform=None, target_size=(224, 224)):
        self.image_paths = image_paths
        self.labels = labels  # 0: nicht verbrannt, 1: verbrannt
        self.transform = transform
        self.target_size = target_size # Define a target size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with open_image(image_path) as img:
            if img is None:
                 # Return dummy data or raise error if image loading failed
                 print(f"Warnung: Konnte Bild {image_path} nicht laden, überspringe.")
                 # Return a black tensor of the target size
                 return torch.zeros(3, *self.target_size), -1 if self.labels is not None else torch.zeros(3, *self.target_size)

            # Apply transforms if they exist
            if self.transform:
                try:
                    # PIL Image is expected by most torchvision transforms
                    image = self.transform(img)
                except Exception as e:
                    print(f"FEHLER bei Transformation von {image_path}: {e}")
                     # Return a black tensor on error
                    image = torch.zeros(3, *self.target_size)
            else:
                # Default transform: Resize and convert to tensor
                image = TVF.resize(img, list(self.target_size)) # Ensure size consistency
                image = TVF.to_tensor(image)

            # Ensure image is the correct size after transform
            if image.shape[1:] != self.target_size:
                 image = TF.interpolate(image.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)


        if self.labels is not None:
            label = self.labels[idx] if self.labels is not None else -1 # Use -1 for invalid label
            return image, label
        return image

def show_images(images, titles=None, cols=5, figsize=(15, 10)):
    """Zeigt mehrere Bilder in einem Grid an"""
    if not images:
        print("Keine Bilder zum Anzeigen vorhanden.")
        return

    rows = (len(images) + cols - 1) // cols # Correct calculation for rows
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if isinstance(img, torch.Tensor):
            # Konvertiere Tensor zu NumPy für die Anzeige
            img_np = img.detach().cpu() # Ensure on CPU and detached from graph
            # Permute dimensions if needed (C, H, W) -> (H, W, C)
            if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
                 img_np = img_np.permute(1, 2, 0).numpy()
            elif img_np.ndim == 2: # Handle grayscale
                 img_np = img_np.numpy()
            else:
                 print(f"Warnung: Unerwartete Tensorform für Bild {i}: {img.shape}")
                 plt.title("Fehlerhafte Form")
                 plt.axis('off')
                 continue

            # Normalisierte Bilder zurück zu [0,1] Range bringen
            img_np = np.clip(img_np, 0, 1)
            # Handle single-channel grayscale if needed
            if img_np.shape[-1] == 1:
                img_np = img_np.squeeze(-1) # Remove last dimension for grayscale display
            plt.imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)

        elif isinstance(img, np.ndarray):
            img_np = np.clip(img, 0, 1) if img.max() <= 1.0 else np.clip(img, 0, 255).astype(np.uint8)
            plt.imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
        elif isinstance(img, Image.Image):
             plt.imshow(img)
        else:
            print(f"Warnung: Unbekannter Bildtyp für Bild {i}: {type(img)}")
            plt.title("Unbekannter Typ")

        plt.axis('off')
        if titles is not None and i < len(titles):
            plt.title(titles[i])

    plt.tight_layout()
    plt.show()

def save_augmented_images(images, output_dir, base_filename, batch_size=16):
    """Speichert die augmentierten Bilder in Batches"""
    if not images:
        return 0 # Return count of saved images

    os.makedirs(output_dir, exist_ok=True)
    saved_count = 0

    # Verarbeite Bilder in Batches, um Speichernutzung zu reduzieren
    for batch_idx in range(0, len(images), batch_size):
        batch_start_time = time.time()
        batch_end = min(batch_idx + batch_size, len(images))
        batch = images[batch_idx:batch_end]

        img_objects_to_save = []

        # Convert images to PIL format first
        for i, img_data in enumerate(batch):
            idx = batch_idx + i
            pil_img = None
            try:
                if isinstance(img_data, torch.Tensor):
                    # Ensure tensor is on CPU, detached, and in range [0, 1] or [0, 255]
                    img_tensor = img_data.detach().cpu()
                    # Clamp values for safety before converting
                    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
                    pil_img = TVF.to_pil_image(img_tensor)
                elif isinstance(img_data, np.ndarray):
                    # Ensure correct data type and range for PIL
                    if img_data.max() <= 1.0 and img_data.min() >= 0.0:
                        img_to_save = (img_data * 255).astype(np.uint8)
                    else:
                        img_to_save = np.clip(img_data, 0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_to_save)
                elif isinstance(img_data, Image.Image):
                    pil_img = img_data # Already a PIL image
                else:
                     print(f"Warnung: Überspringe Speichern für unbekannten Datentyp: {type(img_data)}")
                     continue

                if pil_img:
                    img_objects_to_save.append((pil_img, os.path.join(output_dir, f"{base_filename}_{idx}.jpg")))

            except Exception as e:
                print(f"FEHLER bei der Konvertierung von Bild {idx} für das Speichern: {e}")
                # Optionally save a placeholder or log the error

        # Save the converted PIL images
        for pil_img, save_path in img_objects_to_save:
             try:
                 pil_img.save(save_path, quality=95) # Save as JPEG with quality
                 saved_count += 1
             except Exception as e:
                 print(f"FEHLER beim Speichern von Bild {save_path}: {e}")
             finally:
                 pil_img.close() # Close PIL image after saving

        # Explizit Speicher freigeben after processing the batch
        del batch
        del img_objects_to_save
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # print(f"Batch {batch_idx // batch_size + 1} saved in {time.time() - batch_start_time:.2f}s") # Optional timing per batch

    return saved_count


# Verbesserte Pizza-spezifische Augmentierungen mit GPU-Unterstützung
class PizzaBurningEffect(nn.Module):
    """Optimierte Simulation von Verbrennungseffekten für Pizza-Bilder"""

    def __init__(self, burn_intensity_min=0.2, burn_intensity_max=0.8):
        super().__init__()
        # Clamp intensities to avoid invalid values
        self.burn_intensity_min = max(0.0, burn_intensity_min)
        self.burn_intensity_max = min(1.0, burn_intensity_max)

    def forward(self, img_batch):
        # Expects a batch of images (B, C, H, W) on the target device
        if not isinstance(img_batch, torch.Tensor):
            raise TypeError("Input must be a PyTorch Tensor")
        if img_batch.ndim != 4:
             raise ValueError(f"Input tensor must be 4D (B, C, H, W), got {img_batch.ndim}D")
        if img_batch.device != device:
             img_batch = img_batch.to(device) # Ensure tensor is on the correct device

        b, c, h, w = img_batch.shape
        results = torch.empty_like(img_batch) # Pre-allocate result tensor

        # Create coordinate grid once for the batch size
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device, dtype=img_batch.dtype),
            torch.linspace(-1, 1, w, device=device, dtype=img_batch.dtype),
            indexing='ij'
        )
        # Calculate distance from center (shape H, W)
        dist_from_center = torch.sqrt(x_coords**2 + y_coords**2)

        # Calculate edge weight (shape H, W)
        edge_weight = torch.exp(2 * (dist_from_center - 0.7))
        # Normalize edge weight per image if needed, or globally
        edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min() + 1e-6) # Add epsilon for stability

        # Process each image in the batch
        for i in range(b):
            # --- Generate Mask for this specific image ---
            # Random burn intensity for this image
            burn_intensity = random.uniform(self.burn_intensity_min, self.burn_intensity_max)

            # Create spots mask (shape H, W) - Initialize for this image
            spots_mask = torch.zeros_like(dist_from_center, device=device)
            num_spots = random.randint(3, 8)

            for _ in range(num_spots):
                spot_x = random.uniform(-1, 1)
                spot_y = random.uniform(-1, 1)
                spot_radius = random.uniform(0.05, 0.2)
                spot_intensity = random.uniform(0.5, 1.0)

                spot_dist = torch.sqrt((x_coords - spot_x)**2 + (y_coords - spot_y)**2)
                # Gaussian falloff for spots
                spot_mask_single = torch.exp(-(spot_dist**2) / (2 * spot_radius**2)) * spot_intensity # Use Gaussian falloff
                spots_mask = torch.maximum(spots_mask, spot_mask_single)

            # Combine edge and spot effects for this image (shape H, W)
            burn_mask = torch.maximum(edge_weight, spots_mask) * burn_intensity
            # Clamp mask values to [0, 1]
            burn_mask = torch.clamp(burn_mask, 0.0, 1.0)

            # Expand mask to match channel dimension (1, H, W) -> (C, H, W)
            burn_mask_expanded = burn_mask.unsqueeze(0).expand(c, h, w)

            # --- Apply Burn Effect ---
            current_img = img_batch[i] # Get the i-th image (C, H, W)

            # Create darkening effect tensor (more brown/black)
            darkening = torch.stack([
                current_img[0] * 0.7,  # Reduziere Rot weniger
                current_img[1] * 0.4,  # Reduziere Grün stärker
                current_img[2] * 0.3   # Reduziere Blau am stärksten
            ], dim=0)

            # Apply the burn effect using the mask
            burnt_img = current_img * (1 - burn_mask_expanded) + darkening * burn_mask_expanded

            # Clamp final result to [0, 1] and store
            results[i] = torch.clamp(burnt_img, 0.0, 1.0)

        return results


class SimpleOvenEffect(nn.Module):
    """Optimierte Simulation von Ofeneffekten mit GPU-Unterstützung (Batch-fähig)"""

    def __init__(self, steam_prob=0.3, warmth_prob=0.3, shadow_prob=0.2):
        super().__init__()
        self.steam_prob = steam_prob
        self.warmth_prob = warmth_prob
        self.shadow_prob = shadow_prob

    def forward(self, img_batch):
        # Expects a batch of images (B, C, H, W) on the target device
        if not isinstance(img_batch, torch.Tensor):
            raise TypeError("Input must be a PyTorch Tensor")
        if img_batch.ndim != 4:
             raise ValueError(f"Input tensor must be 4D (B, C, H, W), got {img_batch.ndim}D")
        if img_batch.device != device:
             img_batch = img_batch.to(device)

        b, c, h, w = img_batch.shape
        result_batch = img_batch.clone() # Start with a copy

        # --- Steam Effect ---
        apply_steam = torch.rand(b, device=device) < self.steam_prob
        if torch.any(apply_steam):
            steam_indices = torch.where(apply_steam)[0]
            steam_opacity = torch.rand(len(steam_indices), 1, 1, 1, device=device) * 0.2 + 0.1 # (N_steam, 1, 1, 1)

            # Create gradient mask (H, W) -> (1, 1, H, W)
            y_coords = torch.linspace(1.0, 0.2, h, device=device, dtype=img_batch.dtype).view(1, 1, h, 1).expand(1, 1, h, w)
            # Generate random noise base (N_steam, 1, H, W)
            steam_base = torch.rand(len(steam_indices), 1, h, w, device=device, dtype=img_batch.dtype) * y_coords

            # Apply Gaussian Blur using PyTorch
            kernel_size = int(random.uniform(9, 25)) // 2 * 2 + 1  # Random odd kernel size
            sigma = random.uniform(5, 15)
            # Apply blur to the batch of steam bases
            steam_blurred = TVF.gaussian_blur(steam_base, kernel_size=kernel_size, sigma=sigma)

            # Create final steam mask (N_steam, 1, H, W)
            steam_mask = (steam_blurred * steam_opacity).expand(-1, c, -1, -1) # Expand to C channels

            # Apply steam effect only to selected images
            steam_color = torch.full_like(result_batch[steam_indices], 0.9, device=device) # Slightly grayish
            result_batch[steam_indices] = result_batch[steam_indices] * (1 - steam_mask) + steam_color * steam_mask

        # --- Warmth Effect ---
        apply_warmth = torch.rand(b, device=device) < self.warmth_prob
        if torch.any(apply_warmth):
            warmth_indices = torch.where(apply_warmth)[0]
            warmth_factor = torch.rand(len(warmth_indices), 1, 1, 1, device=device) * 0.1 + 0.05 # (N_warmth, 1, 1, 1)

            # Apply warmth: Increase Red, Decrease Blue slightly
            result_batch[warmth_indices, 0] = torch.clamp(result_batch[warmth_indices, 0] * (1 + warmth_factor), 0, 1) # Red
            result_batch[warmth_indices, 2] = torch.clamp(result_batch[warmth_indices, 2] * (1 - warmth_factor / 2), 0, 1) # Blue

        # --- Shadow Effect ---
        apply_shadow = torch.rand(b, device=device) < self.shadow_prob
        if torch.any(apply_shadow):
            shadow_indices = torch.where(apply_shadow)[0]
            n_shadow = len(shadow_indices)

            # Generate shadow parameters per image
            shadow_opacity = torch.rand(n_shadow, 1, 1, 1, device=device) * 0.3 + 0.1 # (N_shadow, 1, 1, 1)
            shadow_x = torch.randint(0, w, (n_shadow, 1, 1), device=device) # (N_shadow, 1, 1)
            shadow_y = torch.randint(0, h, (n_shadow, 1, 1), device=device) # (N_shadow, 1, 1)
            shadow_radius = torch.randint(w // 4, w // 2, (n_shadow, 1, 1), device=device).float() # (N_shadow, 1, 1)

            # Create coordinate grid (H, W) -> (1, H, W) for broadcasting
            y_coords, x_coords = torch.meshgrid(
                torch.arange(h, device=device, dtype=img_batch.dtype),
                torch.arange(w, device=device, dtype=img_batch.dtype),
                indexing='ij'
            )
            coords = torch.stack([y_coords, x_coords], dim=0).unsqueeze(0) # (1, 2, H, W)

            # Calculate distances for all shadow images simultaneously
            # Center coords need shape (N_shadow, 2, 1, 1) for broadcasting
            centers = torch.cat([shadow_y.unsqueeze(1), shadow_x.unsqueeze(1)], dim=1) # (N_shadow, 2, 1, 1)
            # Distances shape: (N_shadow, H, W)
            shadow_dist = torch.sqrt(torch.sum((coords - centers)**2, dim=1)).float()

            # Calculate shadow mask (N_shadow, H, W) -> (N_shadow, 1, H, W)
            shadow_mask = torch.exp(-shadow_dist / shadow_radius.squeeze(-1)) * shadow_opacity.squeeze(-1)
            shadow_mask = shadow_mask.unsqueeze(1).expand(-1, c, -1, -1) # Expand channels (N_shadow, C, H, W)

            # Apply shadow effect
            result_batch[shadow_indices] = result_batch[shadow_indices] * (1 - shadow_mask)


        # Clamp final result
        return torch.clamp(result_batch, 0.0, 1.0)


# 1. Optimierte Pizza-Basis-Augmentierung mit GPU-Unterstützung & Generator
def pizza_basic_augmentation_generator(image_paths, num_total_augmentations, batch_size=16, target_size=(224, 224)):
    """Grundlegende Augmentierung speziell für Pizza-Bilder mit Batch-Verarbeitung als Generator"""

    # Define transforms - apply ToTensor last
    basic_transform = transforms.Compose([
        transforms.RandomApply([transforms.RandomRotation(180)], p=0.8),
        transforms.RandomApply([transforms.RandomResizedCrop(target_size, scale=(0.7, 1.0), ratio=(0.9, 1.1), antialias=True)], p=0.7), # Use target_size
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # Applied together often
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(), # Converts PIL [0, 255] to Tensor [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Example normalization
    ])

    # Simple transform for loading if the main transform fails
    fallback_transform = transforms.Compose([
        transforms.Resize(target_size, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Estimate how many times to loop through the dataset
    num_originals = len(image_paths)
    if num_originals == 0: return
    loops_needed = (num_total_augmentations + num_originals - 1) // num_originals
    print(f"Basis Aug: Ziel={num_total_augmentations}, Originale={num_originals}, Durchläufe benötigt={loops_needed}")

    generated_count = 0
    augmented_batch = []

    for loop in range(loops_needed):
        print(f"Basis Aug: Durchlauf {loop+1}/{loops_needed}")
        # Use a standard dataset/loader for robust loading
        dataset = PizzaAugmentationDataset(image_paths, transform=None, target_size=target_size) # Load PIL first
        # Use num_workers > 0 for parallel loading if not bottlenecked by GPU/transforms
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=min(4, os.cpu_count() // 2), pin_memory=True if device != torch.device('cpu') else False)

        pbar = tqdm(total=min(num_originals, num_total_augmentations - generated_count), desc=f"Basis Aug {loop+1}")
        for img_pil in dataloader: # DataLoader loads PIL images now
             if isinstance(img_pil, (list, tuple)): # Handle case where label is returned
                 img_pil = img_pil[0]
             if img_pil is None or (isinstance(img_pil, torch.Tensor) and img_pil.numel() == 0):
                 pbar.update(1)
                 continue # Skip if image failed to load

             try:
                 # Apply the main transform (expects PIL)
                 augmented_tensor = basic_transform(TVF.to_pil_image(img_pil.squeeze(0))) # Remove batch dim, convert Tensor->PIL
             except Exception as e:
                 print(f"Warnung: Fehler bei Basis-Transformation (Bild {pbar.n}): {e}. Verwende Fallback.")
                 try:
                     # Ensure img_pil is PIL before fallback
                     pil_img_fallback = TVF.to_pil_image(img_pil.squeeze(0))
                     augmented_tensor = fallback_transform(pil_img_fallback)
                 except Exception as fe:
                     print(f"FEHLER: Fallback-Transformation fehlgeschlagen: {fe}")
                     pbar.update(1)
                     continue # Skip this image entirely

             augmented_batch.append(augmented_tensor.to(device)) # Move tensor to target device
             generated_count += 1
             pbar.update(1)

             if len(augmented_batch) >= batch_size:
                 yield torch.stack(augmented_batch) # Stack into a batch tensor
                 augmented_batch = []
                 # Optional: Add small sleep to prevent potential deadlocks with dataloader workers
                 # time.sleep(0.01)

             if generated_count >= num_total_augmentations:
                 pbar.close()
                 break # Stop generating if target count reached

        pbar.close()
        del dataloader # Explicitly delete dataloader and dataset
        del dataset
        gc.collect() # Clean up memory after each loop
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        if generated_count >= num_total_augmentations:
            break # Exit outer loop if target count reached

    # Yield any remaining images
    if augmented_batch:
        yield torch.stack(augmented_batch)

# 2. Optimierte Pizza-Verbrennungs-Augmentierung mit GPU-Unterstützung & Generator
def pizza_burning_augmentation_generator(image_paths, num_total_augmentations, batch_size=16, target_size=(224, 224)):
    """Fügt Verbrennungseffekte zu Pizza-Bildern hinzu (Generator, Batch-Verarbeitung)"""

    # Transformations applied before burning/oven effects
    pre_effect_transform = transforms.Compose([
        transforms.RandomApply([transforms.RandomRotation(180)], p=0.5),
        transforms.RandomApply([transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0), ratio=(0.95, 1.05), antialias=True)], p=0.5), # Less aggressive crop
        transforms.ToTensor(), # Convert PIL [0, 255] to Tensor [0, 1]
        # No normalization here, apply after effects if needed
    ])

    # Define effect modules (move to device)
    burning_effect = PizzaBurningEffect().to(device)
    oven_effect = SimpleOvenEffect().to(device) # Can be applied optionally

    # Estimate loops
    num_originals = len(image_paths)
    if num_originals == 0: return
    loops_needed = (num_total_augmentations + num_originals - 1) // num_originals
    print(f"Burn Aug: Ziel={num_total_augmentations}, Originale={num_originals}, Durchläufe benötigt={loops_needed}")

    generated_count = 0
    augmented_batch = []

    for loop in range(loops_needed):
        print(f"Burn Aug: Durchlauf {loop+1}/{loops_needed}")
        # Dataset loads PIL images
        dataset = PizzaAugmentationDataset(image_paths, transform=None, target_size=target_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=min(4, os.cpu_count() // 2), pin_memory=True if device != torch.device('cpu') else False)

        pbar = tqdm(total=min(num_originals * batch_size, num_total_augmentations - generated_count), desc=f"Burn Aug {loop+1}")

        for img_batch_pil in dataloader: # Loader provides batches of PIL images
            if isinstance(img_batch_pil, (list, tuple)): img_batch_pil = img_batch_pil[0]
            if img_batch_pil is None or (isinstance(img_batch_pil, torch.Tensor) and img_batch_pil.numel() == 0):
                 pbar.update(batch_size) # Update progress bar by batch size
                 continue

            # Apply pre-effect transforms to each image in the batch (on CPU is fine for transforms)
            # We need to apply ToTensor individually before stacking
            tensor_batch_list = []
            valid_indices = []
            for i, img_pil in enumerate(img_batch_pil):
                try:
                     # Ensure img_pil is PIL Image
                     pil_img = TVF.to_pil_image(img_pil.squeeze(0)) if isinstance(img_pil, torch.Tensor) else img_pil
                     img_tensor = pre_effect_transform(pil_img) # Output is Tensor [0, 1]
                     tensor_batch_list.append(img_tensor)
                     valid_indices.append(i)
                except Exception as e:
                     print(f"Warnung: Fehler bei Pre-Effect-Transformation (Bild {pbar.n + i}): {e}")

            if not tensor_batch_list:
                pbar.update(batch_size)
                continue # Skip if batch became empty

            # Stack valid tensors into a batch and move to device
            img_tensor_batch = torch.stack(tensor_batch_list).to(device)

            # Apply burning effect (expects batch on device)
            try:
                 burnt_batch = burning_effect(img_tensor_batch)
            except Exception as e:
                 print(f"FEHLER im BurningEffect: {e}. Überspringe Batch.")
                 pbar.update(len(tensor_batch_list))
                 del img_tensor_batch, tensor_batch_list
                 gc.collect(); torch.cuda.empty_cache()
                 continue


            # Optionally apply oven effect (expects batch on device)
            if random.random() < 0.6: # Apply oven effect to 60% of burnt images
                try:
                    burnt_batch = oven_effect(burnt_batch)
                except Exception as e:
                    print(f"FEHLER im OvenEffect: {e}.") # Continue with just burnt image


            # --- Post-Effect Normalization (Example) ---
            # If your model expects normalized input, apply it here
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            final_batch = normalize(burnt_batch)
            # --- ---

            # Add processed images to the yield batch
            augmented_batch.extend(final_batch.cpu()) # Move back to CPU for yielding/saving list
            generated_count += len(final_batch)
            pbar.update(len(final_batch))

            if len(augmented_batch) >= batch_size:
                 yield torch.stack(augmented_batch[:batch_size]).to(device) # Yield a batch on the target device
                 augmented_batch = augmented_batch[batch_size:] # Keep remainder

            if generated_count >= num_total_augmentations:
                 pbar.close()
                 break # Stop generating

        pbar.close()
        del dataloader
        del dataset
        gc.collect(); torch.cuda.empty_cache()

        if generated_count >= num_total_augmentations:
            break

    # Yield remaining images
    if augmented_batch:
        yield torch.stack(augmented_batch).to(device) # Yield final batch on target device


# 3. Optimiertes MixUp für Pizza-Bilder mit GPU-Unterstützung
def pizza_mixup(img_batch1, img_batch2, alpha=0.3):
    """Implementiert effizientes MixUp für Pizza-Bild-Batches"""
    # Ensure batches are tensors on the correct device
    if not isinstance(img_batch1, torch.Tensor): img_batch1 = torch.stack(img_batch1)
    if not isinstance(img_batch2, torch.Tensor): img_batch2 = torch.stack(img_batch2)

    img_batch1 = img_batch1.to(device)
    img_batch2 = img_batch2.to(device)

    # Ensure batches have the same size (B, C, H, W)
    if img_batch1.shape != img_batch2.shape:
        print(f"Warnung: MixUp-Batches haben unterschiedliche Formen ({img_batch1.shape} vs {img_batch2.shape}). Versuche Größenanpassung.")
        target_shape = img_batch1.shape
        try:
            img_batch2 = TF.interpolate(img_batch2, size=target_shape[2:], mode='bilinear', align_corners=False)
        except Exception as e:
            print(f"FEHLER: Größenanpassung für MixUp fehlgeschlagen: {e}. Überspringe MixUp.")
            return img_batch1 # Return first batch as fallback

    # Generate mixing parameters (lambda) for the batch from Beta distribution
    # Ensure lambda has shape (B, 1, 1, 1) for broadcasting
    lam = torch.tensor(np.random.beta(alpha, alpha, size=img_batch1.shape[0]), dtype=img_batch1.dtype, device=device)
    lam = lam.view(-1, 1, 1, 1) # Reshape for broadcasting

    # Mix the batches
    mixed_batch = lam * img_batch1 + (1 - lam) * img_batch2
    return torch.clamp(mixed_batch, 0.0, 1.0) # Clamp results


# 4. Optimiertes CutMix für Pizza-Bilder mit GPU-Unterstützung
def pizza_cutmix(img_batch1, img_batch2):
    """Implementiert effizientes CutMix für Pizza-Bild-Batches mit Keil- oder Kreisform"""
     # Ensure batches are tensors on the correct device
    if not isinstance(img_batch1, torch.Tensor): img_batch1 = torch.stack(img_batch1)
    if not isinstance(img_batch2, torch.Tensor): img_batch2 = torch.stack(img_batch2)

    img_batch1 = img_batch1.to(device)
    img_batch2 = img_batch2.to(device)

    # Ensure batches have the same size (B, C, H, W)
    if img_batch1.shape != img_batch2.shape:
        print(f"Warnung: CutMix-Batches haben unterschiedliche Formen ({img_batch1.shape} vs {img_batch2.shape}). Versuche Größenanpassung.")
        target_shape = img_batch1.shape
        try:
            img_batch2 = TF.interpolate(img_batch2, size=target_shape[2:], mode='bilinear', align_corners=False)
        except Exception as e:
            print(f"FEHLER: Größenanpassung für CutMix fehlgeschlagen: {e}. Überspringe CutMix.")
            return img_batch1

    b, c, h, w = img_batch1.shape
    result_batch = img_batch1.clone()

    # Generate coordinates grid once (H, W)
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=device, dtype=img_batch1.dtype),
        torch.arange(w, device=device, dtype=img_batch1.dtype),
        indexing='ij'
    )
    center_x, center_y = w // 2, h // 2

    # Generate masks for the entire batch
    masks = torch.zeros(b, h, w, dtype=torch.bool, device=device) # Boolean mask (B, H, W)

    # Decide shape per image
    use_circle = torch.rand(b) < 0.5

    # --- Circular Masks ---
    circle_indices = torch.where(use_circle)[0]
    if len(circle_indices) > 0:
        radii = torch.rand(len(circle_indices), device=device) * 0.4 + 0.3 # Random radius in [0.3, 0.7] of min(h, w)/2
        radii_pixels = radii * min(h, w) / 2
        dist_from_center = torch.sqrt(((x_coords - center_x)**2 + (y_coords - center_y)**2)) # (H, W)
        # Broadcast comparison: (N_circle, 1, 1) <= (H, W) -> (N_circle, H, W)
        masks[circle_indices] = dist_from_center.unsqueeze(0) <= radii_pixels.view(-1, 1, 1)

    # --- Wedge Masks ---
    wedge_indices = torch.where(~use_circle)[0]
    if len(wedge_indices) > 0:
        start_angles = torch.rand(len(wedge_indices), device=device) * 2 * np.pi # (N_wedge,)
        angle_widths = torch.rand(len(wedge_indices), device=device) * (np.pi / 2 - np.pi / 6) + np.pi / 6 # 30-90 degrees

        # Calculate angles relative to center (H, W)
        delta_x = (x_coords - center_x)
        delta_y = (y_coords - center_y)
        angles = torch.atan2(delta_y, delta_x)
        angles = torch.where(angles < 0, angles + 2 * np.pi, angles) # Normalize to [0, 2pi]

        # Broadcast angle comparison: (N_wedge, 1, 1) vs (H, W) -> (N_wedge, H, W)
        start = start_angles.view(-1, 1, 1)
        width = angle_widths.view(-1, 1, 1)
        end = start + width

        # Handle wrap-around case
        wrap_mask = end > (2 * np.pi)
        no_wrap_mask = ~wrap_mask

        wedge_mask = torch.zeros(len(wedge_indices), h, w, dtype=torch.bool, device=device)

        # No wrap-around condition
        wedge_mask[no_wrap_mask] = (angles >= start[no_wrap_mask]) & (angles <= end[no_wrap_mask])
        # Wrap-around condition
        wedge_mask[wrap_mask] = (angles >= start[wrap_mask]) | (angles <= (end[wrap_mask] % (2 * np.pi)))

        masks[wedge_indices] = wedge_mask

    # Expand mask to (B, C, H, W) and apply
    masks_expanded = masks.unsqueeze(1).expand(b, c, h, w)
    result_batch = torch.where(masks_expanded, img_batch2, result_batch)

    return result_batch


# 5. Optimierte Verbrennungsprogression mit GPU-Unterstützung
def pizza_burning_progression(img_batch, num_steps=5):
    """Erzeugt eine Reihe von Bildern mit kontrolliertem Verbrennungsgrad für einen Batch"""
    # Expects a batch (B, C, H, W) on the target device
    if not isinstance(img_batch, torch.Tensor): raise TypeError("Input must be a PyTorch Tensor")
    if img_batch.ndim != 4: raise ValueError(f"Input tensor must be 4D (B, C, H, W), got {img_batch.ndim}D")
    if img_batch.device != device: img_batch = img_batch.to(device)

    b = img_batch.shape[0]
    all_results = [img_batch.clone()] # Start with original batch

    # Reuse the same burn effect instance for efficiency
    burn_effect = PizzaBurningEffect(burn_intensity_min=0.0, burn_intensity_max=0.0).to(device)

    # Generate progressions for the batch
    for i in range(num_steps):
        # Calculate intensity for this step (increasing)
        intensity_min = 0.1 + (i * 0.15)
        intensity_max = 0.2 + (i * 0.15)
        # Clamp to valid range [0, 1]
        intensity_min = min(max(0.0, intensity_min), 1.0)
        intensity_max = min(max(0.0, intensity_max), 1.0)

        # Update the parameters of the existing module
        burn_effect.burn_intensity_min = intensity_min
        burn_effect.burn_intensity_max = intensity_max

        # Apply effect to the original batch (not cumulatively)
        try:
            # Apply to a clone of the original to avoid modifying it
            burnt_batch_step = burn_effect(img_batch.clone())
            all_results.append(burnt_batch_step)
        except Exception as e:
            print(f"FEHLER bei Brennstufe {i+1}: {e}. Breche Progression ab.")
            break # Stop progression if an error occurs

    # Concatenate results along a new dimension (e.g., Step, B, C, H, W)
    # Or flatten into (Step * B, C, H, W)
    # Flattening is usually more convenient for saving/further processing
    if len(all_results) > 1:
        flat_results = torch.cat(all_results, dim=0) # Stacks along batch dim: [orig, step1, step2,...]
        return flat_results
    else:
        return img_batch # Return original if no steps were successful


# 6. Optimierte Augmentierungspipeline mit GPU-Unterstützung und Generator-Pattern
def augment_pizza_dataset(input_dir, output_dir, total_augmentations, batch_size=16, target_size=(224, 224)):
    """Erzeugt einen erweiterten Datensatz für Pizza-Verbrennungserkennung mit GPU-Beschleunigung"""
    # Input validation
    if not os.path.isdir(input_dir):
        print(f"FEHLER: Eingabeverzeichnis '{input_dir}' nicht gefunden.")
        return
    if total_augmentations <= 0:
        print("FEHLER: 'total_augmentations' muss größer als 0 sein.")
        return

    # Sammle alle Bilddateien
    try:
        all_files = os.listdir(input_dir)
        image_files = [os.path.join(input_dir, f) for f in all_files
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))] # Added more extensions
    except Exception as e:
        print(f"FEHLER beim Lesen des Eingabeverzeichnisses '{input_dir}': {e}")
        return

    if not image_files:
        print(f"Keine gültigen Bilddateien in '{input_dir}' gefunden!")
        return
    print(f"Gefunden: {len(image_files)} Originalbilder.")

    os.makedirs(output_dir, exist_ok=True)

    # Strategie für die Bildverteilung (adjust percentages as needed)
    basic_count = int(total_augmentations * 0.4)  # 40% basic augmentations
    burning_count = int(total_augmentations * 0.3)  # 30% burning effects
    mixing_count = int(total_augmentations * 0.15) # 15% mixed images (MixUp/CutMix)
    progression_count = int(total_augmentations * 0.15) # 15% progression steps
    # Ensure total is roughly correct, adjust last category if needed
    total_planned = basic_count + burning_count + mixing_count + progression_count
    progression_count += total_augmentations - total_planned # Assign remainder to progression
    print(f"Plan: Basic={basic_count}, Burning={burning_count}, Mixing={mixing_count}, Progression={progression_count}")


    all_augmented_saved_count = 0
    global_batch_counter = 0 # Unique counter for saved batch filenames

    # --- 1. Grundlegende Augmentierung ---
    if basic_count > 0:
        print("\n--- Starte Basis-Augmentierung ---")
        basic_saved_count = 0
        try:
            basic_gen = pizza_basic_augmentation_generator(image_files, basic_count, batch_size, target_size)
            for batch_idx, basic_batch in enumerate(basic_gen):
                saved = save_augmented_images(basic_batch.cpu(), output_dir, f"pizza_basic_batch{global_batch_counter}", batch_size)
                basic_saved_count += saved
                all_augmented_saved_count += saved
                global_batch_counter += 1
                # Free GPU memory used by the batch
                del basic_batch
                gc.collect(); torch.cuda.empty_cache()
            print(f"Basis-Augmentierung: {basic_saved_count} Bilder gespeichert.")
        except Exception as e:
            print(f"FEHLER während der Basis-Augmentierung: {e}")
        finally:
            gc.collect(); torch.cuda.empty_cache() # Final cleanup for this section


    # --- 2. Verbrennungseffekte ---
    if burning_count > 0:
        print("\n--- Starte Verbrennungs-Augmentierung ---")
        burn_saved_count = 0
        try:
            burn_gen = pizza_burning_augmentation_generator(image_files, burning_count, batch_size, target_size)
            for batch_idx, burn_batch in enumerate(burn_gen):
                # Batch should already be on device, move to CPU for saving
                saved = save_augmented_images(burn_batch.cpu(), output_dir, f"pizza_burnt_batch{global_batch_counter}", batch_size)
                burn_saved_count += saved
                all_augmented_saved_count += saved
                global_batch_counter += 1
                del burn_batch
                gc.collect(); torch.cuda.empty_cache()
            print(f"Verbrennungs-Augmentierung: {burn_saved_count} Bilder gespeichert.")
        except Exception as e:
            print(f"FEHLER während der Verbrennungs-Augmentierung: {e}")
        finally:
             gc.collect(); torch.cuda.empty_cache()


    # --- 3. Gemischte Bilder (MixUp/CutMix) ---
    if mixing_count > 0 and len(image_files) >= 2:
        print("\n--- Starte MixUp/CutMix-Augmentierung ---")
        mix_saved_count = 0
        generated_mix_count = 0
        try:
            # Need two DataLoaders to draw pairs efficiently
            # Use a simple ToTensor + Resize transform for loading pairs
            mix_transform = transforms.Compose([
                transforms.Resize(target_size, antialias=True),
                transforms.ToTensor()
                # Maybe add normalization if effects depend on it, otherwise do it later
            ])
            dataset_mix = PizzaAugmentationDataset(image_files, transform=mix_transform, target_size=target_size)
            # Use persistent workers if OS supports it well, can speed up loading
            # Drop last ensures batches always have batch_size (important for pairing)
            loader1 = DataLoader(dataset_mix, batch_size=batch_size, shuffle=True, num_workers=min(4, os.cpu_count() // 2), pin_memory=True if device != torch.device('cpu') else False, drop_last=True, persistent_workers=True if os.name != 'nt' else False)
            loader2 = DataLoader(dataset_mix, batch_size=batch_size, shuffle=True, num_workers=min(4, os.cpu_count() // 2), pin_memory=True if device != torch.device('cpu') else False, drop_last=True, persistent_workers=True if os.name != 'nt' else False)
            mix_pbar = tqdm(total=mixing_count, desc="Mix Aug")

            num_batches_needed = (mixing_count + batch_size -1) // batch_size
            loaders_iter = zip(iter(loader1), iter(loader2)) # Combine loaders

            for i in range(num_batches_needed):
                 try:
                     batch1_data, batch2_data = next(loaders_iter)
                 except StopIteration:
                     print("Warnung: Dataloader für Mix/CutMix erschöpft. Erstelle neue Iteratoren.")
                     # Recreate iterators if one runs out faster (shouldn't happen with drop_last=True and same dataset size)
                     loaders_iter = zip(iter(loader1), iter(loader2))
                     try:
                         batch1_data, batch2_data = next(loaders_iter)
                     except StopIteration:
                          print("FEHLER: Konnte keine weiteren Batches für Mix/CutMix laden.")
                          break # Stop if can't get more batches


                 # Handle potential labels from dataset
                 batch1 = batch1_data[0] if isinstance(batch1_data, (list, tuple)) else batch1_data
                 batch2 = batch2_data[0] if isinstance(batch2_data, (list, tuple)) else batch2_data

                 # Move batches to device
                 batch1 = batch1.to(device)
                 batch2 = batch2.to(device)

                 # Apply MixUp or CutMix
                 if random.random() < 0.5:
                     mixed_batch = pizza_mixup(batch1, batch2, alpha=0.4) # More aggressive alpha for stronger mix
                 else:
                     mixed_batch = pizza_cutmix(batch1, batch2)

                 # --- Optional: Apply Post-Mix Normalization ---
                 # If normalization wasn't in mix_transform, apply it now
                 # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                 # final_mixed_batch = normalize(mixed_batch)
                 final_mixed_batch = mixed_batch # Assuming normalization happens elsewhere or isn't needed here

                 saved = save_augmented_images(final_mixed_batch.cpu(), output_dir, f"pizza_mixed_batch{global_batch_counter}", batch_size)
                 mix_saved_count += saved
                 all_augmented_saved_count += saved
                 generated_mix_count += len(final_mixed_batch)
                 global_batch_counter += 1
                 mix_pbar.update(len(final_mixed_batch))

                 del batch1, batch2, mixed_batch, final_mixed_batch
                 gc.collect(); torch.cuda.empty_cache()

                 if generated_mix_count >= mixing_count:
                     break # Stop if enough mixed images are generated

            mix_pbar.close()
            del loader1, loader2, dataset_mix # Cleanup loaders
            print(f"Mix/CutMix-Augmentierung: {mix_saved_count} Bilder gespeichert.")

        except Exception as e:
             print(f"FEHLER während der Mix/CutMix-Augmentierung: {e}")
        finally:
             gc.collect(); torch.cuda.empty_cache()


    # --- 4. Verbrennungsprogression ---
    if progression_count > 0:
        print("\n--- Starte Verbrennungs-Progression ---")
        prog_saved_count = 0
        generated_prog_count = 0
        try:
            num_steps = 5 # Number of burn levels per image (0 to 4) -> 5 images total per original
            # Calculate how many original images need progression to reach the target
            num_originals_for_prog = (progression_count + num_steps) // (num_steps + 1) # +1 for original
            num_originals_for_prog = min(num_originals_for_prog, len(image_files)) # Limit by available images

            if num_originals_for_prog > 0:
                selected_indices = np.random.choice(len(image_files), num_originals_for_prog, replace=False)
                selected_files = [image_files[i] for i in selected_indices]

                prog_transform = transforms.Compose([
                     transforms.Resize(target_size, antialias=True),
                     transforms.ToTensor() # Keep in [0, 1] range for burn effect
                ])
                prog_dataset = PizzaAugmentationDataset(selected_files, transform=prog_transform, target_size=target_size)
                # Process in batches for efficiency
                prog_loader = DataLoader(prog_dataset, batch_size=batch_size // (num_steps + 1), shuffle=False, num_workers=min(4, os.cpu_count() // 2), pin_memory=True if device != torch.device('cpu') else False)
                prog_pbar = tqdm(total=progression_count, desc="Progression Aug")

                for batch_data in prog_loader:
                    original_batch = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
                    original_batch = original_batch.to(device)

                    if original_batch.numel() == 0: continue # Skip empty batches

                    # Generate progression steps for the batch
                    progression_batch = pizza_burning_progression(original_batch, num_steps=num_steps) # Returns flattened batch

                    if progression_batch is None or progression_batch.numel() == 0: continue

                    # --- Optional: Apply Post-Progression Normalization ---
                    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    # final_prog_batch = normalize(progression_batch)
                    final_prog_batch = progression_batch

                    # Save the batch of progression images (includes originals and burnt steps)
                    # Note: save_augmented_images expects a list or batch tensor
                    saved = save_augmented_images(final_prog_batch.cpu(), output_dir, f"pizza_progression_batch{global_batch_counter}", batch_size)
                    prog_saved_count += saved
                    all_augmented_saved_count += saved
                    # Count only the newly generated burnt steps towards the target
                    generated_this_batch = len(final_prog_batch) - len(original_batch)
                    generated_prog_count += generated_this_batch
                    global_batch_counter += 1
                    prog_pbar.update(generated_this_batch)


                    del original_batch, progression_batch, final_prog_batch
                    gc.collect(); torch.cuda.empty_cache()

                    if generated_prog_count >= progression_count:
                        break # Stop if enough progression steps generated

                prog_pbar.close()
                del prog_loader, prog_dataset

            print(f"Verbrennungs-Progression: {prog_saved_count} Bilder gespeichert.")

        except Exception as e:
            print(f"FEHLER während der Verbrennungs-Progression: {e}")
        finally:
            gc.collect(); torch.cuda.empty_cache()


    print(f"\nInsgesamt {all_augmented_saved_count} augmentierte Pizza-Bilder erstellt.")
    print(f"Alle augmentierten Bilder wurden im Verzeichnis '{output_dir}' gespeichert.")

# Hauptfunktion
def main():
    print("="*50)
    print("Pizza-Verbrennungserkennung - Optimierte Datensatz-Augmentierung")
    print("="*50)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Verwendetes Gerät: {device}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Speicher: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # --- Konfiguration ---
    # *** WICHTIG: Pfade anpassen! ***
    input_dir = "/home/emilio/Documents/ai/pizza/img-pizza/"  # Verzeichnis mit Original-Pizza-Bildern
    output_dir = "augmented_pizza_output"  # Ausgabeverzeichnis (wird erstellt wenn nicht vorhanden)

    total_augmentations = 1000 # Gesamtanzahl der zu generierenden Bilder (ungefähres Ziel)
    batch_size = 32           # Batch-Größe für Verarbeitung & Speichern (abhängig von VRAM)
    target_size = (256, 256)  # Zielbildgröße für alle Augmentierungen
    # --- Ende Konfiguration ---

    # Überprüfe Eingabepfad frühzeitig
    if not os.path.isdir(input_dir):
        print(f"\nFEHLER: Eingabeverzeichnis '{input_dir}' existiert nicht.")
        print("Bitte passen Sie den Pfad in der 'main'-Funktion an.")
        return

    print(f"\nEinstellungen:")
    print(f"  Eingabeverzeichnis: {input_dir}")
    print(f"  Ausgabeverzeichnis: {output_dir}")
    print(f"  Ziel-Augmentierungen: {total_augmentations}")
    print(f"  Batch-Größe: {batch_size}")
    print(f"  Zielbildgröße: {target_size}")
    print("-"*50)

    # Starte die Augmentierung und Zeitmessung
    start_time_cpu = time.time()
    start_event_gpu = None
    if device.type == 'cuda':
        start_event_gpu = torch.cuda.Event(enable_timing=True)
        start_event_gpu.record()

    augment_pizza_dataset(input_dir, output_dir,
                         total_augmentations=total_augmentations,
                         batch_size=batch_size,
                         target_size=target_size)

    end_time_cpu = time.time()
    elapsed_time_cpu = end_time_cpu - start_time_cpu
    elapsed_time_gpu = None

    if start_event_gpu:
        end_event_gpu = torch.cuda.Event(enable_timing=True)
        end_event_gpu.record()
        torch.cuda.synchronize() # Wait for all GPU operations to complete
        elapsed_time_gpu = start_event_gpu.elapsed_time(end_event_gpu) / 1000.0 # Time in seconds

    print("\n" + "="*50)
    print("Augmentierung abgeschlossen!")
    print(f"Gesamtzeit (CPU): {elapsed_time_cpu:.2f} Sekunden")
    if elapsed_time_gpu is not None:
        print(f"Gesamtzeit (GPU): {elapsed_time_gpu:.2f} Sekunden")
    print(f"Ausgabe gespeichert in: {os.path.abspath(output_dir)}")
    print("="*50)

if __name__ == "__main__":
    main()