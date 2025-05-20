import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional  as TF
import torchvision.transforms as transforms
from torchvision.transforms import functional as TVF
from torch.utils.data import Dataset, DataLoader
import random
import gc
from contextlib import contextmanager
from tqdm import tqdm
from pathlib import Path

# Gerät für Berechnungen festlegen - GPU wenn verfügbar
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Verwende Gerät: {device}")

# Abhängigkeiten prüfen und sicher importieren
try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    print("SciPy nicht installiert - einige Funktionen werden eingeschränkt sein")
    SCIPY_AVAILABLE = False

@contextmanager
def open_image(path):
    """Kontext-Manager für sicheres Öffnen und Schließen von Bildern"""
    img = Image.open(path).convert('RGB')
    try:
        yield img
    finally:
        img.close()

class PizzaAugmentationDataset(Dataset):
    """Dataset-Klasse für Pizza-Bildaugmentierung mit PyTorch"""
    
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels  # 0: nicht verbrannt, 1: verbrannt
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with open_image(image_path) as img:
            if self.transform:
                image = self.transform(img)
            else:
                image = TVF.to_tensor(img)
        
        if self.labels is not None:
            return image, self.labels[idx]
        return image

def show_images(images, titles=None, cols=5, figsize=(15, 10)):
    """Zeigt mehrere Bilder in einem Grid an"""
    rows = len(images) // cols + (1 if len(images) % cols != 0 else 0)
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if isinstance(img, torch.Tensor):
            # Konvertiere Tensor zu NumPy für die Anzeige
            img = img.cpu().detach()
            if img.device != torch.device('cpu'):
                img = img.cpu()
            img = img.permute(1, 2, 0).numpy()
            # Normalisierte Bilder zurück zu [0,1] Range bringen
            img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.axis('off')
        if titles is not None and i < len(titles):
            plt.title(titles[i])
    plt.tight_layout()
    plt.show()

def save_augmented_images(images, output_dir, base_filename, batch_size=16):
    """Speichert die augmentierten Bilder in Batches"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Verarbeite Bilder in Batches, um Speichernutzung zu reduzieren
    for batch_idx in range(0, len(images), batch_size):
        batch_end = min(batch_idx + batch_size, len(images))
        batch = images[batch_idx:batch_end]
        
        for i, img in enumerate(batch):
            idx = batch_idx + i
            if isinstance(img, torch.Tensor):
                # Stelle sicher, dass das Bild auf CPU ist
                if img.device != torch.device('cpu'):
                    img = img.cpu()
                img = img.detach()
                # Konvertiere Tensor zu PIL für das Speichern
                img = TVF.to_pil_image(img)
            elif isinstance(img, np.ndarray):
                img = Image.fromarray((img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8))
            
            img.save(os.path.join(output_dir, f"{base_filename}_{idx}.jpg"))
        
        # Explizit Speicher freigeben
        del batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Verbesserte Pizza-spezifische Augmentierungen mit GPU-Unterstützung
class PizzaBurningEffect(nn.Module):
    """Optimierte Simulation von Verbrennungseffekten für Pizza-Bilder"""
    
    def __init__(self, burn_intensity_min=0.2, burn_intensity_max=0.8):
        super().__init__()
        self.burn_intensity_min = burn_intensity_min
        self.burn_intensity_max = burn_intensity_max
        
    def forward(self, img):
        # Stelle sicher, dass das Bild ein Tensor ist und auf dem richtigen Gerät liegt
        if not isinstance(img, torch.Tensor):
            img = TVF.to_tensor(img).to(device)
        else:
            img = img.to(device)
        
        # Erzeuge Koordinatengitter mit torch.meshgrid
        h, w = img.shape[1], img.shape[2]
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing='ij'
        )
        
        # Berechne Distanz vom Zentrum (vektorisiert)
        dist = torch.sqrt(x_coords**2 + y_coords**2)
        
        # Exponentieller Abfall vom Rand zur Mitte
        edge_weight = torch.exp(2 * (dist - 0.7))
        # Normalisiere zwischen 0 und 1
        edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min())
        
        # Zufällige Brenn-Intensität
        burn_intensity = random.uniform(self.burn_intensity_min, self.burn_intensity_max)
        
        # Erstelle Spots-Maske (vektorisiert)
        spots_mask = torch.zeros_like(dist, device=device)
        num_spots = random.randint(3, 8)
        
        # Erstelle alle Spotmasken gleichzeitig
        for _ in range(num_spots):
            spot_x = random.uniform(-1, 1)
            spot_y = random.uniform(-1, 1)
            spot_radius = random.uniform(0.05, 0.2)
            spot_intensity = random.uniform(0.5, 1.0)
            
            spot_dist = torch.sqrt((x_coords - spot_x)**2 + (y_coords - spot_y)**2)
            spot_mask = torch.exp(-spot_dist / spot_radius) * spot_intensity
            spots_mask = torch.maximum(spots_mask, spot_mask)
        
        # Kombiniere Rand- und Spot-Effekte
        burn_mask = torch.maximum(edge_weight, spots_mask) * burn_intensity
        
        # Erweitere für alle Kanäle
        burn_mask = burn_mask.unsqueeze(0).expand_as(img)
        
        # Verbrennungsbereiche werden dunkler und bräunlicher
        darkening = img.clone()
        darkening = torch.stack([
            darkening[0] * 0.8,  # Reduziere Rot weniger (für bräunlichen Effekt)
            darkening[1] * 0.5,  # Reduziere Grün stärker
            darkening[2] * 0.5   # Reduziere Blau stärker
        ])
        
        # Anwenden der Verbrennung (vektorisiert)
        result = img * (1 - burn_mask) + darkening * burn_mask
        
        return result

class SimpleOvenEffect(nn.Module):
    """Optimierte Simulation von Ofeneffekten mit GPU-Unterstützung"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, img):
        # Stelle sicher, dass das Bild ein Tensor ist und auf dem richtigen Gerät liegt
        if not isinstance(img, torch.Tensor):
            img = TVF.to_tensor(img).to(device)
        else:
            img = img.to(device)
        
        # Zufällig auswählen, welche Effekte angewendet werden
        effects = []
        if random.random() < 0.3:
            effects.append('steam')
        if random.random() < 0.3:
            effects.append('warmth')
        if random.random() < 0.2:
            effects.append('shadow')
        
        # Kopie nur erstellen, wenn Effekte angewendet werden
        result = img
        
        # Dampfeffekt - macht Teile des Bildes leicht heller und verwaschen
        if 'steam' in effects:
            h, w = img.shape[1], img.shape[2]
            steam_opacity = random.uniform(0.1, 0.3)
            
            # Erzeuge Dampfmaske mit PyTorch (vektorisiert)
            y_coords = torch.linspace(1.0, 0.2, h, device=device).view(-1, 1).expand(-1, w)
            steam_base = torch.rand(1, h, w, device=device) * y_coords
            
            # Glättung der Maske - entweder mit Faltung oder Gaußfilter
            if SCIPY_AVAILABLE:
                # Anmerkung: Hier müssen wir kurz zur CPU wechseln, da gaussian_filter für NumPy ist
                steam_base_np = steam_base.squeeze(0).cpu().numpy()
                steam_smooth_np = gaussian_filter(steam_base_np, sigma=random.uniform(5, 15))
                steam_mask = torch.tensor(steam_smooth_np, device=device).unsqueeze(0) * steam_opacity
            else:
                # Alternative: Verwende PyTorch's Faltungsoperationen für Glättung
                kernel_size = int(random.uniform(9, 25)) // 2 * 2 + 1  # Ungerade Zahl
                steam_mask = TVF.gaussian_blur(steam_base, kernel_size, sigma=random.uniform(5, 15)) * steam_opacity
            
            # Erweitere für alle Kanäle
            steam_mask = steam_mask.expand_as(img)
            
            # Heller machen wo Dampf ist (vektorisiert)
            steam_color = torch.ones_like(img, device=device) * 0.9  # Leicht gräulicher als weiß
            result = img * (1 - steam_mask) + steam_color * steam_mask
        
        # Wärmeeffekt - gibt dem Bild einen leicht wärmeren Farbton
        if 'warmth' in effects:
            warmth = random.uniform(0.05, 0.15)
            # Neue Tensor erstellen, um die Originalwerte zu behalten
            if result is img:
                result = img.clone()
            
            # Erhöhe Rot, reduziere Blau leicht (vektorisiert)
            result_channels = result.clone()
            result_channels[0] = torch.clamp(result[0] * (1 + warmth), 0, 1)
            result_channels[2] = torch.clamp(result[2] * (1 - warmth/2), 0, 1)
            result = result_channels
        
        # Schatteneffekt - optimiert mit PyTorch-Operationen
        if 'shadow' in effects:
            h, w = img.shape[1], img.shape[2]
            shadow_opacity = random.uniform(0.1, 0.4)
            
            # Neue Tensor erstellen, falls noch nicht geschehen
            if result is img:
                result = img.clone()
            
            # Erzeuge Koordinatengitter für den Schatten
            y_coords, x_coords = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing='ij'
            )
            
            # Zufällige Position für Schatten
            shadow_x = random.randint(0, w-1)
            shadow_y = random.randint(0, h-1)
            shadow_radius = random.randint(w//4, w//2)
            
            # Berechne Distanzmaske (vektorisiert)
            shadow_dist = torch.sqrt(((x_coords - shadow_x)**2 + (y_coords - shadow_y)**2).float())
            shadow_mask = torch.exp(-shadow_dist / shadow_radius) * shadow_opacity
            
            # Erweitere für alle Kanäle
            shadow_mask = shadow_mask.unsqueeze(0).expand_as(result)
            
            # Dunkler machen wo Schatten ist (vektorisiert)
            result = result * (1 - shadow_mask)
        
        return result

# 1. Optimierte Pizza-Basis-Augmentierung mit GPU-Unterstützung
def pizza_basic_augmentation(image_paths, num_per_image=10, batch_size=16):
    """Grundlegende Augmentierung speziell für Pizza-Bilder mit Batch-Verarbeitung"""
    
    # Grundlegende Transformationen definieren
    basic_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomRotation(180)  # Pizzas können in jeder Richtung liegen
        ], p=0.8),
        transforms.RandomApply([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1))
        ], p=0.7),
        transforms.RandomApply([
            transforms.RandomHorizontalFlip()
        ], p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        ], p=0.7),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.3),
        transforms.ToTensor(),
    ])
    
    # Dataset und DataLoader für effiziente Verarbeitung
    dataset = PizzaAugmentationDataset(image_paths, transform=basic_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    augmented_images = []
    
    # Für jedes Bild im Dataset
    for original_img in tqdm(dataloader, desc="Basis-Augmentierung"):
        original_img = original_img.to(device)
        
        # Erzeuge mehrere Varianten pro Bild
        for _ in range(num_per_image):
            augmented = basic_transform(TVF.to_pil_image(original_img.squeeze(0).cpu()))
            augmented = augmented.to(device)
            augmented_images.append(augmented)
            
            # Batch-weise Speicherfreigabe
            if len(augmented_images) >= batch_size:
                yield augmented_images
                augmented_images = []
                
                # Speicherbereinigung
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # Restliche Bilder zurückgeben
    if augmented_images:
        yield augmented_images

# 2. Optimierte Pizza-Verbrennungs-Augmentierung mit GPU-Unterstützung
def pizza_burning_augmentation(image_paths, num_per_image=10, batch_size=16):
    """Fügt Verbrennungseffekte zu Pizza-Bildern hinzu mit Batch-Verarbeitung"""
    
    # Instanzen der Transformationsmodule erstellen
    burning_effect = PizzaBurningEffect().to(device)
    oven_effect = SimpleOvenEffect().to(device)
    
    # Grundlegende Transformationen definieren
    base_transform = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomRotation(180)
        ], p=0.5),
        transforms.RandomApply([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1))
        ], p=0.5),
        transforms.ToTensor(),
    ])
    
    # Dataset und DataLoader für effiziente Verarbeitung
    dataset = PizzaAugmentationDataset(image_paths, transform=base_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    augmented_images = []
    
    # Für jedes Bild im Dataset
    for original_img in tqdm(dataloader, desc="Verbrennungs-Augmentierung"):
        original_img = original_img.to(device)
        
        # Erzeuge mehrere Varianten pro Bild
        for _ in range(num_per_image):
            img_tensor = original_img.squeeze(0)
            
            # Anwenden der Effekte
            img_tensor = burning_effect(img_tensor)
            
            if random.random() < 0.5:
                img_tensor = oven_effect(img_tensor)
            
            augmented_images.append(img_tensor)
            
            # Batch-weise Speicherfreigabe
            if len(augmented_images) >= batch_size:
                yield augmented_images
                augmented_images = []
                
                # Speicherbereinigung
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # Restliche Bilder zurückgeben
    if augmented_images:
        yield augmented_images

# 3. Optimiertes MixUp für Pizza-Bilder mit GPU-Unterstützung
def pizza_mixup(img1, img2, alpha=0.3):
    """Implementiert effizientes MixUp für Pizza-Bilder"""
    # Stelle sicher, dass beide Bilder Tensoren sind und auf dem richtigen Gerät liegen
    if not isinstance(img1, torch.Tensor):
        img1 = TVF.to_tensor(img1).to(device)
    else:
        img1 = img1.to(device)
    
    if not isinstance(img2, torch.Tensor):
        img2 = TVF.to_tensor(img2).to(device)
    else:
        img2 = img2.to(device)
    
    # Stelle sicher, dass beide Bilder die gleiche Größe haben
    if img1.shape != img2.shape:
        img2 = TF.interpolate(img2.unsqueeze(0), size=(img1.shape[1], img1.shape[2]), 
                             mode='bilinear', align_corners=False).squeeze(0)
    
    # Erzeuge Mischparameter
    lam = np.random.beta(alpha, alpha)
    
    # Mische die Bilder (vektorisiert)
    mixed = lam * img1 + (1 - lam) * img2
    
    return mixed

# 4. Optimiertes CutMix für Pizza-Bilder mit GPU-Unterstützung
def pizza_cutmix(img1, img2):
    """Implementiert effizientes CutMix für Pizza-Bilder mit Keil- oder Kreisform"""
    # Stelle sicher, dass beide Bilder Tensoren sind und auf dem richtigen Gerät liegen
    if not isinstance(img1, torch.Tensor):
        img1 = TVF.to_tensor(img1).to(device)
    else:
        img1 = img1.to(device)
    
    if not isinstance(img2, torch.Tensor):
        img2 = TVF.to_tensor(img2).to(device)
    else:
        img2 = img2.to(device)
    
    # Stelle sicher, dass beide Bilder die gleiche Größe haben
    if img1.shape != img2.shape:
        img2 = TF.interpolate(img2.unsqueeze(0), size=(img1.shape[1], img1.shape[2]), 
                             mode='bilinear', align_corners=False).squeeze(0)
    
    # Kopiere das erste Bild
    result = img1.clone()
    
    h, w = img1.shape[1], img1.shape[2]
    center_x, center_y = w // 2, h // 2
    
    # Erzeuge Koordinatengitter
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij'
    )
    
    # Entscheide zwischen Keil oder kreisförmigem Segment
    if random.random() < 0.5:
        # Kreisförmiges Segment (vektorisiert)
        radius = random.uniform(0.3, 0.7) * min(h, w) / 2
        dist = torch.sqrt(((x_coords - center_x)**2 + (y_coords - center_y)**2).float())
        mask = (dist <= radius)
    else:
        # Keilförmiges Segment (vektorisiert)
        start_angle = random.uniform(0, 2 * np.pi)
        angle_width = random.uniform(np.pi/6, np.pi/2)  # 30 bis 90 Grad
        
        # Berechne Winkel für alle Pixel gleichzeitig
        delta_x = (x_coords - center_x).float()
        delta_y = (y_coords - center_y).float()
        angles = torch.atan2(delta_y, delta_x)
        
        # Normalisiere Winkel auf [0, 2π]
        angles = torch.where(angles < 0, angles + 2 * np.pi, angles)
        
        # Maske für Winkelbereich
        if start_angle + angle_width <= 2 * np.pi:
            mask = (angles >= start_angle) & (angles <= start_angle + angle_width)
        else:
            # Behandle Überlauf
            mask = (angles >= start_angle) | (angles <= (start_angle + angle_width) % (2 * np.pi))
    
    # Erweitere Maske für alle Kanäle
    mask = mask.unsqueeze(0).expand_as(result)
    
    # Anwenden der Maske (vektorisiert)
    result = torch.where(mask, img2, result)
    
    return result

# 5. Optimierte Verbrennungsprogression mit GPU-Unterstützung
def pizza_burning_progression(img, num_steps=5):
    """Erzeugt eine Reihe von Bildern mit kontrolliertem Verbrennungsgrad"""
    # Stelle sicher, dass das Bild ein Tensor ist und auf dem richtigen Gerät liegt
    if not isinstance(img, torch.Tensor):
        img = TVF.to_tensor(img).to(device)
    else:
        img = img.to(device)
    
    results = [img.clone()]  # Originalbild als erstes
    
    # Erstelle Verbennungseffekte mit verschiedenen Intensitäten direkt vom Original
    # (Nicht kumulativ, um unrealistische Verbrennung zu vermeiden)
    for i in range(num_steps):
        # Zunehmender Verbrennungsgrad
        intensity_min = 0.1 + (i * 0.15)
        intensity_max = 0.2 + (i * 0.15)
        
        burn_effect = PizzaBurningEffect(
            burn_intensity_min=intensity_min,
            burn_intensity_max=intensity_max
        ).to(device)
        
        # Wende den Effekt direkt auf das Original an, nicht auf die vorherige Stufe
        burnt_img = burn_effect(img.clone())
        results.append(burnt_img)
    
    return results

# 6. Optimierte Augmentierungspipeline mit GPU-Unterstützung und Generator-Pattern
def augment_pizza_dataset(input_dir, output_dir, img_per_original=20, batch_size=16):
    """Erzeugt einen erweiterten Datensatz für Pizza-Verbrennungserkennung mit GPU-Beschleunigung"""
    # Sammle alle Bilddateien aus Unterverzeichnissen
    image_files = []
    input_path = Path(input_dir)
    
    # Check if input_dir exists
    if not os.path.exists(input_dir):
        print(f"Eingabeverzeichnis {input_dir} existiert nicht!")
        return
        
    # Check if input_dir has subdirectories (class folders)
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    if class_dirs:
        # If we have class directories, collect images from each
        for class_dir in class_dirs:
            class_images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
            image_files.extend([str(img_path) for img_path in class_images])
    else:
        # If no subdirectories, get images directly from input_dir
        image_files = [str(f) for f in input_path.glob("*.jpg")] + \
                      [str(f) for f in input_path.glob("*.jpeg")] + \
                      [str(f) for f in input_path.glob("*.png")]
    
    if not image_files:
        print("Keine Bilder gefunden!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Strategie für die Bildverteilung
    basic_count = int(img_per_original * 0.4)  # 40% grundlegende Augmentierung
    burning_count = int(img_per_original * 0.3)  # 30% Verbrennungseffekte
    mixing_count = int(img_per_original * 0.2)  # 20% gemischte Bilder
    progression_count = img_per_original - basic_count - burning_count - mixing_count  # Rest
    
    all_augmented_count = 0
    
    # 1. Grundlegende Augmentierung (mit Generator für Speichereffizienz)
    print("Führe grundlegende Pizza-Augmentierung durch...")
    basic_img_count = 0
    for batch_idx, basic_batch in enumerate(pizza_basic_augmentation(
                                            image_files, 
                                            num_per_image=basic_count//len(image_files) + 1, 
                                            batch_size=batch_size)):
        save_augmented_images(basic_batch, output_dir, f"pizza_basic_batch{batch_idx}", batch_size)
        basic_img_count += len(basic_batch)
        all_augmented_count += len(basic_batch)
        
        # Stoppe, wenn genug Bilder erzeugt wurden
        if basic_img_count >= basic_count:
            break
    
    print(f"Erzeugt: {basic_img_count} grundlegende Pizza-Bilder")
    
    # 2. Verbrennungseffekte (mit Generator für Speichereffizienz)
    print("Führe Pizza-Verbrennungs-Augmentierung durch...")
    burn_img_count = 0
    for batch_idx, burn_batch in enumerate(pizza_burning_augmentation(
                                          image_files, 
                                          num_per_image=burning_count//len(image_files) + 1, 
                                          batch_size=batch_size)):
        save_augmented_images(burn_batch, output_dir, f"pizza_burnt_batch{batch_idx}", batch_size)
        burn_img_count += len(burn_batch)
        all_augmented_count += len(burn_batch)
        
        # Stoppe, wenn genug Bilder erzeugt wurden
        if burn_img_count >= burning_count:
            break
    
    print(f"Erzeugt: {burn_img_count} verbrannte Pizza-Bilder")
    
    # 3. Gemischte Bilder - nur wenn genügend Bilder vorhanden sind
    if len(image_files) >= 2:
        print("Erzeuge gemischte Pizza-Bilder...")
        mixed_images = []
        mix_count = 0
        
        # Verarbeite die Bilder in Batches für Speichereffizienz
        for i in range(0, mixing_count, batch_size):
            batch_size_actual = min(batch_size, mixing_count - i)
            
            # Dataset für Bildpaare
            dataset = PizzaAugmentationDataset(
                image_files, 
                transform=transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor()
                ])
            )
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            
            # Sammle Bilder für MixUp/CutMix
            available_images = []
            for img in dataloader:
                available_images.append(img.squeeze(0).to(device))
                if len(available_images) >= 2 * batch_size_actual:
                    break
            
            # Für jedes Mischungspaar im Batch
            mixed_batch = []
            for j in range(batch_size_actual):
                if len(available_images) < 2:
                    break
                
                # Wähle zwei Bilder
                idx1, idx2 = random.sample(range(len(available_images)), 2)
                img1, img2 = available_images[idx1], available_images[idx2]
                
                # Abwechselnd MixUp und CutMix anwenden
                if random.random() < 0.5:
                    mixed = pizza_mixup(img1, img2, alpha=0.3)
                else:
                    mixed = pizza_cutmix(img1, img2)
                
                mixed_batch.append(mixed)
                mix_count += 1
            
            # Speichere und gib Speicher frei
            if mixed_batch:
                save_augmented_images(mixed_batch, output_dir, f"pizza_mixed_batch{i//batch_size}", batch_size)
                all_augmented_count += len(mixed_batch)
            
            # Explizit Speicher freigeben
            del available_images
            del mixed_batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"Erzeugt: {mix_count} gemischte Pizza-Bilder")
    
    # 4. Verbrennungsprogression für ausgewählte Bilder
    if progression_count > 0:
        print("Erzeuge Verbrennungsprogression...")
        progression_count_actual = 0
        
        # Wähle zufällig einige Bilder für die Progression
        selected_indices = np.random.choice(
            len(image_files), 
            min(progression_count // 5 + 1, len(image_files)), 
            replace=False
        )
        
        for batch_idx, idx_batch in enumerate(np.array_split(selected_indices, 
                                                          max(1, len(selected_indices) // batch_size))):
            progression_batch = []
            
            for idx in idx_batch:
                with open_image(image_files[idx]) as img:
                    img_tensor = TVF.to_tensor(img).to(device)
                    progression = pizza_burning_progression(img_tensor, num_steps=min(5, progression_count))
                    progression_batch.extend(progression)
                    progression_count_actual += len(progression)
            
            # Speichere und gib Speicher frei
            if progression_batch:
                save_augmented_images(progression_batch, output_dir, 
                                     f"pizza_progression_batch{batch_idx}", batch_size)
                all_augmented_count += len(progression_batch)
            
            # Explizit Speicher freigeben
            del progression_batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Stoppe, wenn genug Bilder erzeugt wurden
            if progression_count_actual >= progression_count:
                break
        
        print(f"Erzeugt: {progression_count_actual} Pizza-Progressionsbilder")
    
    print(f"Insgesamt {all_augmented_count} augmentierte Pizza-Bilder erstellt.")
    print(f"Alle augmentierten Bilder wurden im Verzeichnis '{output_dir}' gespeichert.")

# Hauptfunktion
def main():
    print("Pizza-Verbrennungserkennung - Optimierte Datensatz-Augmentierung")
    print(f"GPU verfügbar: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    input_dir = "augmented_pizza"  # Updated directory path
    output_dir = "augmented_pizza_output"  # Ausgabeverzeichnis
    
    # Konfiguration
    img_per_original = 20  # Anzahl generierter Bilder pro Original
    batch_size = 16        # Batch-Größe für effiziente Verarbeitung
    
    # Starte die Augmentierung
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    if start_time:
        start_time.record()
    
    augment_pizza_dataset(input_dir, output_dir, img_per_original=img_per_original, batch_size=batch_size)
    
    if end_time:
        end_time.record()
        torch.cuda.synchronize()
        print(f"Gesamtzeit: {start_time.elapsed_time(end_time) / 1000:.2f} Sekunden")
    
    print("Augmentierung abgeschlossen!")

if __name__ == "__main__":
    main()
