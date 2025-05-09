import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import functional as TVF
from torch.utils.data import Dataset, DataLoader
import random
import gc
from contextlib import contextmanager
from tqdm import tqdm
import warnings
import json
from pathlib import Path

# Unterdrücke nicht kritische Warnungen
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ================ KONFIGURATION UND HILFSFUNKTIONEN ================

def parse_arguments():
    """Kommandozeilen-Argumente parsen"""
    parser = argparse.ArgumentParser(description='Pizza-Verbrennungserkennung - Erweiterte Datensatz-Augmentierung')
    parser.add_argument('--input_dir', type=str, default='./pizza_images', 
                        help='Verzeichnis mit Original-Pizza-Bildern')
    parser.add_argument('--output_dir', type=str, default='./augmented_pizza',
                        help='Ausgabeverzeichnis für augmentierte Bilder')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random Seed für Reproduzierbarkeit')
    parser.add_argument('--img_per_original', type=int, default=40,
                        help='Anzahl generierter Bilder pro Original')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch-Größe für effiziente Verarbeitung')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='GPU verwenden, wenn verfügbar')
    parser.add_argument('--save_stats', action='store_true', default=True,
                        help='Statistiken über die Augmentierung speichern')
    parser.add_argument('--show_samples', action='store_true', default=False,
                        help='Beispiel-Augmentierungen anzeigen')
    
    args = parser.parse_args()
    return args

def setup_environment(args):
    """Umgebung einrichten (Gerät, Random Seed)"""
    # Gerät festlegen
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Verwende GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Verwende CPU")
    
    # Random Seed setzen, wenn angegeben
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Random Seed gesetzt: {args.seed}")
    
    # Prüfen, ob SciPy verfügbar ist
    try:
        from scipy.ndimage import gaussian_filter
        scipy_available = True
        print("SciPy ist verfügbar - erweiterte Filter werden genutzt")
    except ImportError:
        scipy_available = False
        print("SciPy nicht installiert - verwende PyTorch-basierte Alternativen")
    
    return device, scipy_available

def validate_and_prepare_paths(args):
    """Überprüft und bereitet Pfade vor"""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Prüfen, ob Eingabeverzeichnis existiert
    if not input_dir.exists():
        raise FileNotFoundError(f"Eingabeverzeichnis nicht gefunden: {input_dir}")
    
    # Ausgabeverzeichnis erstellen, falls es nicht existiert
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Unterverzeichnisse für verschiedene Augmentierungstypen
    subdirs = {
        'basic': output_dir / 'basic',
        'burnt': output_dir / 'burnt',
        'mixed': output_dir / 'mixed',
        'progression': output_dir / 'progression',
        'segment': output_dir / 'segment',
        'combined': output_dir / 'combined'
    }
    
    for subdir in subdirs.values():
        subdir.mkdir(exist_ok=True)
    
    return input_dir, output_dir, subdirs

def get_image_files(input_dir):
    """Sammelt alle Bilddateien im Eingabeverzeichnis"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for file in input_dir.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            try:
                # Öffne das Bild kurz, um sicherzustellen, dass es gültig ist
                with Image.open(file) as img:
                    img.verify()  # Validiere das Bild
                image_files.append(file)
            except (IOError, SyntaxError) as e:
                print(f"Warnung: Kann Bild nicht verarbeiten {file}: {e}")
    
    if not image_files:
        raise ValueError(f"Keine gültigen Bilder im Verzeichnis gefunden: {input_dir}")
    
    print(f"Gefunden: {len(image_files)} gültige Bilder")
    return image_files

@contextmanager
def open_image(path):
    """Kontext-Manager für sicheres Öffnen und Schließen von Bildern"""
    try:
        img = Image.open(path).convert('RGB')
        yield img
    except Exception as e:
        print(f"Fehler beim Öffnen von {path}: {e}")
        raise
    finally:
        if 'img' in locals():
            img.close()

def get_optimal_batch_size(suggested_batch_size, device):
    """Ermittelt eine optimale Batch-Größe basierend auf verfügbarem Speicher"""
    if device.type == 'cuda':
        # Konservative Schätzung des verfügbaren GPU-Speichers
        total_memory = torch.cuda.get_device_properties(0).total_memory
        already_used = torch.cuda.memory_allocated(0)
        available = total_memory - already_used
        
        # Nehmen wir an, dass ein typisches Bild ~2MB im Speicher belegt
        estimated_image_size = 2 * 1024 * 1024  # 2MB in Bytes
        
        # Verlasse 20% Spielraum
        safe_memory = available * 0.8
        max_images = int(safe_memory / estimated_image_size)
        
        # Begrenze die Batch-Größe auf einen sinnvollen Bereich
        optimal_batch_size = max(4, min(max_images, suggested_batch_size))
        
        # Runde auf die nächste Zweierpotenz ab
        optimal_batch_size = 2 ** int(np.log2(optimal_batch_size))
        
        print(f"Optimale Batch-Größe für GPU: {optimal_batch_size}")
        return optimal_batch_size
    else:
        # Für CPU: Verwende einen konservativen Wert
        return min(8, suggested_batch_size)

class AugmentationStats:
    """Klasse zum Verfolgen und Speichern von Augmentierungsstatistiken"""
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
        """Aktualisiert die Statistiken für einen Augmentierungstyp"""
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
        """Setzt die Anzahl der Eingabebilder"""
        self.stats["input_images"] = count
    
    def save(self):
        """Speichert die Statistiken in eine JSON-Datei"""
        self.stats["duration_seconds"] = time.time() - self.start_time
        
        # Berechne nützliche Metriken
        if self.stats["input_images"] > 0:
            self.stats["images_per_original"] = self.stats["output_images"] / self.stats["input_images"]
        
        # Speichere als JSON
        stats_file = self.output_dir / "augmentation_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"Statistiken gespeichert in {stats_file}")

# ================ DATASET UND UTILITY KLASSEN ================

class PizzaAugmentationDataset(Dataset):
    """Erweiterte Dataset-Klasse für Pizza-Bildaugmentierung"""
    
    def __init__(self, image_paths, transform=None, device=None, cache_size=0):
        self.image_paths = image_paths
        self.transform = transform
        self.device = device
        
        # Optionaler Bildercache für häufig verwendete Bilder
        self.cache_size = min(cache_size, len(image_paths))
        self.cache = {}
        
        if self.cache_size > 0:
            print(f"Initialisiere Bildercache mit {self.cache_size} Bildern...")
            for i in range(self.cache_size):
                self._load_and_cache(i)
    
    def _load_and_cache(self, idx):
        """Lädt ein Bild und speichert es im Cache"""
        if idx in self.cache:
            return
        
        path = self.image_paths[idx]
        try:
            with open_image(path) as img:
                # Speichere als Tensor direkt im Cache
                tensor = TVF.to_tensor(img)
                if self.device:
                    tensor = tensor.to(self.device)
                self.cache[idx] = tensor
        except Exception as e:
            print(f"Fehler beim Cachen von Bild {path}: {e}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Versuche zuerst, aus dem Cache zu laden
        if idx in self.cache:
            image = self.cache[idx]
        else:
            # Wenn nicht im Cache, lade aus Datei
            try:
                with open_image(self.image_paths[idx]) as img:
                    image = TVF.to_tensor(img)
                    if self.device:
                        image = image.to(self.device)
            except Exception as e:
                # Fehlerbehandlung - gib ein schwarzes Bild zurück
                print(f"Fehler beim Laden von Bild {self.image_paths[idx]}: {e}")
                image = torch.zeros(3, 224, 224)
                if self.device:
                    image = image.to(self.device)
        
        # Wende Transformation an, wenn vorhanden
        if self.transform:
            try:
                # Wenn das Bild ein Tensor ist und die Transformation PIL erwartet
                if isinstance(image, torch.Tensor) and not isinstance(self.transform, transforms.Compose):
                    # Konvertiere zu PIL für nicht-Tensor-Transformation
                    pil_image = TVF.to_pil_image(image.cpu() if self.device else image)
                    image = self.transform(pil_image)
                    if self.device and isinstance(image, torch.Tensor):
                        image = image.to(self.device)
                else:
                    # Direkte Transformation
                    image = self.transform(image)
            except Exception as e:
                print(f"Fehler bei Transformation von Bild {self.image_paths[idx]}: {e}")
        
        return image

def show_images(images, titles=None, cols=5, figsize=(15, 10), save_path=None):
    """Zeigt mehrere Bilder in einem Grid an mit optionalem Speichern"""
    if not images:
        print("Keine Bilder zum Anzeigen vorhanden")
        return
    
    rows = len(images) // cols + (1 if len(images) % cols != 0 else 0)
    plt.figure(figsize=figsize)
    
    for i, img in enumerate(images):
        if i >= rows * cols:
            break
            
        plt.subplot(rows, cols, i + 1)
        
        if isinstance(img, torch.Tensor):
            # Konvertiere Tensor zu NumPy für die Anzeige
            img = img.cpu().detach()
            if img.device != torch.device('cpu'):
                img = img.cpu()
            img = img.permute(1, 2, 0).numpy()
            # Normalisiere auf [0,1] falls nötig
            img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.axis('off')
        if titles is not None and i < len(titles):
            plt.title(titles[i])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Bild gespeichert unter: {save_path}")
    
    plt.show()

def save_augmented_images(images, output_dir, base_filename, batch_size=16, metadata=None):
    """Speichert die augmentierten Bilder in Batches mit einheitlicher Größe"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    TARGET_SIZE = 224  # Feste Zielgröße für alle Bilder
    
    # Verarbeite Bilder in Batches
    for batch_idx in range(0, len(images), batch_size):
        batch_end = min(batch_idx + batch_size, len(images))
        batch = images[batch_idx:batch_end]
        
        for i, img in enumerate(batch):
            idx = batch_idx + i
            output_path = output_dir / f"{base_filename}_{idx:04d}.jpg"
            
            try:
                if isinstance(img, torch.Tensor):
                    # Stelle sicher, dass das Bild auf CPU ist
                    if img.device != torch.device('cpu'):
                        img = img.cpu()
                    img = img.detach()
                    
                    # Normalisiere auf [0,1] falls nötig
                    if img.min() < 0 or img.max() > 1:
                        img = torch.clamp(img, 0, 1)
                    
                    # Stelle einheitliche Größe sicher
                    current_size = img.shape[1:3]
                    if current_size[0] != TARGET_SIZE or current_size[1] != TARGET_SIZE:
                        img = F.interpolate(img.unsqueeze(0), size=(TARGET_SIZE, TARGET_SIZE), 
                                           mode='bilinear', align_corners=False).squeeze(0)
                    
                    # Konvertiere Tensor zu PIL für das Speichern
                    img = TVF.to_pil_image(img)
                elif isinstance(img, np.ndarray):
                    # Behandle NumPy-Arrays
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    img = Image.fromarray(img.astype(np.uint8))
                    
                    # Stelle einheitliche Größe sicher
                    if img.width != TARGET_SIZE or img.height != TARGET_SIZE:
                        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
                elif isinstance(img, Image.Image):
                    # Stelle einheitliche Größe für PIL-Bilder sicher
                    if img.width != TARGET_SIZE or img.height != TARGET_SIZE:
                        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
                
                # Speichern mit Qualitätseinstellungen
                img.save(output_path, quality=92, optimize=True)
                
                # Speichere Metadaten, wenn vorhanden
                if metadata:
                    meta_path = output_path.with_suffix('.json')
                    with open(meta_path, 'w') as f:
                        json.dump(metadata, f)
            except Exception as e:
                print(f"Fehler beim Speichern von Bild {output_path}: {e}")
        
        # Explizit Speicher freigeben
        del batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return len(images)
# def save_augmented_images(images, output_dir, base_filename, batch_size=16, metadata=None):
#     """Speichert die augmentierten Bilder in Batches mit optionalen Metadaten"""
#     output_dir = Path(output_dir)
#     output_dir.mkdir(exist_ok=True)
    
#     # Verarbeite Bilder in Batches, um Speichernutzung zu reduzieren
#     for batch_idx in range(0, len(images), batch_size):
#         batch_end = min(batch_idx + batch_size, len(images))
#         batch = images[batch_idx:batch_end]
        
#         for i, img in enumerate(batch):
#             idx = batch_idx + i
#             output_path = output_dir / f"{base_filename}_{idx:04d}.jpg"
            
#             try:
#                 if isinstance(img, torch.Tensor):
#                     # Stelle sicher, dass das Bild auf CPU ist
#                     if img.device != torch.device('cpu'):
#                         img = img.cpu()
#                     img = img.detach()
                    
#                     # Normalisiere auf [0,1] falls nötig
#                     if img.min() < 0 or img.max() > 1:
#                         img = torch.clamp(img, 0, 1)
                    
#                     # Konvertiere Tensor zu PIL für das Speichern
#                     img = TVF.to_pil_image(img)
#                 elif isinstance(img, np.ndarray):
#                     # Behandle NumPy-Arrays
#                     if img.max() <= 1.0:
#                         img = (img * 255).astype(np.uint8)
#                     img = Image.fromarray(img.astype(np.uint8))
                
#                 # Speichern mit Qualitätseinstellungen
#                 img.save(output_path, quality=92, optimize=True)
                
#                 # Speichere Metadaten, wenn vorhanden
#                 if metadata:
#                     meta_path = output_path.with_suffix('.json')
#                     with open(meta_path, 'w') as f:
#                         json.dump(metadata, f)
#             except Exception as e:
#                 print(f"Fehler beim Speichern von Bild {output_path}: {e}")
        
#         # Explizit Speicher freigeben
#         del batch
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
    
#     return len(images)

# ================ ERWEITERTE AUGMENTIERUNGSMODULE ================

class EnhancedPizzaBurningEffect(nn.Module):
    """Stark verbesserte Simulation von Verbrennungseffekten für Pizza-Bilder"""
    
    def __init__(self, 
                 burn_intensity_min=0.2, 
                 burn_intensity_max=0.8, 
                 burn_pattern='random',
                 color_variation=True):
        super().__init__()
        self.burn_intensity_min = burn_intensity_min
        self.burn_intensity_max = burn_intensity_max
        self.burn_pattern = burn_pattern  # 'random', 'edge', 'spot', 'streak'
        self.color_variation = color_variation
    
    def _create_edge_burn_mask(self, h, w, device):
        """Erstellt eine Maske für Randverbrennung"""
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing='ij'
        )
        
        # Distanz vom Zentrum
        dist = torch.sqrt(x_coords**2 + y_coords**2)
        
        # Zufällige Verzerrung für unregelmäßige Ränder
        noise = torch.randn(h, w, device=device) * 0.05
        dist = dist + noise
        
        # Exponentieller Abfall vom Rand zur Mitte mit zufälliger Schwelle
        threshold = random.uniform(0.6, 0.8)
        edge_weight = torch.exp(3 * (dist - threshold))
        
        # Normalisiere zwischen 0 und 1
        edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min())
        
        # Füge zufällige Variationen hinzu
        if random.random() < 0.3:
            # Asymmetrische Verbrennung - mehr auf einer Seite
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
        """Erstellt eine Maske für Fleckenverbrennung"""
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing='ij'
        )
        
        spots_mask = torch.zeros(h, w, device=device)
        num_spots = random.randint(3, 10)
        
        # Erstelle Spots mit unterschiedlicher Intensität und Form
        for _ in range(num_spots):
            # Zufällige Position, leicht zum Rand tendierend
            r = random.uniform(0.3, 1.0)  # Radius vom Zentrum
            theta = random.uniform(0, 2*np.pi)  # Winkel
            spot_x = r * np.cos(theta)
            spot_y = r * np.sin(theta)
            
            # Verschiedene Formen durch Skalierung der Achsen
            if random.random() < 0.3:
                # Elliptisch statt kreisförmig
                x_scale = random.uniform(0.5, 1.5)
                y_scale = random.uniform(0.5, 1.5)
                spot_dist = torch.sqrt(((x_coords - spot_x)/x_scale)**2 + 
                                        ((y_coords - spot_y)/y_scale)**2)
            else:
                # Kreisförmig
                spot_dist = torch.sqrt((x_coords - spot_x)**2 + (y_coords - spot_y)**2)
            
            # Parameter für den Spot
            spot_radius = random.uniform(0.05, 0.25)
            spot_intensity = random.uniform(0.6, 1.0)
            spot_falloff = random.uniform(1.0, 3.0)
            
            # Erzeuge Spot mit unterschiedlichen Profilen
            if random.random() < 0.5:
                # Exponentielles Profil
                spot_mask = torch.exp(-spot_dist * spot_falloff / spot_radius) * spot_intensity
            else:
                # Quadratisches Profil für schärfere Kanten
                normalized_dist = spot_dist / spot_radius
                spot_mask = torch.maximum(torch.zeros_like(normalized_dist), 
                                          (1 - normalized_dist**2)) * spot_intensity
            
            # Kombiniere mit bestehender Maske (Maximum für überlappende Bereiche)
            spots_mask = torch.maximum(spots_mask, spot_mask)
        
        return spots_mask
    
    def _create_streak_burn_mask(self, h, w, device):
        """Erstellt eine Maske für streifenförmige Verbrennung"""
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing='ij'
        )
        
        streaks_mask = torch.zeros(h, w, device=device)
        num_streaks = random.randint(1, 4)
        
        for _ in range(num_streaks):
            # Definiere eine Linie mit zufälliger Orientierung
            theta = random.uniform(0, np.pi)  # Winkel der Linie
            rho = random.uniform(-0.8, 0.8)   # Abstand vom Ursprung
            
            # Normale Linie: x*cos(theta) + y*sin(theta) = rho
            dist_to_line = torch.abs(x_coords * np.cos(theta) + y_coords * np.sin(theta) - rho)
            
            # Parameter für den Streifen
            streak_width = random.uniform(0.05, 0.15)
            streak_intensity = random.uniform(0.7, 1.0)
            
            # Erzeuge Streifen mit Gaußschem Profil
            streak_mask = torch.exp(-(dist_to_line**2) / (2 * streak_width**2)) * streak_intensity
            
            # Füge leichte Variation zur Linie hinzu
            noise = torch.randn(h, w, device=device) * 0.03
            streak_mask = streak_mask * (1 + noise)
            
            # Kombiniere mit bestehender Maske
            streaks_mask = torch.maximum(streaks_mask, streak_mask)
        
        return streaks_mask
    
    def _create_random_burn_mask(self, h, w, device):
        """Kombiniert verschiedene Verbrennungsmuster zufällig"""
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
        
        # Normalisiere zwischen 0 und 1
        if mask.max() > 0:
            mask = mask / mask.max()
        
        return mask
    
    def _create_burn_color(self, burn_level):
        """Erstellt realistischere Verbrennungsfarben basierend auf Intensität"""
        # Farbvariationen von sehr leicht verbrannt bis verkohlt
        if self.color_variation:
            if burn_level < 0.2:  # Leicht gebräunt
                return torch.tensor([0.85, 0.65, 0.45])
            elif burn_level < 0.5:  # Mittel verbrannt
                return torch.tensor([0.65, 0.40, 0.25])
            elif burn_level < 0.8:  # Stark verbrannt
                return torch.tensor([0.35, 0.20, 0.15])
            else:  # Verkohlt
                return torch.tensor([0.15, 0.10, 0.10])
        else:
            # Vereinfachte Version ohne Variation
            return torch.tensor([
                0.8 - burn_level * 0.6,    # Reduziere Rot
                0.5 - burn_level * 0.4,    # Reduziere Grün stärker
                0.4 - burn_level * 0.35    # Reduziere Blau ähnlich
            ])
    
    def forward(self, img):
        # Stelle sicher, dass das Bild ein Tensor ist und auf dem richtigen Gerät liegt
        if not isinstance(img, torch.Tensor):
            img = TVF.to_tensor(img)
        
        device = img.device
        h, w = img.shape[1], img.shape[2]
        
        # Erzeuge Verbrennungsmaske basierend auf ausgewähltem Muster
        if self.burn_pattern == 'edge':
            burn_mask_2d = self._create_edge_burn_mask(h, w, device)
        elif self.burn_pattern == 'spot':
            burn_mask_2d = self._create_spot_burn_mask(h, w, device)
        elif self.burn_pattern == 'streak':
            burn_mask_2d = self._create_streak_burn_mask(h, w, device)
        else:  # 'random' oder default
            burn_mask_2d = self._create_random_burn_mask(h, w, device)
        
        # Zufällige Brenn-Intensität
        burn_intensity = random.uniform(self.burn_intensity_min, self.burn_intensity_max)
        burn_mask_2d = burn_mask_2d * burn_intensity
        
        # Erweitere für alle Kanäle
        burn_mask = burn_mask_2d.unsqueeze(0).expand_as(img)
        
        # Erstelle Verbrennungseffekt mit verschiedenen Farbstufen
        result = img.clone()
        
        # Anwenden unterschiedlicher Verbrennungsfarben je nach Intensität
        burn_levels = [0.25, 0.5, 0.75, 1.0]
        
        for level in burn_levels:
            level_mask = (burn_mask_2d > level * 0.8) & (burn_mask_2d <= level)
            if level_mask.any():
                burn_color = self._create_burn_color(level).to(device)
                
                # Erweitere die Maske und Farbe für alle Kanäle
                level_mask_3d = level_mask.unsqueeze(0).expand_as(img)
                burn_color_3d = burn_color.view(3, 1, 1).expand_as(img)
                
                # Mische ursprüngliches Bild mit Verbrennungsfarbe
                blend_factor = torch.ones_like(img) * (level * burn_intensity)
                result = torch.where(level_mask_3d, 
                                    img * (1 - blend_factor) + burn_color_3d * blend_factor,
                                    result)
        
        # Sehr starke Verbrennung (verkohlt)
        charred_mask = (burn_mask_2d > 0.8)
        if charred_mask.any():
            charred_color = torch.tensor([0.05, 0.05, 0.05]).to(device)
            charred_mask_3d = charred_mask.unsqueeze(0).expand_as(img)
            charred_color_3d = charred_color.view(3, 1, 1).expand_as(img)
            
            result = torch.where(charred_mask_3d, charred_color_3d, result)
        
        # Füge subtile Texturen für verbrannte Bereiche hinzu
        if random.random() < 0.7:
            texture_noise = torch.randn_like(burn_mask_2d) * 0.05
            texture_mask = (burn_mask_2d > 0.3).unsqueeze(0).expand_as(img)
            result = torch.where(texture_mask, result * (1 + texture_noise.unsqueeze(0)), result)
        
        # Begrenze Werte auf gültigen Bereich
        result = torch.clamp(result, 0, 1)
        
        return result

class EnhancedOvenEffect(nn.Module):
    """Verbesserte Simulation von Ofeneffekten mit realistischeren Details"""
    
    def __init__(self, effect_strength=1.0, scipy_available=False):
        super().__init__()
        self.effect_strength = effect_strength
        self.scipy_available = scipy_available
        
    def _apply_gaussian_blur(self, tensor, kernel_size, sigma, device):
        """Wendet Gaußsche Unschärfe auf einen Tensor an"""
        if self.scipy_available:
            # Effizientere Implementierung mit SciPy, aber erfordert CPU-Transfer
            from scipy.ndimage import gaussian_filter
            tensor_np = tensor.squeeze(0).cpu().numpy()
            blurred_np = gaussian_filter(tensor_np, sigma=sigma)
            return torch.tensor(blurred_np, device=device).unsqueeze(0)
        else:
            # PyTorch-basierte Alternative
            # Stelle sicher, dass kernel_size ungerade ist
            kernel_size = max(3, int(kernel_size) // 2 * 2 + 1)
            return TVF.gaussian_blur(tensor, kernel_size=kernel_size, sigma=sigma)
    
    def forward(self, img):
        # Stelle sicher, dass das Bild ein Tensor ist und auf dem richtigen Gerät liegt
        if not isinstance(img, torch.Tensor):
            img = TVF.to_tensor(img)
        
        device = img.device
        h, w = img.shape[1], img.shape[2]
        
        # Wähle Effekte basierend auf Stärke und Zufall
        effects = []
        if random.random() < 0.4 * self.effect_strength:
            effects.append('steam')
        if random.random() < 0.5 * self.effect_strength:
            effects.append('warmth')
        if random.random() < 0.3 * self.effect_strength:
            effects.append('shadow')
        if random.random() < 0.4 * self.effect_strength:
            effects.append('lighting')
        if random.random() < 0.25 * self.effect_strength:
            effects.append('bloom')
        
        # Falls keine Effekte ausgewählt wurden, wähle einen zufälligen
        if not effects and random.random() < 0.7:
            effects.append(random.choice(['steam', 'warmth', 'shadow', 'lighting']))
        
        # Kopie nur erstellen, wenn Effekte angewendet werden
        result = img.clone() if effects else img
        
        # Dampfeffekt - realistischer mit Gradienten und Bewegungsunschärfe
        if 'steam' in effects:
            steam_opacity = random.uniform(0.1, 0.3) * self.effect_strength
            
            # Erzeuge verbesserte Dampfmaske
            # Basis ist ein vertikaler Gradient
            y_coords = torch.linspace(1.0, 0.0, h, device=device).view(-1, 1).expand(-1, w)
            
            # Füge zufällige Variationen hinzu
            noise_scale = random.uniform(0.3, 0.7)
            noise = torch.rand(h, w, device=device) * noise_scale
            
            # Kombiniere mit Gradienten für realistischeren Dampf
            steam_base = y_coords * noise
            
            # Erzeuge Wirbel
            num_swirls = random.randint(2, 5)
            swirl_mask = torch.zeros_like(steam_base)
            
            for _ in range(num_swirls):
                # Zufällige Position für Wirbel
                swirl_x = random.randint(0, w-1)
                swirl_y = random.randint(0, h//3, h//2)  # Mehr im oberen Bereich
                swirl_radius = random.randint(h//8, h//4)
                
                # Erzeuge Koordinaten für alle Pixel
                y_grid, x_grid = torch.meshgrid(
                    torch.arange(h, device=device),
                    torch.arange(w, device=device),
                    indexing='ij'
                )
                
                # Berechne Distanz und Winkel zum Wirbelzentrum
                dist = torch.sqrt(((x_grid - swirl_x)**2 + (y_grid - swirl_y)**2).float())
                swirl_strength = torch.exp(-dist / swirl_radius) * random.uniform(0.5, 1.0)
                
                swirl_mask = torch.maximum(swirl_mask, swirl_strength)
            
            # Kombiniere mit Steam-Basis
            steam_base = steam_base * (1 + swirl_mask * 0.5)
            
            # Glättung der Maske
            sigma = random.uniform(5, 20)
            kernel_size = int(sigma * 3) // 2 * 2 + 1  # Ungerade Zahl ~3*sigma
            steam_mask = self._apply_gaussian_blur(
                steam_base.unsqueeze(0), kernel_size, sigma, device
            ).squeeze(0) * steam_opacity
            
            # Füge leichte Bewegungsunschärfe hinzu
            if random.random() < 0.5:
                angle = random.uniform(-30, 30)  # Winkel in Grad
                motion_kernel_size = random.randint(5, 15)
                steam_mask = TVF.motion_blur(
                    steam_mask.unsqueeze(0),
                    kernel_size=motion_kernel_size,
                    angle=angle,
                    direction=random.choice([-1, 1])
                ).squeeze(0)
            
            # Erweitere für alle Kanäle
            steam_mask_3d = steam_mask.unsqueeze(0).expand_as(result)
            
            # Heller machen wo Dampf ist, mit leicht bläulicher Färbung für Realismus
            steam_color = torch.ones_like(result)
            steam_color[2] = steam_color[2] * 1.05  # Etwas mehr Blau
            
            result = result * (1 - steam_mask_3d) + steam_color * steam_mask_3d
        
        # Wärmeeffekt - gibt dem Bild einen wärmeren Farbton mit subtilen Variationen
        if 'warmth' in effects:
            warmth = random.uniform(0.05, 0.15) * self.effect_strength
            
            # Erzeuge Wärmegradienten (mehr im Zentrum)
            y_coords, x_coords = torch.meshgrid(
                torch.linspace(-1, 1, h, device=device),
                torch.linspace(-1, 1, w, device=device),
                indexing='ij'
            )
            
            # Kreisförmiger Gradient vom Zentrum
            dist_from_center = torch.sqrt(x_coords**2 + y_coords**2)
            warmth_gradient = torch.exp(-dist_from_center * 2) * 0.5 + 0.5
            
            # Subtile Variationen
            if random.random() < 0.3:
                # Verschiebe Zentrum leicht
                center_x = random.uniform(-0.3, 0.3)
                center_y = random.uniform(-0.3, 0.3)
                dist_from_offset = torch.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                warmth_gradient = torch.exp(-dist_from_offset * 2) * 0.5 + 0.5
            
            # Anwenden des Wärmeeffekts mit Gradienten
            warmth_factor = warmth * warmth_gradient
            
            # Kanalweise Anpassung
            result_channels = result.clone()
            # Rot erhöhen
            result_channels[0] = torch.clamp(result[0] * (1 + warmth_factor), 0, 1)
            # Grün leicht erhöhen
            result_channels[1] = torch.clamp(result[1] * (1 + warmth_factor * 0.3), 0, 1)
            # Blau reduzieren
            result_channels[2] = torch.clamp(result[2] * (1 - warmth_factor * 0.2), 0, 1)
            
            result = result_channels
        
        # Schatteneffekt mit realistischeren Übergängen
        if 'shadow' in effects:
            shadow_opacity = random.uniform(0.15, 0.4) * self.effect_strength
            
            # Erzeuge Koordinatengitter
            y_coords, x_coords = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing='ij'
            )
            
            # Parameter für den Schatten
            shadow_type = random.choice(['corner', 'side', 'spot'])
            
            if shadow_type == 'corner':
                # Schatteneffekt von einer Ecke
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
                # Schatten von einer Seite
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
                # Runder Schatten an zufälliger Position
                shadow_x = random.randint(w//4, w*3//4)
                shadow_y = random.randint(h//4, h*3//4)
                shadow_radius = random.uniform(0.3, 0.5) * min(h, w)
                
                shadow_dist = torch.sqrt(((x_coords - shadow_x)**2 + (y_coords - shadow_y)**2).float())
                shadow_mask = (1 - torch.exp(-shadow_dist**2 / (2 * shadow_radius**2))) * shadow_opacity
            
            # Füge leichte Unschärfe zum Schatten hinzu
            sigma = random.uniform(10, 30)
            kernel_size = int(sigma * 2) // 2 * 2 + 1
            shadow_mask = self._apply_gaussian_blur(
                shadow_mask.unsqueeze(0), kernel_size, sigma, device
            ).squeeze(0)
            
            # Erweitere für alle Kanäle
            shadow_mask_3d = shadow_mask.unsqueeze(0).expand_as(result)
            
            # Dunkler machen wo Schatten ist
            result = result * (1 - shadow_mask_3d)
        
        # Beleuchtungseffekt - simuliert ungleichmäßige Beleuchtung im Ofen
        if 'lighting' in effects:
            light_intensity = random.uniform(0.1, 0.25) * self.effect_strength
            
            # Erzeuge Koordinatengitter
            y_coords, x_coords = torch.meshgrid(
                torch.linspace(-1, 1, h, device=device),
                torch.linspace(-1, 1, w, device=device),
                indexing='ij'
            )
            
            # Parameter für das Licht
            light_x = random.uniform(-0.5, 0.5)
            light_y = random.uniform(-0.5, 0.5)
            light_radius = random.uniform(0.4, 0.8)
            
            # Berechne Distanz zum Lichtquelle
            light_dist = torch.sqrt((x_coords - light_x)**2 + (y_coords - light_y)**2)
            
            # Erzeuge Lichtmaske mit weichem Abfall
            light_mask = torch.exp(-light_dist**2 / (2 * light_radius**2)) * light_intensity
            
            # Erweitere für alle Kanäle
            light_mask_3d = light_mask.unsqueeze(0).expand_as(result)
            
            # Heller machen wo Licht ist, mit leicht gelblicher Färbung
            light_color = torch.ones_like(result)
            light_color[0] = light_color[0] * 1.05  # Etwas mehr Rot
            light_color[1] = light_color[1] * 1.03  # Etwas mehr Grün
            
            result = result * (1 - light_mask_3d) + torch.minimum(
                light_color * result * (1 + light_mask_3d),
                torch.ones_like(result)
            )
        
        # Bloom-Effekt - simuliert leichtes Glühen bei hellen Stellen
        if 'bloom' in effects:
            bloom_strength = random.uniform(0.1, 0.2) * self.effect_strength
            
            # Identifiziere helle Bereiche
            luminance = 0.299 * result[0] + 0.587 * result[1] + 0.114 * result[2]
            bright_areas = (luminance > 0.7).float()
            
            # Füge Unschärfe für Glühen hinzu
            sigma = random.uniform(5, 10)
            kernel_size = int(sigma * 3) // 2 * 2 + 1
            glow = self._apply_gaussian_blur(
                bright_areas.unsqueeze(0), kernel_size, sigma, device
            ).squeeze(0) * bloom_strength
            
            # Erweitere für alle Kanäle mit leicht gelblichem Farbton
            glow_mask = glow.unsqueeze(0).expand_as(result)
            
            # Heller machen wo Glühen ist
            glow_color = torch.ones_like(result)
            glow_color[0] = 1.05  # Mehr Rot
            glow_color[1] = 1.03  # Mehr Grün
            glow_color[2] = 0.95  # Weniger Blau
            
            result = torch.minimum(result + glow_mask * glow_color, torch.ones_like(result))
        
        # Begrenze Werte auf gültigen Bereich
        result = torch.clamp(result, 0, 1)
        
        return result

class PizzaSegmentEffect(nn.Module):
    """Modulares Effektsystem für einzelne Pizza-Segmente"""
    
    def __init__(self, device, burning_min=0.0, burning_max=0.9):
        super().__init__()
        self.device = device
        self.burning_min = burning_min
        self.burning_max = burning_max
        
        # Instanziiere Effekt-Module
        self.burning_effect = EnhancedPizzaBurningEffect(
            burn_intensity_min=burning_min,
            burn_intensity_max=burning_max
        ).to(device)
        
        self.oven_effect = EnhancedOvenEffect(
            effect_strength=random.uniform(0.5, 1.0)
        ).to(device)
    
    def _create_segment_mask(self, h, w, num_segments=8):
        """Erzeugt eine Maske für Pizzasegmente"""
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, h, device=self.device),
            torch.linspace(-1, 1, w, device=self.device),
            indexing='ij'
        )
        
        # Berechne Winkel für alle Pixel (in Radianten)
        angles = torch.atan2(y_coords, x_coords)
        # Normalisiere Winkel auf [0, 2π]
        angles = torch.where(angles < 0, angles + 2 * np.pi, angles)
        
        # Distanz vom Zentrum
        dist_from_center = torch.sqrt(x_coords**2 + y_coords**2)
        
        # Kreismaske für Pizza-Bereich
        pizza_radius = random.uniform(0.7, 0.9)
        pizza_mask = (dist_from_center <= pizza_radius).float()
        
        # Segmentmasken
        segment_size = 2 * np.pi / num_segments
        segment_masks = []
        
        for i in range(num_segments):
            start_angle = i * segment_size
            end_angle = (i + 1) * segment_size
            
            # Maske für aktuelles Segment
            if end_angle <= 2 * np.pi:
                segment_mask = ((angles >= start_angle) & (angles < end_angle)).float()
            else:
                # Behandle Überlauf
                segment_mask = ((angles >= start_angle) | (angles < (end_angle % (2 * np.pi)))).float()
            
            # Kombiniere mit Pizza-Maske
            segment_mask = segment_mask * pizza_mask
            
            segment_masks.append(segment_mask)
        
        return segment_masks
    
    def forward(self, img):
        # Stelle sicher, dass das Bild ein Tensor ist und auf dem richtigen Gerät liegt
        if not isinstance(img, torch.Tensor):
            img = TVF.to_tensor(img).to(self.device)
        else:
            img = img.to(self.device)
        
        h, w = img.shape[1], img.shape[2]
        
        # Erstelle zufällige Anzahl von Segmenten
        num_segments = random.randint(4, 8)
        segment_masks = self._create_segment_mask(h, w, num_segments)
        
        # Erstelle für jedes Segment unterschiedliche Effektkombinationen
        result = img.clone()
        
        for i, segment_mask in enumerate(segment_masks):
            # Zufällige Effekte pro Segment
            segment_img = img.clone()
            
            # Zufällige Verbrennungsstärke
            burn_intensity = random.uniform(0, 1)
            
            if burn_intensity > 0.1:  # Anwenden von Verbrennungseffekten
                # Anpassen der Brennintensität pro Segment
                self.burning_effect.burn_intensity_min = self.burning_min + burn_intensity * 0.3
                self.burning_effect.burn_intensity_max = min(self.burning_min + burn_intensity * 0.6, self.burning_max)
                
                # Wähle Muster
                self.burning_effect.burn_pattern = random.choice(['random', 'edge', 'spot'])
                
                # Anwenden des Effekts
                segment_img = self.burning_effect(segment_img)
            
            # Zufällige Anwendung von Ofeneffekten
            if random.random() < 0.5:
                segment_img = self.oven_effect(segment_img)
            
            # Erweitere Maske für alle Kanäle
            segment_mask_3d = segment_mask.unsqueeze(0).expand_as(result)
            
            # Kombiniere mit Ergebnis
            result = torch.where(segment_mask_3d > 0, segment_img, result)
        
        return result

# ================ AUGMENTIERUNGSPIPELINE ================

def apply_basic_augmentation(dataset, output_dir, count, batch_size, show_samples=False):
    """Erweiterte Basis-Augmentierung mit optimiertem Workflow"""
    device = next(iter(dataset))[0].device
    
    # Optimierte grundlegende Transformationen
    basic_transforms = [
        # Verschiedene Rotationen
        transforms.RandomApply([
            transforms.RandomRotation(180)
        ], p=0.9),
        
        # Verschiedene Ausschnitte
        transforms.RandomApply([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1))
        ], p=0.8),
        
        # Spiegelungen
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1)
        ], p=0.6),
        
        # Farbvariationen
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, 
                saturation=0.4, hue=0.2
            )
        ], p=0.8),
        
        # Perspektivische Transformationen
        transforms.RandomApply([
            transforms.RandomPerspective(distortion_scale=0.2)
        ], p=0.3),
        
        # Verschiedene Filter
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))
        ], p=0.4),
        
        # Schärfen
        transforms.RandomApply([
            transforms.RandomAdjustSharpness(sharpness_factor=2)
        ], p=0.3),
        
        transforms.ToTensor(),
    ]
    
    # Genaue Anzahl der Varianten pro Bild berechnen
    images_per_original = max(1, count // len(dataset))
    remaining = count - (images_per_original * len(dataset))
    
    print(f"Erzeuge {images_per_original} Basis-Varianten pro Bild, plus {remaining} zusätzliche")
    
    augmented_images = []
    sample_images = []
    
    with torch.no_grad():  # Speicheroptimierung
        # Für jedes Bild im Dataset
        for idx, img in enumerate(tqdm(dataset, desc="Basis-Augmentierung")):
            # Zusätzliches Bild für die letzten paar Bilder
            num_variants = images_per_original + (1 if idx < remaining else 0)
            
            # Speichere das Original
            img = img.to(device)
            original_pil = TVF.to_pil_image(img.cpu())
            
            # Erzeuge mehrere Varianten pro Bild
            for variant_idx in range(num_variants):
                # Jede Variante verwendet eine zufällige Kombination von Transformationen
                transform_list = []
                for transform in basic_transforms:
                    if random.random() < 0.5:  # 50% Chance für jede Transformation
                        transform_list.append(transform)
                
                # Stelle sicher, dass wir mindestens eine Transformation haben
                if not any(isinstance(t, transforms.RandomApply) for t in transform_list):
                    # Füge zufällige Transformationen hinzu
                    transform_list.append(random.choice(basic_transforms))
                
                # Füge immer ToTensor hinzu, falls nicht vorhanden
                if not any(isinstance(t, transforms.ToTensor) for t in transform_list):
                    transform_list.append(transforms.ToTensor())
                
                # Erstelle Komposition und wende an
                composed = transforms.Compose(transform_list)
                try:
                    augmented = composed(original_pil).to(device)
                    augmented_images.append(augmented)
                    
                    # Speichere einige Beispiele für die Anzeige
                    if variant_idx == 0 and len(sample_images) < 10:
                        sample_images.append(augmented)
                except Exception as e:
                    print(f"Fehler bei der Basis-Augmentierung: {e}")
                
                # Batch-weise Speicherfreigabe
                if len(augmented_images) >= batch_size:
                    # Speichere Batch
                    save_augmented_images(
                        augmented_images, 
                        output_dir, 
                        f"basic_{len(augmented_images)}", 
                        batch_size
                    )
                    
                    # Speicher freigeben
                    augmented_images = []
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    
    # Restliche Bilder speichern
    if augmented_images:
        save_augmented_images(
            augmented_images, 
            output_dir, 
            f"basic_{len(augmented_images)}", 
            batch_size
        )
    
    # Zeige Beispiele an
    if show_samples and sample_images:
        show_images(
            sample_images[:9],
            titles=["Basis-Augmentierung"] * len(sample_images[:9]),
            save_path=os.path.join(output_dir, "..", "samples_basic.jpg")
        )
    
    generated_count = count - len(augmented_images)
    print(f"Basis-Augmentierung abgeschlossen: {generated_count} Bilder erstellt")
    return generated_count

def apply_burning_augmentation(dataset, output_dir, count, batch_size, show_samples=False):
    """Erweiterte Verbrennungs-Augmentierung mit verschiedenen Stilen"""
    device = next(iter(dataset))[0].device
    
    # Erstelle Verbrennungseffekt-Instanzen mit verschiedenen Stilen
    burning_effects = [
        # Leichte Verbrennung
        EnhancedPizzaBurningEffect(
            burn_intensity_min=0.1, burn_intensity_max=0.3,
            burn_pattern='random'
        ).to(device),
        
        # Mittlere Randverbrennung
        EnhancedPizzaBurningEffect(
            burn_intensity_min=0.3, burn_intensity_max=0.6,
            burn_pattern='edge'
        ).to(device),
        
        # Starke, fleckige Verbrennung
        EnhancedPizzaBurningEffect(
            burn_intensity_min=0.5, burn_intensity_max=0.9,
            burn_pattern='spot'
        ).to(device),
        
        # Streifige Verbrennung
        EnhancedPizzaBurningEffect(
            burn_intensity_min=0.3, burn_intensity_max=0.7,
            burn_pattern='streak'
        ).to(device),
    ]
    
    # Oven-Effekt-Instanz
    oven_effect = EnhancedOvenEffect().to(device)
    
    # Genaue Anzahl der Varianten pro Bild berechnen
    images_per_original = max(1, count // len(dataset))
    remaining = count - (images_per_original * len(dataset))
    
    print(f"Erzeuge {images_per_original} Verbrennungs-Varianten pro Bild, plus {remaining} zusätzliche")
    
    augmented_images = []
    sample_images = []
    
    with torch.no_grad():  # Speicheroptimierung
        # Für jedes Bild im Dataset
        for idx, img in enumerate(tqdm(dataset, desc="Verbrennungs-Augmentierung")):
            # Zusätzliches Bild für die letzten paar Bilder
            num_variants = images_per_original + (1 if idx < remaining else 0)
            
            # Erzeuge mehrere Varianten pro Bild
            for variant_idx in range(num_variants):
                try:
                    img_tensor = img.to(device)
                    
                    # Wähle zufälligen Verbrennungseffekt
                    burn_effect = random.choice(burning_effects)
                    
                    # Anwenden der Effekte
                    img_tensor = burn_effect(img_tensor)
                    
                    # Optional: Ofeneffekt hinzufügen
                    if random.random() < 0.7:
                        img_tensor = oven_effect(img_tensor)
                    
                    augmented_images.append(img_tensor)
                    
                    # Speichere einige Beispiele für die Anzeige
                    if variant_idx == 0 and len(sample_images) < 10:
                        sample_images.append(img_tensor)
                
                except Exception as e:
                    print(f"Fehler bei der Verbrennungs-Augmentierung: {e}")
                
                # Batch-weise Speicherfreigabe
                if len(augmented_images) >= batch_size:
                    # Speichere Batch
                    save_augmented_images(
                        augmented_images, 
                        output_dir, 
                        f"burnt_{len(augmented_images)}", 
                        batch_size
                    )
                    
                    # Speicher freigeben
                    augmented_images = []
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    
    # Restliche Bilder speichern
    if augmented_images:
        save_augmented_images(
            augmented_images, 
            output_dir, 
            f"burnt_{len(augmented_images)}", 
            batch_size
        )
    
    # Zeige Beispiele an
    if show_samples and sample_images:
        show_images(
            sample_images[:9],
            titles=["Verbrennung"] * len(sample_images[:9]),
            save_path=os.path.join(output_dir, "..", "samples_burnt.jpg")
        )
    
    generated_count = count - len(augmented_images)
    print(f"Verbrennungs-Augmentierung abgeschlossen: {generated_count} Bilder erstellt")
    return generated_count

def apply_mixed_augmentation(dataset, output_dir, count, batch_size, show_samples=False):
    """Erweiterte Methoden zum Mischen von Bildern (MixUp, CutMix, CopyPaste)"""
    device = next(iter(dataset))[0].device
    
    # Benötigte Paare für die angegebene Anzahl
    num_pairs_needed = count
    
    print(f"Erzeuge {count} gemischte Bilder (MixUp, CutMix, etc.)")
    
    # Speichere alle Bilder aus dem Dataset für effizientes Sampling
    all_images = []
    for img in tqdm(dataset, desc="Bilder laden"):
        all_images.append(img.to(device))
    
    if len(all_images) < 2:
        print("Nicht genügend Bilder für Mischungsaugmentierung")
        return 0
    
    augmented_images = []
    sample_images = []
    mix_methods = ['mixup', 'cutmix', 'copypaste']
    
    def mixup(img1, img2, alpha=0.3):
        """Verbesserte MixUp mit zusätzlichen Optionen"""
        # Stelle sicher, dass beide Bilder die gleiche Größe haben
        if img1.shape != img2.shape:
            h, w = img1.shape[1:3]
            img2 = F.interpolate(img2.unsqueeze(0), size=(h, w), 
                                 mode='bilinear', align_corners=False).squeeze(0)
        
        # Erzeuge Mischparameter mit Beta-Verteilung
        lam = np.random.beta(alpha, alpha)
        
        # Verschiedene Mischvarianten
        mode = random.choice(['standard', 'channel_specific', 'spatial'])
        
        if mode == 'standard':
            # Standard MixUp: einfache lineare Interpolation
            mixed = lam * img1 + (1 - lam) * img2
        
        elif mode == 'channel_specific':
            # Kanalspezifisches MixUp: jeder Kanal hat unterschiedliche Mischungsrate
            channel_weights = torch.tensor([
                np.random.beta(alpha, alpha),
                np.random.beta(alpha, alpha),
                np.random.beta(alpha, alpha)
            ], device=device).view(3, 1, 1)
            
            mixed = channel_weights * img1 + (1 - channel_weights) * img2
        
        else:  # 'spatial'
            # Räumliches MixUp: Mischungsrate variiert über das Bild
            h, w = img1.shape[1:3]
            
            # Erzeuge Maske mit Perlin-ähnlichem Rauschen
            mask = torch.zeros(h, w, device=device)
            scale = random.randint(2, 5)  # Skala des Rauschens
            
            for i in range(scale):
                # Erstelle grobes Rauschen und interpoliere
                noise_size = 2**(i+2)
                noise = torch.randn(noise_size, noise_size, device=device)
                noise = F.interpolate(
                    noise.unsqueeze(0).unsqueeze(0), 
                    size=(h, w), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze()
                
                # Addiere mit abnehmender Gewichtung für gröbere Skalen
                mask += noise / (2**i)
            
            # Normalisiere und passe an gewünschte Mischung an
            mask = torch.sigmoid((mask - mask.mean()) / mask.std() * 3)  # Kontrastverstärkung
            mask = mask * (2*lam - 1) + (1 - lam)  # Passe an durchschnittliche Mischungsrate an
            mask = torch.clamp(mask, 0, 1)
            
            # Erweitere Maske für alle Kanäle
            mask = mask.unsqueeze(0).expand_as(img1)
            
            # Räumliche Mischung
            mixed = mask * img1 + (1 - mask) * img2
        
        return mixed
    
    def cutmix(img1, img2):
        """Verbesserte CutMix-Varianten speziell für Pizza-Bilder"""
        # Stelle sicher, dass beide Bilder die gleiche Größe haben
        if img1.shape != img2.shape:
            h, w = img1.shape[1:3]
            img2 = F.interpolate(img2.unsqueeze(0), size=(h, w), 
                                 mode='bilinear', align_corners=False).squeeze(0)
        
        h, w = img1.shape[1:3]
        center_x, center_y = w // 2, h // 2
        
        # Erzeuge Koordinatengitter
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij'
        )
        
        # Verschiedene Formen für CutMix
        shape = random.choice(['circle', 'wedge', 'checkerboard', 'halves'])
        
        if shape == 'circle':
            # Kreisförmiges Segment
            radius = random.uniform(0.3, 0.7) * min(h, w) / 2
            dist = torch.sqrt(((x_coords - center_x)**2 + (y_coords - center_y)**2).float())
            
            # Füge Unschärfe zum Rand hinzu
            falloff_width = random.uniform(5, 20)
            mask_float = 1.0 - torch.sigmoid((dist - radius) / falloff_width)
            
            # Binäre Maske für CutMix
            mask = (mask_float > 0.5)
        
        elif shape == 'wedge':
            # Keilförmiges Segment für Pizzastücke
            start_angle = random.uniform(0, 2 * np.pi)
            angle_width = random.uniform(np.pi/6, np.pi/2)  # 30 bis 90 Grad
            
            # Berechne Winkel für alle Pixel
            delta_x = (x_coords - center_x).float()
            delta_y = (y_coords - center_y).float()
            angles = torch.atan2(delta_y, delta_x)
            
            # Normalisiere Winkel auf [0, 2π]
            angles = torch.where(angles < 0, angles + 2 * np.pi, angles)
            
            # Distanz vom Zentrum für Kreisform
            dist = torch.sqrt(delta_x**2 + delta_y**2)
            circle_mask = (dist <= min(h, w) / 2 * 0.9)
            
            # Maske für Winkelbereich
            if start_angle + angle_width <= 2 * np.pi:
                angle_mask = (angles >= start_angle) & (angles <= start_angle + angle_width)
            else:
                # Behandle Überlauf
                angle_mask = (angles >= start_angle) | (angles <= (start_angle + angle_width) % (2 * np.pi))
            
            # Kombiniere mit Kreismaske
            mask = angle_mask & circle_mask
        
        elif shape == 'checkerboard':
            # Schachbrettmuster
            tile_size = random.randint(10, 30)
            mask = ((x_coords // tile_size + y_coords // tile_size) % 2 == 0)
        
        else:  # 'halves'
            # Teilt das Bild in Hälften
            direction = random.choice(['horizontal', 'vertical', 'diagonal'])
            
            if direction == 'horizontal':
                split_point = random.randint(h // 4, 3 * h // 4)
                mask = (y_coords < split_point)
            elif direction == 'vertical':
                split_point = random.randint(w // 4, 3 * w // 4)
                mask = (x_coords < split_point)
            else:  # diagonal
                slope = random.choice([-1, 1])  # Aufsteigend oder absteigend
                offset = random.randint(-min(h, w) // 4, min(h, w) // 4)
                mask = (y_coords * slope + offset < x_coords)
        
        # Erweitere Maske für alle Kanäle
        mask_3d = mask.unsqueeze(0).expand_as(img1)
        
        # Anwenden der Maske
        result = torch.where(mask_3d, img2, img1)
        
        return result
    
    def copypaste(img1, img2):
        """Pizza-spezifische Copy-Paste-Augmentierung"""
        # Stelle sicher, dass beide Bilder die gleiche Größe haben
        if img1.shape != img2.shape:
            h, w = img1.shape[1:3]
            img2 = F.interpolate(img2.unsqueeze(0), size=(h, w), 
                                 mode='bilinear', align_corners=False).squeeze(0)
        
        h, w = img1.shape[1:3]
        
        # Wir simulieren das Kopieren von verbrannten oder speziellen Bereichen
        # Zuerst identifizieren wir potenzielle interessante Bereiche im zweiten Bild
        
        # Vereinfachte Segmentierung: Wir betrachten dunkle Bereiche als "verbrannt"
        # oder helle Bereiche als "Belag"
        feature_type = random.choice(['dark', 'bright'])
        
        if feature_type == 'dark':
            # Dunkle Bereiche identifizieren (verbrannt)
            luminance = 0.299 * img2[0] + 0.587 * img2[1] + 0.114 * img2[2]
            threshold = random.uniform(0.2, 0.4)
            features = (luminance < threshold).float()
        else:
            # Helle Bereiche identifizieren (Käse, Belag)
            luminance = 0.299 * img2[0] + 0.587 * img2[1] + 0.114 * img2[2]
            threshold = random.uniform(0.6, 0.8)
            features = (luminance > threshold).float()
        
        # Erodiere und dilatiere, um Rauschen zu reduzieren
        kernel_size = random.randint(3, 7)
        padding = kernel_size // 2
        
        # Approximiere morphologische Operationen mit Faltung
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device) / (kernel_size ** 2)
        
        # Erodieren: Entferne kleine Features
        features = features.unsqueeze(0).unsqueeze(0)
        eroded = (F.conv2d(features, kernel, padding=padding) > 0.7).float()
        
        # Dilatieren: Erweitere die Features
        dilated = (F.conv2d(eroded, kernel, padding=padding) > 0.1).float()
        
        # Konvertiere zurück
        mask = dilated.squeeze()
        
        # Optional: Glätte die Kanten der Maske
        if random.random() < 0.7:
            mask = F.avg_pool2d(
                mask.unsqueeze(0).unsqueeze(0),
                kernel_size=5, stride=1, padding=2
            ).squeeze()
        
        # Zufällige Verschiebung der Maske
        shift_x = random.randint(-w//4, w//4)
        shift_y = random.randint(-h//4, h//4)
        
        shifted_mask = torch.zeros_like(mask)
        
        # Berücksichtige die Verschiebung bei der Anwendung der Maske
        src_start_y = max(0, -shift_y)
        src_end_y = min(h, h - shift_y)
        src_start_x = max(0, -shift_x)
        src_end_x = min(w, w - shift_x)
        
        dst_start_y = max(0, shift_y)
        dst_end_y = min(h, h + shift_y)
        dst_start_x = max(0, shift_x)
        dst_end_x = min(w, w + shift_x)
        
        # Kopiere den überlappenden Bereich
        if src_end_y > src_start_y and src_end_x > src_start_x:
            shifted_mask[
                dst_start_y:dst_end_y,
                dst_start_x:dst_end_x
            ] = mask[
                src_start_y:src_end_y,
                src_start_x:src_end_x
            ]
        
        # Erweitere Maske für alle Kanäle
        mask_3d = shifted_mask.unsqueeze(0).expand_as(img1)
        
        # Anwenden der Maske für Copy-Paste
        result = torch.where(mask_3d > 0.5, img2, img1)
        
        return result
    
    with torch.no_grad():  # Speicheroptimierung
        for i in range(num_pairs_needed):
            try:
                # Wähle zwei zufällige Bilder
                if len(all_images) >= 2:
                    idx1, idx2 = random.sample(range(len(all_images)), 2)
                    img1, img2 = all_images[idx1], all_images[idx2]
                else:
                    # Falls nicht genug Bilder, dupliziere und ändere das eine verfügbare Bild
                    img1 = all_images[0]
                    img2 = TVF.adjust_brightness(
                        TVF.adjust_contrast(img1, random.uniform(0.8, 1.2)), 
                        random.uniform(0.8, 1.2)
                    )
                
                # Wähle zufällig eine Mischmethode
                method = random.choice(mix_methods)
                
                if method == 'mixup':
                    mixed = mixup(img1, img2, alpha=random.uniform(0.2, 0.5))
                elif method == 'cutmix':
                    mixed = cutmix(img1, img2)
                else:  # copypaste
                    mixed = copypaste(img1, img2)
                
                augmented_images.append(mixed)
                
                # Speichere einige Beispiele für die Anzeige
                if i < 10:
                    sample_images.append(mixed)
            
            except Exception as e:
                print(f"Fehler bei der Mix-Augmentierung: {e}")
            
            # Batch-weise Speicherfreigabe
            if len(augmented_images) >= batch_size:
                save_augmented_images(
                    augmented_images, 
                    output_dir, 
                    f"mixed_{len(augmented_images)}", 
                    batch_size
                )
                
                # Speicher freigeben
                augmented_images = []
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # Restliche Bilder speichern
    if augmented_images:
        save_augmented_images(
            augmented_images, 
            output_dir, 
            f"mixed_{len(augmented_images)}", 
            batch_size
        )
    
    # Zeige Beispiele an
    if show_samples and sample_images:
        show_images(
            sample_images[:9],
            titles=["Gemischt"] * len(sample_images[:9]),
            save_path=os.path.join(output_dir, "..", "samples_mixed.jpg")
        )
    
    generated_count = count - len(augmented_images)
    print(f"Mix-Augmentierung abgeschlossen: {generated_count} Bilder erstellt")
    return generated_count

def apply_progression_augmentation(dataset, output_dir, count, batch_size, show_samples=False):
    """Erweiterte Verbrennungsprogression mit kontrollierbaren Stufen"""
    device = next(iter(dataset))[0].device
    
    # Anzahl der Bilder und Stufen berechnen
    num_progressions = max(1, count // 5)  # Jede Progression hat ca. 5 Stufen
    print(f"Erzeuge {num_progressions} Verbrennungsprogressionen mit mehreren Stufen")
    
    augmented_images = []
    sample_progressions = []
    
    with torch.no_grad():  # Speicheroptimierung
        # Wähle zufällig einige Bilder für die Progression
        indices = np.random.choice(len(dataset), min(num_progressions, len(dataset)), replace=False)
        
        for idx in tqdm(indices, desc="Verbrennungsprogression"):
            try:
                img = dataset[idx].to(device)
                
                # Wähle die Anzahl der Stufen
                num_steps = random.randint(4, 6)
                
                # Entscheide über den Progressionstyp
                progression_type = random.choice([
                    'increasing_uniform',   # Gleichmäßig zunehmende Verbrennung
                    'increasing_edge',      # Vom Rand zunehmend
                    'increasing_spots',     # Zunehmende Spots
                    'mixed'                 # Gemischte Stile
                ])
                
                # Erstelle Verbrennungseffekt-Instanzen für jeden Stil
                burn_effects = []
                
                if progression_type == 'increasing_uniform':
                    # Gleichmäßig zunehmende Verbrennung
                    for i in range(num_steps):
                        intensity_min = 0.1 + (i * 0.15)
                        intensity_max = 0.2 + (i * 0.15)
                        
                        burn_effects.append(EnhancedPizzaBurningEffect(
                            burn_intensity_min=intensity_min,
                            burn_intensity_max=intensity_max,
                            burn_pattern='random'
                        ).to(device))
                
                elif progression_type == 'increasing_edge':
                    # Verbrennung breitet sich vom Rand aus
                    for i in range(num_steps):
                        intensity = 0.3 + (i * 0.15)
                        
                        burn_effects.append(EnhancedPizzaBurningEffect(
                            burn_intensity_min=intensity - 0.1,
                            burn_intensity_max=intensity + 0.1,
                            burn_pattern='edge'
                        ).to(device))
                
                elif progression_type == 'increasing_spots':
                    # Zunehmende Spots-Verbrennung
                    for i in range(num_steps):
                        intensity = 0.3 + (i * 0.15)
                        
                        burn_effects.append(EnhancedPizzaBurningEffect(
                            burn_intensity_min=intensity - 0.1,
                            burn_intensity_max=intensity + 0.1,
                            burn_pattern='spot'
                        ).to(device))
                
                else:  # 'mixed'
                    # Verschiedene Stile für realistischere Progression
                    patterns = ['edge', 'spot', 'streak', 'random']
                    
                    for i in range(num_steps):
                        intensity = 0.2 + (i * 0.15)
                        pattern = patterns[i % len(patterns)]
                        
                        burn_effects.append(EnhancedPizzaBurningEffect(
                            burn_intensity_min=intensity - 0.1,
                            burn_intensity_max=intensity + 0.1,
                            burn_pattern=pattern
                        ).to(device))
                
                # Füge Originalbild am Anfang hinzu
                progression = [img.clone()]
                
                # Wende Effekte in aufsteigender Reihenfolge an
                for effect in burn_effects:
                    burnt_img = effect(img.clone())
                    progression.append(burnt_img)
                
                # Optional: Füge Ofeneffekte hinzu
                if random.random() < 0.3:
                    oven_effect = EnhancedOvenEffect().to(device)
                    for i in range(len(progression)):
                        progression[i] = oven_effect(progression[i])
                
                # Füge zur Ausgabe hinzu
                augmented_images.extend(progression)
                
                # Speichere für Beispielanzeige
                if len(sample_progressions) < 2:
                    sample_progressions.append(progression)
                
            except Exception as e:
                print(f"Fehler bei der Progressions-Augmentierung: {e}")
            
            # Batch-weise Speicherfreigabe
            if len(augmented_images) >= batch_size:
                save_augmented_images(
                    augmented_images, 
                    output_dir, 
                    f"progression_{len(augmented_images)}", 
                    batch_size
                )
                
                # Speicher freigeben
                augmented_images = []
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # Restliche Bilder speichern
    if augmented_images:
        save_augmented_images(
            augmented_images, 
            output_dir, 
            f"progression_{len(augmented_images)}", 
            batch_size
        )
    
    # Zeige Beispiele an
    if show_samples and sample_progressions:
        for i, prog in enumerate(sample_progressions[:2]):
            show_images(
                prog, 
                titles=[f"Stufe {j}" for j in range(len(prog))],
                save_path=os.path.join(output_dir, "..", f"samples_progression_{i}.jpg")
            )
    
    generated_count = count - len(augmented_images)
    print(f"Progressions-Augmentierung abgeschlossen: {generated_count} Bilder erstellt")
    return generated_count

def apply_segment_augmentation(dataset, output_dir, count, batch_size, show_samples=False):
    """Segment-basierte Augmentierung speziell für Pizza-Bilder"""
    device = next(iter(dataset))[0].device
    
    # Anzahl der Bilder pro Original
    images_per_original = max(1, count // len(dataset))
    remaining = count - (images_per_original * len(dataset))
    
    print(f"Erzeuge {images_per_original} Segment-Varianten pro Bild, plus {remaining} zusätzliche")
    
    augmented_images = []
    sample_images = []
    
    with torch.no_grad():  # Speicheroptimierung
        # Für jedes Bild im Dataset
        for idx, img in enumerate(tqdm(dataset, desc="Segment-Augmentierung")):
            # Zusätzliches Bild für die letzten paar Bilder
            num_variants = images_per_original + (1 if idx < remaining else 0)
            
            # Erstelle Segment-Effekt-Instanz
            segment_effect = PizzaSegmentEffect(
                device=device,
                burning_min=0.0,
                burning_max=0.9
            )
            
            # Erzeuge mehrere Varianten mit unterschiedlichen Segmentierungen
            for variant_idx in range(num_variants):
                try:
                    img_tensor = img.to(device)
                    
                    # Anwenden des Segment-Effekts
                    segmented_img = segment_effect(img_tensor)
                    
                    augmented_images.append(segmented_img)
                    
                    # Speichere einige Beispiele für die Anzeige
                    if variant_idx == 0 and len(sample_images) < 10:
                        sample_images.append(segmented_img)
                
                except Exception as e:
                    print(f"Fehler bei der Segment-Augmentierung: {e}")
                
                # Batch-weise Speicherfreigabe
                if len(augmented_images) >= batch_size:
                    save_augmented_images(
                        augmented_images, 
                        output_dir, 
                        f"segment_{len(augmented_images)}", 
                        batch_size
                    )
                    
                    # Speicher freigeben
                    augmented_images = []
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    
    # Restliche Bilder speichern
    if augmented_images:
        save_augmented_images(
            augmented_images, 
            output_dir, 
            f"segment_{len(augmented_images)}", 
            batch_size
        )
    
    # Zeige Beispiele an
    if show_samples and sample_images:
        show_images(
            sample_images[:9],
            titles=["Segmentiert"] * len(sample_images[:9]),
            save_path=os.path.join(output_dir, "..", "samples_segment.jpg")
        )
    
    generated_count = count - len(augmented_images)
    print(f"Segment-Augmentierung abgeschlossen: {generated_count} Bilder erstellt")
    return generated_count

def apply_combination_augmentation(dataset, output_dir, count, batch_size, device, show_samples=False):
    """Kombiniert mehrere Augmentierungstechniken für komplexere Variationen"""
    # Anzahl der Bilder pro Original
    images_per_original = max(1, count // len(dataset))
    remaining = count - (images_per_original * len(dataset))
    
    print(f"Erzeuge {images_per_original} kombinierte Varianten pro Bild, plus {remaining} zusätzliche")
    
    # Instanziiere alle Effekte
    burning_effect = EnhancedPizzaBurningEffect(
        burn_intensity_min=0.2, burn_intensity_max=0.7
    ).to(device)
    
    oven_effect = EnhancedOvenEffect().to(device)
    
    segment_effect = PizzaSegmentEffect(
        device=device,
        burning_min=0.1,
        burning_max=0.8
    )
    
    # Transformationen für Basis-Augmentierung
    base_transforms = [
        transforms.RandomRotation(180),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.RandomAdjustSharpness(sharpness_factor=2)
    ]
    
    augmented_images = []
    sample_images = []
    
    with torch.no_grad():  # Speicheroptimierung
        # Lade alle Bilder für Mix-Operationen
        all_images = []
        for img in dataset:
            all_images.append(img.to(device))
        
        # Für jedes Bild im Dataset
        for idx, img in enumerate(tqdm(dataset, desc="Kombinations-Augmentierung")):
            # Zusätzliches Bild für die letzten paar Bilder
            num_variants = images_per_original + (1 if idx < remaining else 0)
            
            original_img = img.to(device)
            
            # Erzeuge mehrere Varianten mit Kombinationen
            for variant_idx in range(num_variants):
                try:
                    # Starte mit frischer Kopie
                    result = original_img.clone()# Starte mit frischer Kopie
                    result = original_img.clone()
                    
                    # 1. Zufällige Basis-Transformationen
                    if random.random() < 0.7:
                        # Konvertiere zu PIL für Basis-Transformationen
                        pil_img = TVF.to_pil_image(result.cpu())
                        
                        # Wähle einige zufällige Transformationen aus
                        selected_transforms = []
                        for transform in base_transforms:
                            if random.random() < 0.4:
                                selected_transforms.append(transform)
                        
                        # Stelle sicher, dass wenigstens eine Transformation ausgewählt wurde
                        if not selected_transforms:
                            selected_transforms.append(random.choice(base_transforms))
                        
                        # Wende Transformationen an
                        for transform in selected_transforms:
                            pil_img = transform(pil_img)
                        
                        # Zurück zu Tensor
                        result = TVF.to_tensor(pil_img).to(device)
                    
                    # 2. Segment- oder Verbrennungseffekte
                    if random.random() < 0.5:
                        # Segmentbasierte Augmentierung
                        result = segment_effect(result)
                    else:
                        # Verbrennungseffekt für das gesamte Bild
                        if random.random() < 0.7:
                            result = burning_effect(result)
                    
                    # 3. Ofeneffekte hinzufügen
                    if random.random() < 0.6:
                        result = oven_effect(result)
                    
                    # 4. Optional: Mischung mit anderem Bild
                    if random.random() < 0.3 and len(all_images) > 1:
                        # Wähle zufälliges Bild (nicht das aktuelle)
                        other_indices = [i for i in range(len(all_images)) if i != idx]
                        other_idx = random.choice(other_indices)
                        other_img = all_images[other_idx]
                        
                        # Wähle Mix-Methode
                        mix_methods = ['mixup', 'cutmix', 'copypaste']
                        method = random.choice(mix_methods)
                        
                        # Importiere temporär aus der vorherigen Funktion
                        if method == 'mixup':
                            alpha = random.uniform(0.1, 0.3)  # Geringere Mischung für subtileren Effekt
                            lam = np.random.beta(alpha, alpha)
                            result = lam * result + (1 - lam) * other_img
                        elif method == 'cutmix':
                            # Vereinfachtes CutMix für Kombination
                            h, w = result.shape[1:3]
                            center_x, center_y = w // 2, h // 2
                            
                            # Erzeuge Kreismaske
                            y_coords, x_coords = torch.meshgrid(
                                torch.arange(h, device=device),
                                torch.arange(w, device=device),
                                indexing='ij'
                            )
                            
                            radius = random.uniform(0.2, 0.4) * min(h, w) / 2
                            dist = torch.sqrt(((x_coords - center_x)**2 + (y_coords - center_y)**2).float())
                            mask = (dist <= radius)
                            
                            # Erweitere Maske für alle Kanäle
                            mask_3d = mask.unsqueeze(0).expand_as(result)
                            
                            # Anwenden der Maske
                            result = torch.where(mask_3d, other_img, result)
                    
                    augmented_images.append(result)
                    
                    # Speichere einige Beispiele für die Anzeige
                    if variant_idx == 0 and len(sample_images) < 10:
                        sample_images.append(result)
                
                except Exception as e:
                    print(f"Fehler bei der Kombinations-Augmentierung: {e}")
                
                # Batch-weise Speicherfreigabe
                if len(augmented_images) >= batch_size:
                    save_augmented_images(
                        augmented_images, 
                        output_dir, 
                        f"combined_{len(augmented_images)}", 
                        batch_size
                    )
                    
                    # Speicher freigeben
                    augmented_images = []
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    
    # Restliche Bilder speichern
    if augmented_images:
        save_augmented_images(
            augmented_images, 
            output_dir, 
            f"combined_{len(augmented_images)}", 
            batch_size
        )
    
    # Zeige Beispiele an
    if show_samples and sample_images:
        show_images(
            sample_images[:9],
            titles=["Kombiniert"] * len(sample_images[:9]),
            save_path=os.path.join(output_dir, "..", "samples_combined.jpg")
        )
    
    generated_count = count - len(augmented_images)
    print(f"Kombinations-Augmentierung abgeschlossen: {generated_count} Bilder erstellt")
    return generated_count

# ================ HAUPTPROGRAMM ================

def main():
    # Kommandozeilen-Argumente parsen
    try:
        args = parse_arguments()
    except:
        # Fallback für Verwendung als importiertes Modul
        class Args:
            def __init__(self):
                self.input_dir = './pizza_images'
                self.output_dir = './augmented_pizza'
                self.seed = None
                self.img_per_original = 40
                self.batch_size = 16
                self.use_gpu = True
                self.save_stats = True
                self.show_samples = False
        args = Args()
    
    # Umgebung einrichten
    device, scipy_available = setup_environment(args)
    
    # Pfade vorbereiten
    try:
        input_dir, output_dir, subdirs = validate_and_prepare_paths(args)
    except Exception as e:
        print(f"Fehler bei der Vorbereitung der Verzeichnisse: {e}")
        sys.exit(1)
    
    # Bilder sammeln
    try:
        image_files = get_image_files(input_dir)
    except Exception as e:
        print(f"Fehler beim Sammeln der Bilder: {e}")
        sys.exit(1)
    
    # Optimale Batch-Größe bestimmen
    batch_size = get_optimal_batch_size(args.batch_size, device)
    
    # Statistik-Tracker initialisieren
    if args.save_stats:
        stats = AugmentationStats(output_dir)
        stats.set_input_count(len(image_files))
    
    # Gesamtzahl der zu erstellenden Bilder
    total_images = args.img_per_original * len(image_files)
    
    # Strategie für die Bildverteilung
    distribution = {
        'basic': 0.25,      # 25% grundlegende Augmentierung
        'burning': 0.20,    # 20% Verbrennungseffekte
        'mixed': 0.15,      # 15% gemischte Bilder
        'progression': 0.15, # 15% Progressionen
        'segment': 0.10,    # 10% segmentierte Bilder
        'combined': 0.15    # 15% kombinierte Effekte
    }
    
    # Anzahl der Bilder pro Kategorie berechnen
    augmentation_counts = {
        cat: int(total_images * pct) for cat, pct in distribution.items()
    }
    
    # Restliche Bilder der Basis-Augmentierung zuweisen
    remaining = total_images - sum(augmentation_counts.values())
    augmentation_counts['basic'] += remaining
    
    print(f"Erzeuge insgesamt {total_images} augmentierte Bilder mit folgender Verteilung:")
    for cat, count in augmentation_counts.items():
        print(f"  - {cat.capitalize()}: {count} Bilder ({count/total_images*100:.1f}%)")
    
    # Dataset erstellen
    # Minimaler Cache für häufig verwendete Bilder
    cache_size = min(100, len(image_files))
    dataset = PizzaAugmentationDataset(image_files, device=device, cache_size=cache_size)
    
    try:
        # 1. Grundlegende Augmentierung
        print("\n=== Basis-Augmentierung ===")
        basic_count = apply_basic_augmentation(
            dataset, 
            subdirs['basic'],
            augmentation_counts['basic'],
            batch_size,
            show_samples=args.show_samples
        )
        if args.save_stats:
            stats.update('basic', basic_count)
        
        # 2. Verbrennungseffekte
        print("\n=== Verbrennungs-Augmentierung ===")
        burn_count = apply_burning_augmentation(
            dataset, 
            subdirs['burnt'],
            augmentation_counts['burning'],
            batch_size,
            show_samples=args.show_samples
        )
        if args.save_stats:
            stats.update('burning', burn_count)
        
        # 3. Gemischte Bilder
        print("\n=== Mix-Augmentierung ===")
        mix_count = apply_mixed_augmentation(
            dataset, 
            subdirs['mixed'],
            augmentation_counts['mixed'],
            batch_size,
            show_samples=args.show_samples
        )
        if args.save_stats:
            stats.update('mixed', mix_count)
        
        # 4. Verbrennungsprogression
        print("\n=== Progressions-Augmentierung ===")
        progression_count = apply_progression_augmentation(
            dataset, 
            subdirs['progression'],
            augmentation_counts['progression'],
            batch_size,
            show_samples=args.show_samples
        )
        if args.save_stats:
            stats.update('progression', progression_count)
        
        # 5. Segment-basierte Augmentierung
        print("\n=== Segment-Augmentierung ===")
        segment_count = apply_segment_augmentation(
            dataset, 
            subdirs['segment'],
            augmentation_counts['segment'],
            batch_size,
            show_samples=args.show_samples
        )
        if args.save_stats:
            stats.update('segment', segment_count)
        
        # 6. Kombinierte Augmentierung
        print("\n=== Kombinations-Augmentierung ===")
        combined_count = apply_combination_augmentation(
            dataset, 
            subdirs['combined'],
            augmentation_counts['combined'],
            batch_size,
            device,
            show_samples=args.show_samples
        )
        if args.save_stats:
            stats.update('combined', combined_count)
        
        # Gesamtergebnis
        total_generated = basic_count + burn_count + mix_count + progression_count + segment_count + combined_count
        print(f"\n=== Augmentierung abgeschlossen ===")
        print(f"Insgesamt {total_generated} augmentierte Pizza-Bilder erstellt.")
        print(f"Ergebnis: {total_generated / len(image_files):.1f}x Vergrößerung des Datensatzes")
        print(f"Alle augmentierten Bilder wurden im Verzeichnis '{output_dir}' gespeichert.")
        
        # Speichere Statistiken
        if args.save_stats:
            stats.save()
    
    except KeyboardInterrupt:
        print("\nAugmentierung vom Benutzer unterbrochen!")
    except Exception as e:
        print(f"Fehler während der Augmentierung: {e}")
        import traceback
        traceback.print_exc()
    
    # Speicher freigeben
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Starte die Zeitmessung
    start_time = time.time()
    
    # Hauptprogramm ausführen
    main()
    
    # Zeige Gesamtlaufzeit
    elapsed_time = time.time() - start_time
    print(f"Gesamtlaufzeit: {elapsed_time:.2f} Sekunden ({elapsed_time/60:.2f} Minuten)")