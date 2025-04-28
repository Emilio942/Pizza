# -*- coding: utf-8 -*- # Sicherstellen, dass Umlaute korrekt interpretiert werden

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
import warnings

# Deaktiviere spezifische UserWarnings von PIL (optional, falls störend)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "Palette images with Transparency", UserWarning)

# =============================================================================
# BLOCK 0: Grundlegende Konfiguration & Setup
# =============================================================================

# Setze Startmethode für Multiprocessing (wichtig für CUDA & Stabilität)
# 'spawn' ist sicherer als 'fork' (Default unter Linux) bei CUDA-Nutzung.
try:
     # force=True überschreibt evtl. bereits gesetzte Methoden (nützlich in interaktiven Umgebungen)
     torch.multiprocessing.set_start_method('spawn', force=True)
     print("Multiprocessing Startmethode auf 'spawn' gesetzt.")
except RuntimeError as e:
     print(f"Hinweis: Konnte Multiprocessing Startmethode nicht auf 'spawn' setzen (evtl. schon gesetzt oder nicht unterstützt): {e}")
     pass

# =============================================================================
# BLOCK 1: utils.py - Hilfsfunktionen
# =============================================================================

# --- Geräteerkennung ---
def get_device():
    """Erkennt das beste verfügbare Gerät (CUDA, MPS, CPU) und gibt es zurück."""
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Verwende Gerät: NVIDIA CUDA - {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            try:
                _ = torch.tensor([1], device='mps') + torch.tensor([1], device='mps')
                device = torch.device('mps')
                print("Verwende Gerät: Apple Metal Performance Shaders (MPS)")
            except Exception as mps_err:
                print(f"WARNUNG: MPS verfügbar, aber Test fehlgeschlagen ({mps_err}). Fallback auf CPU.")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
            print("Verwende Gerät: CPU")
    except Exception as e:
        print(f"FEHLER bei der Geräteerkennung: {e}. Fallback auf CPU.")
        device = torch.device('cpu')
    return device

device = get_device() # Globale Variable für das Gerät

# --- Speicherverwaltung ---
def clean_memory():
    """Gibt ungenutzten Speicher frei (CPU und GPU)."""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Aktuell keine direkte 'empty_cache' für MPS bekannt
    except Exception as e:
        print(f"Fehler während Speicherbereinigung: {e}")


# --- Bild Laden ---
@contextmanager
def open_image(path, default_background_color=(128, 128, 128)):
    """
    Kontext-Manager für sicheres Öffnen und Schließen von PIL Bildern.
    Konvertiert immer zu RGB. Verwendet eine definierte Hintergrundfarbe
    für transparente Bereiche. Gibt None zurück bei Fehlern.

    Args:
        path (str): Pfad zur Bilddatei.
        default_background_color (tuple): RGB-Tupel für den Hintergrund bei Transparenz.
                                          Default ist Grau (128, 128, 128).
                                          Ändern Sie dies, wenn ein anderer Hintergrund
                                          realistischer für Ihre Daten ist.
    """
    img = None
    try:
        img = Image.open(path)
        img.load() # Stelle sicher, dass Bilddaten geladen sind

        if img.mode == 'RGB':
            # Bereits im richtigen Format
            yield img
            return # Frühzeitig beenden

        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            # Bild hat Transparenz, erstelle neuen Hintergrund und füge Bild ein
            try:
                # Konvertiere das Originalbild (ohne Alpha) separat
                rgb_img_part = img.convert('RGB')

                # Extrahiere Alpha-Maske
                if img.mode == 'RGBA':
                    mask = img.getchannel('A')
                elif img.mode == 'LA':
                    mask = img.getchannel('A')
                    # Konvertiere Graustufen+Alpha zu RGB
                    rgb_img_part = img.convert('L').convert('RGB')
                elif img.mode == 'P':
                    # Versuche Transparenzinfo aus Palette zu nutzen
                    # Dies ist oft komplex und kann fehlschlagen
                    try:
                         mask = img.convert('RGBA').getchannel('A')
                    except ValueError: # Fallback wenn RGBA Konvertierung scheitert
                         print(f"Warnung: Konnte Transparenzmaske für Palettenbild {os.path.basename(path)} nicht zuverlässig extrahieren. Fülle mit Hintergrundfarbe.")
                         img = Image.new('RGB', img.size, default_background_color) # Kompletter Hintergrund
                         yield img
                         return
                else: # Sollte nicht passieren, aber sicherheitshalber
                    mask = None

                # Erstelle Hintergrund und füge Bild mit Maske ein
                background = Image.new('RGB', img.size, default_background_color)
                if mask:
                    background.paste(rgb_img_part, (0, 0), mask)
                    img_processed = background
                else: # Falls keine Maske extrahiert werden konnte
                    img_processed = rgb_img_part # Nur den RGB Teil nehmen

                yield img_processed

            except Exception as alpha_err:
                print(f"Warnung: Fehler bei Verarbeitung von Transparenz in {os.path.basename(path)} ({alpha_err}). Konvertiere direkt zu RGB (kann Artefakte erzeugen).")
                # Fallback: Direkte Konvertierung, kann unerwünschte Ergebnisse liefern
                yield img.convert('RGB')

        else:
            # Andere Modi (z.B. L, CMYK) direkt nach RGB konvertieren
            yield img.convert('RGB')

    except FileNotFoundError:
        print(f"FEHLER: Bilddatei nicht gefunden: {path}")
        yield None
    except Image.DecompressionBombError:
        print(f"FEHLER: Bilddatei zu groß (Decompression Bomb): {path}")
        yield None
    except Exception as e:
        print(f"FEHLER beim Öffnen/Verarbeiten von Bild {path}: {type(e).__name__} - {e}")
        yield None
    finally:
        if img:
            try:
                img.close()
            except Exception:
                pass # Ignoriere Fehler beim Schließen

# --- Bild Anzeigen ---
def show_images(images, titles=None, cols=5, figsize=(15, 10), denormalize_std_mean=None):
    """
    Zeigt mehrere Bilder (PIL, NumPy, Tensor) in einem Grid an.
    Kann optional Denormalisierung anwenden, wenn std/mean gegeben sind.
    Stellt sicher, dass die Denormalisierungs-Parameter korrekt sind!
    """
    # (Code unverändert, da Analyse keine Fehler fand, nur Hinweis auf korrekte std/mean)
    if not images:
        print("Keine Bilder zum Anzeigen vorhanden.")
        return

    num_images = len(images)
    rows = (num_images + cols - 1) // cols
    plt.figure(figsize=figsize)

    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        img_display = None
        title = titles[i] if titles and i < len(titles) else f"Image {i+1}"

        try:
            if isinstance(img, torch.Tensor):
                img_tensor = img.detach().cpu().float() # Sicherstellen: CPU, float, keine Gradients

                # Optional: Denormalisierung
                if denormalize_std_mean:
                    # WICHTIG: Diese Werte müssen exakt der verwendeten Normalisierung entsprechen!
                    mean = torch.tensor(denormalize_std_mean['mean']).view(-1, 1, 1)
                    std = torch.tensor(denormalize_std_mean['std']).view(-1, 1, 1)
                    img_tensor = img_tensor * std + mean

                # Konvertiere Tensor zu NumPy für die Anzeige
                if img_tensor.ndim == 3 and img_tensor.shape[0] in [1, 3]: # (C, H, W)
                    img_display = img_tensor.permute(1, 2, 0).numpy()
                elif img_tensor.ndim == 2: # (H, W) Grayscale
                    img_display = img_tensor.numpy()
                elif img_tensor.ndim == 4 and img_tensor.shape[0] == 1: # (1, C, H, W) -> (C, H, W)
                    img_display = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
                else:
                    print(f"Warnung: Unerwartete Tensorform: {img.shape}")
                    title += "\n(Fehlerhafte Form)"
                    img_display = np.zeros((50, 50)) # Placeholder

            elif isinstance(img, np.ndarray):
                img_display = img

            elif isinstance(img, Image.Image):
                img_display = np.array(img) # Konvertiere PIL zu NumPy für imshow

            else:
                print(f"Warnung: Unbekannter Bildtyp: {type(img)}")
                title += "\n(Unbekannter Typ)"
                img_display = np.zeros((50,50)) # Placeholder

            # Clamping und Typ-Anpassung für imshow
            if img_display is not None:
                # Geänderte Logik: Prüfe ob Wertebereich eher [0,1] oder [0,255] ist
                if img_display.max() <= 1.1 and img_display.min() >= -0.1: # Toleranz für float-Ungenauigkeiten -> [0, 1] Bereich
                    img_display = np.clip(img_display, 0.0, 1.0)
                elif img_display.dtype != np.uint8 : # Wahrscheinlich [0, 255] Bereich aber falscher Typ
                    img_display = np.clip(img_display, 0, 255).astype(np.uint8)
                # Falls bereits uint8, wird angenommen, dass es [0,255] ist

                # Handhabung von Single-Channel für cmap
                cmap = 'gray' if img_display.ndim == 2 or (img_display.ndim == 3 and img_display.shape[-1] == 1) else None
                if img_display.ndim == 3 and img_display.shape[-1] == 1:
                    img_display = img_display.squeeze(-1) # (H, W, 1) -> (H, W)

                plt.imshow(img_display, cmap=cmap)

        except Exception as e:
            print(f"FEHLER beim Vorbereiten von Bild {i} für die Anzeige: {e}")
            plt.title(f"{title}\n(Anzeigefehler)")
            plt.imshow(np.zeros((50,50)), cmap='gray') # Error placeholder

        plt.axis('off')
        plt.title(title)

    plt.tight_layout()
    plt.show()


# --- Bild Speichern ---
def save_augmented_images(images_data, output_dir, base_filename, batch_size=16, start_idx=0, quality=90, save_format='jpg'):
    """
    Speichert augmentierte Bilder (Tensoren, NumPy, PIL) in Batches.
    Gibt die Anzahl der erfolgreich gespeicherten Bilder zurück.
    `images_data` kann eine Liste oder ein Batch-Tensor sein.

    WICHTIG: Erwartet Tensoren im Wertebereich [0, 1]! Keine automatische
             Denormalisierung mehr enthalten, da dies zu fehlerhaften
             Bildern führte, wenn die Normalisierungsstatistik unbekannt war.
             Stellen Sie sicher, dass die übergebenen Tensoren [0, 1] sind.

    Args:
        images_data (list | torch.Tensor): Liste oder Batch von Bildern.
        output_dir (str): Ausgabeordner.
        base_filename (str): Basisname für gespeicherte Dateien.
        batch_size (int): Größe der Speicher-Batches (nur für interne Verarbeitung).
        start_idx (int): Startindex für die Nummerierung der Dateien.
        quality (int): Qualität für JPEG-Speicherung (1-100).
        save_format (str): Zielformat ('jpg', 'png').
    """
    if not images_data:
        return 0

    os.makedirs(output_dir, exist_ok=True)
    saved_count = 0
    save_format = save_format.lower()
    if save_format not in ['jpg', 'jpeg', 'png']:
        print(f"Warnung: Ungültiges Speicherformat '{save_format}'. Verwende 'jpg'.")
        save_format = 'jpg'
    file_extension = 'jpg' if save_format in ['jpg', 'jpeg'] else 'png'


    # Konvertiere Tensor-Batch in Liste von Tensoren auf CPU
    if isinstance(images_data, torch.Tensor):
        images_list = list(images_data.cpu())
    elif isinstance(images_data, (list, tuple)):
        images_list = images_data
    else:
        print(f"FEHLER: Ungültiger Datentyp für save_augmented_images: {type(images_data)}")
        return 0

    num_images = len(images_list)
    pbar = tqdm(total=num_images, desc=f"Speichere '{base_filename}' (.{file_extension})", leave=False, unit="img")

    for batch_idx in range(0, num_images, batch_size):
        batch_end = min(batch_idx + batch_size, num_images)
        current_batch = images_list[batch_idx:batch_end]
        pil_images_to_save = []

        for i, img_data in enumerate(current_batch):
            global_idx = start_idx + batch_idx + i
            pil_img = None
            try:
                if isinstance(img_data, torch.Tensor):
                    img_tensor = img_data.detach().cpu().float()
                    # KORREKTUR: Keine Denormalisierung mehr! Erwarte [0, 1].
                    # Clampe auf [0, 1] und konvertiere zu PIL [0, 255]
                    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
                    pil_img = TVF.to_pil_image(img_tensor)

                elif isinstance(img_data, np.ndarray):
                    # Stelle korrekten Datentyp und Range sicher
                    if img_data.max() <= 1.0 and img_data.min() >= -0.1: # Toleranz für Float
                        img_to_save = (np.clip(img_data, 0.0, 1.0) * 255).astype(np.uint8)
                    elif img_data.dtype != np.uint8:
                        img_to_save = np.clip(img_data, 0, 255).astype(np.uint8)
                    else:
                        img_to_save = img_data # Bereits uint8
                    pil_img = Image.fromarray(img_to_save)

                elif isinstance(img_data, Image.Image):
                    # Stelle sicher, dass PIL im RGB Modus ist
                    pil_img = img_data.convert('RGB') if img_data.mode != 'RGB' else img_data

                else:
                     print(f"Warnung: Überspringe Speichern für unbekannten Datentyp: {type(img_data)} bei Index {global_idx}")
                     continue

                if pil_img:
                    save_path = os.path.join(output_dir, f"{base_filename}_{global_idx:06d}.{file_extension}")
                    pil_images_to_save.append((pil_img, save_path))

            except Exception as e:
                print(f"FEHLER bei der Konvertierung von Bild {global_idx} für das Speichern: {type(e).__name__} - {e}")

        # Speichere die konvertierten PIL Bilder
        for pil_img, save_path in pil_images_to_save:
             try:
                 save_options = {}
                 if file_extension == 'jpg':
                     save_options['quality'] = quality
                     save_options['optimize'] = True
                 elif file_extension == 'png':
                     # PNG Optionen können hier hinzugefügt werden (z.B. compress_level)
                     save_options['optimize'] = True # Kann auch für PNG nützlich sein

                 pil_img.save(save_path, **save_options)
                 saved_count += 1
                 pbar.update(1)
             except Exception as e:
                 print(f"FEHLER beim Speichern von Bild {save_path}: {type(e).__name__} - {e}")
             # PIL Images werden nicht explizit geschlossen hier

        del current_batch, pil_images_to_save
        # clean_memory() # Nur aufrufen, wenn Speicher wirklich knapp wird

    pbar.close()
    # clean_memory() # Optional: Einmal am Ende des Speicherns
    return saved_count


# =============================================================================
# BLOCK 2: dataset.py - Dataset Klasse
# =============================================================================

class PizzaAugmentationDataset(Dataset):
    """
    PyTorch Dataset-Klasse für das Laden von Pizza-Bildern.
    Kann optional initiale Transformationen anwenden.
    Gibt bei Ladefehlern einen Dummy-Tensor zurück.
    Nutzt die verbesserte `open_image` Funktion.
    """
    def __init__(self, image_paths, labels=None, transform=None, target_size=(256, 256), transparent_background_color=(128, 128, 128)):
        # (Code größtenteils unverändert, fügt aber background_color hinzu)
        if not isinstance(image_paths, list):
             raise TypeError("image_paths muss eine Liste sein.")
        self.image_paths = image_paths
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        self.transform = transform
        self.target_size = tuple(target_size)
        self.transparent_background_color = transparent_background_color

        self.error_tensor = torch.zeros(3, *self.target_size, dtype=torch.float32)
        self.error_label = torch.tensor(-1, dtype=torch.long) if self.labels is not None else -1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx >= len(self.image_paths):
             raise IndexError("Index außerhalb des gültigen Bereichs")
        image_path = self.image_paths[idx]

        try:
            # Verwende die verbesserte open_image Funktion
            with open_image(image_path, default_background_color=self.transparent_background_color) as img:
                if img is None:
                    print(f"Warnung: Überspringe Index {idx}, da Bild {os.path.basename(image_path)} nicht geladen werden konnte.")
                    if self.labels is not None:
                        return self.error_tensor, self.error_label
                    else:
                        return self.error_tensor

                # Wende Transformation an
                if self.transform:
                    image = self.transform(img)
                else:
                    # Standard: Resize und ToTensor [0, 1]
                    image = TVF.resize(img, list(self.target_size), antialias=True)
                    image = TVF.to_tensor(image)

                # Sicherheitscheck: Endgröße (defensiv)
                if image.shape[1:] != self.target_size:
                     # Diese Warnung sollte untersucht werden, wenn sie häufig auftritt!
                     print(f"Warnung: Bild {os.path.basename(image_path)} hat nach Transform falsche Größe {image.shape[1:]}, interpoliere zu {self.target_size}.")
                     image = TF.interpolate(image.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)

        except Exception as e:
            print(f"FEHLER bei Verarbeitung/Transformation von Index {idx} ({os.path.basename(image_path)}): {type(e).__name__} - {e}")
            if self.labels is not None:
                return self.error_tensor, self.error_label
            else:
                return self.error_tensor

        # Rückgabe
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image


# =============================================================================
# BLOCK 3: augmentations.py - Kernlogik der Augmentierungen
# =============================================================================

class PizzaBurningEffect(nn.Module):
    """
    Simuliert Verbrennungseffekte auf Pizza-Bildern (Batch-fähig, GPU-optimiert).
    Kombiniert Kantenverdunklung und zufällige Brandflecken.
    Erwartet Input-Tensoren im Bereich [0, 1].
    Brandfarbe ist nun parameterisierbar.
    """
    def __init__(self, burn_intensity_min=0.2, burn_intensity_max=0.8,
                 edge_focus=2.0, spot_radius_min=0.05, spot_radius_max=0.2,
                 spot_count_min=3, spot_count_max=8,
                 burn_color_mult=(0.6, 0.3, 0.1)): # KORREKTUR: Parameter für Farbe
        super().__init__()
        self.burn_intensity_min = max(0.0, burn_intensity_min)
        self.burn_intensity_max = min(1.0, burn_intensity_max)
        self.edge_focus = edge_focus
        self.spot_radius_min = spot_radius_min
        self.spot_radius_max = spot_radius_max
        self.spot_count_min = spot_count_min
        self.spot_count_max = spot_count_max
        # KORREKTUR: Speichere Farbmultiplikatoren
        if len(burn_color_mult) != 3:
             raise ValueError("burn_color_mult muss ein Tupel/Liste mit 3 Werten sein (R, G, B Multiplikatoren).")
        self.burn_color_mult = burn_color_mult
        # Hinweis: Das Kantenfokus-Verhalten (`pow`) wurde in der Analyse diskutiert.
        # Es wird hier beibehalten, sollte aber visuell geprüft werden.

    def forward(self, img_batch):
        # (Validierung und Maskenberechnung wie zuvor)
        if not isinstance(img_batch, torch.Tensor):
            raise TypeError("Input muss ein PyTorch Tensor sein.")
        if img_batch.ndim != 4:
             raise ValueError(f"Input Tensor muss 4D sein (B, C, H, W), bekam {img_batch.ndim}D.")
        if img_batch.min() < -0.1 or img_batch.max() > 1.1: # Toleranz
             print("Warnung (PizzaBurningEffect): Input-Tensor scheint nicht im Bereich [0, 1] zu sein. Ergebnisse könnten unerwartet sein.")

        img_batch = img_batch.to(device) # Sicherstellen, dass auf dem richtigen Gerät
        b, c, h, w = img_batch.shape
        dtype = img_batch.dtype
        results = torch.empty_like(img_batch)
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device, dtype=dtype),
            torch.linspace(-1, 1, w, device=device, dtype=dtype),
            indexing='ij'
        )
        dist_from_center = torch.sqrt(x_coords**2 + y_coords**2)
        # Visuell prüfen, ob diese Kantenfokus-Funktion passt:
        edge_weight = (dist_from_center / (dist_from_center.max() + 1e-6)).pow(self.edge_focus * 4)
        min_ew, max_ew = edge_weight.min(), edge_weight.max()
        edge_weight = (edge_weight - min_ew) / (max_ew - min_ew + 1e-6)

        for i in range(b):
            burn_intensity = random.uniform(self.burn_intensity_min, self.burn_intensity_max)
            num_spots = random.randint(self.spot_count_min, self.spot_count_max)
            spots_mask = torch.zeros_like(dist_from_center, device=device)
            if num_spots > 0:
                spot_x = torch.rand(num_spots, device=device) * 2 - 1
                spot_y = torch.rand(num_spots, device=device) * 2 - 1
                spot_radius = torch.rand(num_spots, device=device) * (self.spot_radius_max - self.spot_radius_min) + self.spot_radius_min
                spot_strength = torch.rand(num_spots, device=device) * 0.5 + 0.5
                dist_sq = (x_coords.unsqueeze(-1) - spot_x)**2 + (y_coords.unsqueeze(-1) - spot_y)**2
                gauss_spots = torch.exp(-dist_sq / (2 * (spot_radius**2) + 1e-6)) * spot_strength
                spots_mask, _ = torch.max(gauss_spots, dim=-1)

            combined_mask = torch.maximum(edge_weight * 0.8, spots_mask)
            final_burn_mask = torch.clamp(combined_mask * burn_intensity, 0.0, 1.0)
            final_burn_mask_expanded = final_burn_mask.unsqueeze(0).expand(c, h, w)

            # --- Wende den Verbrennungseffekt an ---
            current_img = img_batch[i] # (C, H, W)

            # KORREKTUR: Verwende die parameterisierten Farbmultiplikatoren
            dark_color = torch.stack([
                current_img[0] * self.burn_color_mult[0],
                current_img[1] * self.burn_color_mult[1],
                current_img[2] * self.burn_color_mult[2]
            ], dim=0)

            burnt_img = current_img * (1 - final_burn_mask_expanded) + dark_color * final_burn_mask_expanded
            results[i] = torch.clamp(burnt_img, 0.0, 1.0) # Ergebnis ist [0, 1]

        return results

class SimpleOvenEffect(nn.Module):
    """
    Simuliert einfache Ofeneffekte (Dampf, Wärme, Schatten) auf Bild-Batches.
    GPU-optimiert und Batch-fähig. Erwartet Input-Tensoren im Bereich [0, 1].
    Hinweis: Der Dampf-Weichzeichner verwendet eine Schleife und ist bei
             sehr großen Batches möglicherweise ineffizient.
    """
    # (Code unverändert, aber Kommentar hinzugefügt)
    def __init__(self, steam_prob=0.3, warmth_prob=0.3, shadow_prob=0.2, steam_blur_kernel_min=9, steam_blur_kernel_max=25, steam_blur_sigma_min=5, steam_blur_sigma_max=15):
        super().__init__()
        self.steam_prob = steam_prob
        self.warmth_prob = warmth_prob
        self.shadow_prob = shadow_prob
        self.steam_blur_kernel_min = steam_blur_kernel_min
        self.steam_blur_kernel_max = steam_blur_kernel_max
        self.steam_blur_sigma_min = steam_blur_sigma_min
        self.steam_blur_sigma_max = steam_blur_sigma_max
        if self.steam_blur_kernel_min % 2 == 0: self.steam_blur_kernel_min += 1
        if self.steam_blur_kernel_max % 2 == 0: self.steam_blur_kernel_max += 1

    def forward(self, img_batch):
        # (Validierung wie zuvor)
        if not isinstance(img_batch, torch.Tensor): raise TypeError("Input muss ein PyTorch Tensor sein.")
        if img_batch.ndim != 4: raise ValueError(f"Input Tensor muss 4D (B, C, H, W), bekam {img_batch.ndim}D.")
        if img_batch.min() < -0.1 or img_batch.max() > 1.1: print("Warnung (SimpleOvenEffect): Input-Tensor scheint nicht im Bereich [0, 1] zu sein.")

        img_batch = img_batch.to(device)
        b, c, h, w = img_batch.shape
        dtype = img_batch.dtype
        result_batch = img_batch.clone()

        # --- Dampf-Effekt ---
        apply_steam = torch.rand(b, device=device) < self.steam_prob
        steam_indices = torch.where(apply_steam)[0]
        num_steam = len(steam_indices)
        if num_steam > 0:
            steam_opacity = torch.rand(num_steam, 1, 1, 1, device=device, dtype=dtype) * 0.2 + 0.1
            kernel_sizes = torch.randint(self.steam_blur_kernel_min // 2, self.steam_blur_kernel_max // 2 + 1, (num_steam,), device=device) * 2 + 1
            sigmas = torch.rand(num_steam, device=device) * (self.steam_blur_sigma_max - self.steam_blur_sigma_min) + self.steam_blur_sigma_min
            y_gradient = torch.linspace(1.0, 0.2, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(num_steam, 1, h, w)
            steam_base = torch.rand(num_steam, 1, h, w, device=device, dtype=dtype) * y_gradient
            steam_blurred = torch.empty_like(steam_base)
            # TODO: Ineffiziente Schleife für Dampf-Blur, wenn variierende Kernel/Sigmas pro Bild benötigt werden.
            #       Bei Performance-Problemen ggf. kornia oder eigene Implementierung prüfen.
            for i in range(num_steam):
                 idx_in_batch = steam_indices[i] # Index im Original-Batch
                 k = int(kernel_sizes[i].item())
                 s = float(sigmas[i].item())
                 # Wende auf das entsprechende Slice im steam_base an
                 steam_blurred[i:i+1] = TVF.gaussian_blur(steam_base[i:i+1], kernel_size=k, sigma=s)

            steam_mask = (steam_blurred * steam_opacity).expand(-1, c, -1, -1)
            steam_color = torch.full_like(result_batch[steam_indices], 0.95, device=device)
            result_batch[steam_indices] = result_batch[steam_indices] * (1 - steam_mask) + steam_color * steam_mask

        # --- Wärme-Effekt ---
        apply_warmth = torch.rand(b, device=device) < self.warmth_prob
        warmth_indices = torch.where(apply_warmth)[0]
        num_warmth = len(warmth_indices)
        if num_warmth > 0:
            warmth_factor = torch.rand(num_warmth, 1, 1, 1, device=device, dtype=dtype) * 0.15 + 0.05
            result_batch[warmth_indices, 0] = torch.clamp(result_batch[warmth_indices, 0] * (1 + warmth_factor), 0, 1)
            result_batch[warmth_indices, 2] = torch.clamp(result_batch[warmth_indices, 2] * (1 - warmth_factor * 0.5), 0, 1)

        # --- Schatten-Effekt ---
        apply_shadow = torch.rand(b, device=device) < self.shadow_prob
        shadow_indices = torch.where(apply_shadow)[0]
        num_shadow = len(shadow_indices)
        if num_shadow > 0:
            shadow_opacity = torch.rand(num_shadow, 1, 1, 1, device=device, dtype=dtype) * 0.3 + 0.1
            shadow_rel_x = torch.rand(num_shadow, 1, 1, 1, device=device, dtype=dtype) * 1.6 - 0.8
            shadow_rel_y = torch.rand(num_shadow, 1, 1, 1, device=device, dtype=dtype) * 1.6 - 0.8
            shadow_center_x = (shadow_rel_x + 1) / 2 * w
            shadow_center_y = (shadow_rel_y + 1) / 2 * h
            diagonal = np.sqrt(h**2 + w**2)
            shadow_radius = (torch.rand(num_shadow, 1, 1, 1, device=device, dtype=dtype) * 0.3 + 0.4) * diagonal
            y_coords_s, x_coords_s = torch.meshgrid(torch.arange(h, device=device, dtype=dtype), torch.arange(w, device=device, dtype=dtype), indexing='ij')
            dist_sq_s = (x_coords_s.unsqueeze(0) - shadow_center_x)**2 + (y_coords_s.unsqueeze(0) - shadow_center_y)**2
            shadow_mask = torch.exp(-torch.sqrt(dist_sq_s + 1e-6) / (shadow_radius + 1e-6)) * shadow_opacity
            shadow_mask = shadow_mask.unsqueeze(1).expand(-1, c, -1, -1)
            result_batch[shadow_indices] = result_batch[shadow_indices] * (1 - shadow_mask)

        return torch.clamp(result_batch, 0.0, 1.0) # Ergebnis ist [0, 1]

# --- MixUp --- (Keine Änderungen nötig basierend auf Analyse)
def pizza_mixup(img_batch1, img_batch2, alpha=0.3):
    """
    Implementiert effizientes MixUp für Pizza-Bild-Batches auf der GPU.
    Gibt gemischten Batch [0, 1] und die Lambdas zurück.
    """
    # (Code unverändert)
    if not isinstance(img_batch1, torch.Tensor): raise TypeError("img_batch1 must be a tensor")
    if not isinstance(img_batch2, torch.Tensor): raise TypeError("img_batch2 must be a tensor")
    img_batch1 = img_batch1.to(device)
    img_batch2 = img_batch2.to(device)

    if img_batch1.shape != img_batch2.shape:
        print(f"Warnung: MixUp Batches have different shapes ({img_batch1.shape} vs {img_batch2.shape}). Resizing batch 2.")
        try:
            img_batch2 = TF.interpolate(img_batch2, size=img_batch1.shape[2:], mode='bilinear', align_corners=False)
        except Exception as e:
            print(f"FEHLER: Resizing for MixUp failed: {e}. Skipping MixUp.")
            return img_batch1, torch.ones(img_batch1.shape[0], device=device)

    lam = torch.tensor(np.random.beta(alpha, alpha, size=img_batch1.shape[0]), dtype=img_batch1.dtype, device=device)
    lam_expanded = lam.view(-1, 1, 1, 1)
    mixed_batch = lam_expanded * img_batch1 + (1 - lam_expanded) * img_batch2
    mixed_batch = torch.clamp(mixed_batch, 0.0, 1.0) # Ergebnis ist [0, 1]
    return mixed_batch, lam

# --- CutMix --- (Keine Änderungen nötig basierend auf Analyse)
def pizza_cutmix(img_batch1, img_batch2, alpha=1.0):
    """
    Implementiert effizientes CutMix für Pizza-Bild-Batches mit Keil- oder Kreisform auf GPU.
    Gibt gemischten Batch [0, 1] und die Lambdas zurück.
    """
    # (Code unverändert)
    if not isinstance(img_batch1, torch.Tensor): raise TypeError("img_batch1 must be a tensor")
    if not isinstance(img_batch2, torch.Tensor): raise TypeError("img_batch2 must be a tensor")
    img_batch1 = img_batch1.to(device)
    img_batch2 = img_batch2.to(device)

    if img_batch1.shape != img_batch2.shape:
        print(f"Warnung: CutMix Batches have different shapes ({img_batch1.shape} vs {img_batch2.shape}). Resizing batch 2.")
        try:
            img_batch2 = TF.interpolate(img_batch2, size=img_batch1.shape[2:], mode='bilinear', align_corners=False)
        except Exception as e:
            print(f"FEHLER: Resizing for CutMix failed: {e}. Skipping CutMix.")
            return img_batch1, torch.zeros(img_batch1.shape[0], device=device)

    b, c, h, w = img_batch1.shape
    dtype = img_batch1.dtype
    result_batch = img_batch1.clone()
    lam = torch.tensor(np.random.beta(alpha, alpha, size=b), dtype=dtype, device=device)
    y_coords, x_coords = torch.meshgrid(torch.arange(h, device=device, dtype=dtype), torch.arange(w, device=device, dtype=dtype), indexing='ij')
    center_x, center_y = w // 2, h // 2
    masks = torch.zeros(b, h, w, dtype=torch.bool, device=device)
    use_circle = torch.rand(b, device=device) < 0.5
    target_area = lam * h * w

    # Kreis-Masken
    circle_indices = torch.where(use_circle)[0]
    if len(circle_indices) > 0:
        radii_pixels = torch.sqrt(target_area[circle_indices] / np.pi)
        radii_pixels = torch.clamp(radii_pixels, 0, min(h, w) / 2)
        dist_from_center = torch.sqrt(((x_coords - center_x)**2 + (y_coords - center_y)**2))
        masks[circle_indices] = dist_from_center.unsqueeze(0) <= radii_pixels.view(-1, 1, 1)

    # Keil-Masken
    wedge_indices = torch.where(~use_circle)[0]
    if len(wedge_indices) > 0:
        wedge_radius = min(h, w) / 2
        # Vermeide Division durch Null, falls Radius 0 ist (sehr kleines Bild)
        angle_widths = 2 * target_area[wedge_indices] / (wedge_radius**2 + 1e-6)
        angle_widths = torch.clamp(angle_widths, np.pi / 12, np.pi)
        start_angles = torch.rand(len(wedge_indices), device=device) * 2 * np.pi
        delta_x = x_coords - center_x
        delta_y = y_coords - center_y
        angles = torch.atan2(delta_y, delta_x)
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        start = start_angles.view(-1, 1, 1)
        width = angle_widths.view(-1, 1, 1)
        end = (start + width) # Endwinkel kann > 2pi sein
        # Maskenberechnung mit Wrap-Around Handling (korrigiert)
        # Normalisiere Winkel in [start, start + width]
        # Check if angle is between start and end, considering wrap around 2*pi
        # Condition: (angle >= start and angle <= end) OR (end < start and (angle >= start OR angle <= end % (2*pi)))
        current_angles_expanded = angles.unsqueeze(0).expand(len(wedge_indices), h, w)
        wedge_mask = torch.zeros(len(wedge_indices), h, w, dtype=torch.bool, device=device)

        # No wrap-around case (start <= end)
        no_wrap_indices = torch.where(start <= end)[0]
        if len(no_wrap_indices)>0:
             wedge_mask[no_wrap_indices] = (current_angles_expanded[no_wrap_indices] >= start[no_wrap_indices]) & \
                                           (current_angles_expanded[no_wrap_indices] <= end[no_wrap_indices])

        # Wrap-around case (start > end, e.g., start=350deg, width=30deg -> end=380deg)
        wrap_indices = torch.where(start > end)[0]
        if len(wrap_indices)>0:
             wedge_mask[wrap_indices] = (current_angles_expanded[wrap_indices] >= start[wrap_indices]) | \
                                        (current_angles_expanded[wrap_indices] <= (end[wrap_indices] % (2 * np.pi)))


        masks[wedge_indices] = wedge_mask


    masks_expanded = masks.unsqueeze(1).expand(b, c, h, w)
    result_batch = torch.where(masks_expanded, img_batch2, result_batch) # Ergebnis ist [0, 1]
    return result_batch, lam


# --- Verbrennungs-Progression ---
def pizza_burning_progression(img_batch, num_steps=5, start_intensity=0.1, step_intensity=0.15, burn_color_mult=(0.6, 0.3, 0.1)):
    """
    Erzeugt eine Reihe von Bildern mit kontrolliert ansteigendem Verbrennungsgrad.
    Gibt einen einzelnen Tensor zurück, der alle Stufen enthält (Batch-Dimension erweitert).
    Erwartet Input-Tensoren im Bereich [0, 1] und gibt [0, 1] zurück.
    Verwendet die parameterisierte Brandfarbe.
    """
    # (Code angepasst, um burn_color_mult durchzureichen)
    if not isinstance(img_batch, torch.Tensor): raise TypeError("Input muss ein PyTorch Tensor sein.")
    if img_batch.ndim != 4: raise ValueError(f"Input Tensor muss 4D sein (B, C, H, W), bekam {img_batch.ndim}D.")
    img_batch = img_batch.to(device)

    b = img_batch.shape[0]
    all_results = [img_batch.clone()]

    # KORREKTUR: Übergebe burn_color_mult an die Effekt-Instanz
    burn_effect = PizzaBurningEffect(burn_intensity_min=0.0, burn_intensity_max=0.0, burn_color_mult=burn_color_mult).to(device)

    current_min_intensity = start_intensity
    current_max_intensity = start_intensity + step_intensity * 0.5

    for i in range(num_steps):
        intensity_min = min(max(0.0, current_min_intensity), 1.0)
        intensity_max = min(max(0.0, current_max_intensity), 1.0)
        burn_effect.burn_intensity_min = intensity_min
        burn_effect.burn_intensity_max = intensity_max

        try:
            burnt_batch_step = burn_effect(img_batch.clone())
            all_results.append(burnt_batch_step)
        except Exception as e:
            print(f"FEHLER bei Brennstufe {i+1} der Progression: {type(e).__name__} - {e}. Breche Progression ab.")
            break

        current_min_intensity += step_intensity
        current_max_intensity += step_intensity

    if len(all_results) > 1:
        flat_results = torch.cat(all_results, dim=0)
        return flat_results # Ergebnis ist [0, 1]
    else:
        return img_batch # Ergebnis ist [0, 1]


# =============================================================================
# BLOCK 4: generators.py - Generatoren für augmentierte Batches
# =============================================================================

def pizza_basic_augmentation_generator(image_paths, num_total_augmentations, batch_size=16, target_size=(256, 256), num_workers=4, pin_memory=True, prefetch_factor=2, transparent_background_color=(128,128,128)):
    """
    Generator für grundlegende Augmentierungen.
    Nutzt DataLoader für effizientes Laden und wendet torchvision Transforms an.
    Yielded Batches von Tensoren im Bereich [0, 1] (unnormalisiert).
    Hinweis: Lädt Bilder einzeln, was weniger effizient sein kann als Batch-Laden.
    """
    # (Code angepasst: Entfernt Normalisierung, fügt Background Color hinzu)
    if not image_paths: print("Warnung (BasicGen): Keine Bildpfade."); return
    if num_total_augmentations <= 0: print("Warnung (BasicGen): num_total_augmentations <= 0."); return

    num_originals = len(image_paths)
    loops_needed = max(1, (num_total_augmentations + num_originals - 1) // num_originals)
    print(f"Basis Aug: Ziel={num_total_augmentations}, Originale={num_originals}. Benötigt ca. {loops_needed} Durchläufe.")

    # Transformationen OHNE Normalisierung am Ende
    basic_transform_no_norm = transforms.Compose([
        transforms.RandomApply([transforms.RandomRotation(degrees=(-180, 180), interpolation=TVF.InterpolationMode.BILINEAR)], p=0.7),
        transforms.RandomApply([transforms.RandomResizedCrop(target_size, scale=(0.6, 1.0), ratio=(0.8, 1.2), antialias=True)], p=0.8),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)], p=0.7),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.5))], p=0.3),
        transforms.Resize(target_size, antialias=True), # Sicherstellen, dass Größe stimmt
        transforms.ToTensor(), # Konvertiert PIL [0, 255] zu Tensor [0, 1]
        # KORREKTUR: Normalisierung entfernt, da dies vor dem Speichern problematisch ist.
        # transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    fallback_transform_no_norm = transforms.Compose([
        transforms.Resize(target_size, antialias=True),
        transforms.ToTensor(), # Zu Tensor [0, 1]
        # KORREKTUR: Normalisierung entfernt
    ])

    generated_count = 0
    persistent_workers = (num_workers > 0) and (os.name != 'nt')

    for loop in range(loops_needed):
        if generated_count >= num_total_augmentations: break
        print(f"\nBasis Aug: Durchlauf {loop+1}/{loops_needed}")

        # KORREKTUR: Übergebe background_color an Dataset
        dataset = PizzaAugmentationDataset(image_paths, transform=None, target_size=target_size, transparent_background_color=transparent_background_color) # Lädt PIL
        # Hinweis: Lädt Bilder einzeln (batch_size=1), um torchvision Transforms einfach anzuwenden.
        #          Kann ein Engpass sein. Alternative: Batch-Laden + kornia/nn.Module Transforms.
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=True, num_workers=num_workers,
            pin_memory=pin_memory and (device.type == 'cuda'),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers
        )

        batch_accumulator = []
        pbar = tqdm(total=min(num_originals, num_total_augmentations - generated_count), desc=f"Basis Aug {loop+1}", unit="img")

        for data in dataloader:
            img_pil_or_tensor = data[0] if isinstance(data, (list, tuple)) else data
            if torch.equal(img_pil_or_tensor, dataset.error_tensor):
                 pbar.update(1); continue

            try:
                # Konvertiere zu PIL falls nötig (sollte direkt vom Dataset kommen)
                if isinstance(img_pil_or_tensor, torch.Tensor):
                    img_pil = TVF.to_pil_image(img_pil_or_tensor.squeeze(0))
                elif isinstance(img_pil_or_tensor, Image.Image):
                    img_pil = img_pil_or_tensor
                else:
                    raise TypeError(f"Unerwarteter Datentyp: {type(img_pil_or_tensor)}")

                # Wende Transformation ohne Normalisierung an
                augmented_tensor = basic_transform_no_norm(img_pil) # Ergibt Tensor [0, 1]

            except Exception as e:
                print(f"\nWarnung: Fehler Basis-Transform (Bild {pbar.n+1}, Loop {loop+1}): {type(e).__name__} - {e}. Fallback.")
                try:
                    if not isinstance(img_pil, Image.Image): # Sicherstellen
                        if isinstance(img_pil_or_tensor, torch.Tensor): img_pil = TVF.to_pil_image(img_pil_or_tensor.squeeze(0))
                        else: raise TypeError("Fallback nicht möglich ohne PIL")
                    augmented_tensor = fallback_transform_no_norm(img_pil) # Ergibt Tensor [0, 1]
                except Exception as fe:
                    print(f"FEHLER: Fallback fehlgeschlagen: {type(fe).__name__} - {fe}. Überspringe.")
                    pbar.update(1); continue

            batch_accumulator.append(augmented_tensor.to(device)) # Auf Zielgerät
            pbar.update(1)
            generated_count += 1

            if len(batch_accumulator) >= batch_size:
                yield torch.stack(batch_accumulator) # Yield Batch [0, 1]
                del batch_accumulator; batch_accumulator = []

            if generated_count >= num_total_augmentations: break

        pbar.close()
        del dataloader, dataset
        clean_memory()

        if generated_count >= num_total_augmentations: break

    if batch_accumulator:
        yield torch.stack(batch_accumulator) # Yield Rest-Batch [0, 1]
        del batch_accumulator
        clean_memory()

    print(f"Basis Aug: Generator beendet. {generated_count} Bilder generiert.")


def pizza_effects_augmentation_generator(image_paths, num_total_augmentations, batch_size=16, target_size=(256, 256),
                                         use_burning=True, burn_effect_params={},
                                         use_oven=True, oven_effect_params={},
                                         burn_prob=0.7, oven_prob=0.5,
                                         num_workers=4, pin_memory=True, prefetch_factor=2,
                                         transparent_background_color=(128, 128, 128)):
    """
    Generator für spezielle Pizza-Effekte (Verbrennung, Ofen).
    Lädt Bilder, wendet optionale Pre-Effekt-Transformationen an,
    wendet dann die GPU-beschleunigten Effekte an.
    Yielded Batches von Tensoren im Bereich [0, 1] (unnormalisiert).
    """
    # (Code angepasst: Entfernt Normalisierung, übergibt Effekt-Parameter)
    if not image_paths: print("Warnung (EffectsGen): Keine Bildpfade."); return
    if num_total_augmentations <= 0: print("Warnung (EffectsGen): num_total_augmentations <= 0."); return
    if not use_burning and not use_oven: print("Warnung (EffectsGen): Keine Effekte aktiviert."); return

    num_originals = len(image_paths)
    loops_needed = max(1, (num_total_augmentations + num_originals - 1) // num_originals)
    effect_name = [name for name, use in [("Verbrennung", use_burning), ("Ofen", use_oven)] if use]
    print(f"{'/'.join(effect_name)} Aug: Ziel={num_total_augmentations}, Originale={num_originals}. Benötigt ca. {loops_needed} Durchläufe.")

    # Transformationen *vor* den Effekten (ohne Normalisierung)
    pre_effect_transform = transforms.Compose([
        transforms.RandomApply([transforms.RandomRotation(degrees=(-90, 90), interpolation=TVF.InterpolationMode.BILINEAR)], p=0.5),
        transforms.RandomApply([transforms.RandomResizedCrop(target_size, scale=(0.7, 1.0), ratio=(0.9, 1.1), antialias=True)], p=0.6),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(target_size, antialias=True),
        transforms.ToTensor(), # Zu Tensor [0, 1]
    ])

    # Effekt-Module mit Parametern initialisieren
    burning_effect = PizzaBurningEffect(**burn_effect_params).to(device) if use_burning else None
    oven_effect = SimpleOvenEffect(**oven_effect_params).to(device) if use_oven else None

    generated_count = 0
    persistent_workers = (num_workers > 0) and (os.name != 'nt')

    for loop in range(loops_needed):
        if generated_count >= num_total_augmentations: break
        print(f"\n{'/'.join(effect_name)} Aug: Durchlauf {loop+1}/{loops_needed}")

        # KORREKTUR: Dataset erhält jetzt pre_effect_transform direkt
        dataset = PizzaAugmentationDataset(image_paths, transform=pre_effect_transform, target_size=target_size, transparent_background_color=transparent_background_color)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=pin_memory and (device.type == 'cuda'),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers, drop_last=True
        )

        pbar = tqdm(total=min(len(dataloader) * batch_size, num_total_augmentations - generated_count), desc=f"{'/'.join(effect_name)} Aug {loop+1}", unit="img")

        for data in dataloader:
            img_batch = data[0] if isinstance(data, (list, tuple)) else data # Sollte Tensor [0, 1] sein
            if img_batch.shape[0] == 0 or torch.equal(img_batch[0], dataset.error_tensor):
                 continue

            img_batch = img_batch.to(device, non_blocking=pin_memory and device.type=='cuda')
            current_batch_size = img_batch.shape[0]
            processed_batch = img_batch # Start mit pre-transformiertem Batch [0, 1]

            try:
                # Wende Effekte an (Ergebnis bleibt [0, 1])
                if burning_effect and random.random() < burn_prob:
                    processed_batch = burning_effect(processed_batch)
                if oven_effect and random.random() < oven_prob:
                     processed_batch = oven_effect(processed_batch)

            except Exception as e:
                 print(f"\nFEHLER Effektanwendung (Loop {loop+1}): {type(e).__name__} - {e}. Überspringe Batch.")
                 pbar.update(current_batch_size); clean_memory(); continue

            # KORREKTUR: Keine Normalisierung hier mehr. Yield direkt [0, 1] Tensor.
            yield processed_batch # Yield Batch [0, 1]

            pbar.update(current_batch_size)
            generated_count += current_batch_size

            del processed_batch # Speicher freigeben
            # clean_memory() # Nicht unbedingt nach jedem Batch

            if generated_count >= num_total_augmentations: break

        pbar.close()
        del dataloader, dataset
        clean_memory()

        if generated_count >= num_total_augmentations: break

    print(f"{'/'.join(effect_name)} Aug: Generator beendet. {generated_count} Bilder generiert.")


# =============================================================================
# BLOCK 5: pipeline.py - Haupt-Augmentierungs-Pipeline
# =============================================================================

def augment_pizza_dataset(input_dir, output_dir, total_augmentations, batch_size=32, target_size=(256, 256),
                          num_workers=4, strategy=None, save_format='jpg', save_quality=90,
                          transparent_background_color=(128, 128, 128),
                          burn_effect_params={}, oven_effect_params={},
                          progression_steps=4, mixup_alpha=0.4, cutmix_alpha=1.0):
    """
    Orchestriert den gesamten Augmentierungsprozess für Pizza-Bilder.
    Nutzt verschiedene Generatoren basierend auf der Strategie.
    Speichert die Ergebnisse ([0, 1] Tensoren werden zu Bildern) im Ausgabeordner.
    KORREKTUR: Behandelt burn und oven Effekte jetzt als separate Schritte.

    Args:
        input_dir (str): Pfad zu Originalbildern.
        output_dir (str): Ausgabeordner.
        total_augmentations (int): Zielanzahl generierter Bilder.
        batch_size (int): Batch-Größe für Verarbeitung/Speichern.
        target_size (tuple): Zielbildgröße (H, W).
        num_workers (int): Anzahl DataLoader Worker.
        strategy (dict, optional): Anteile der Augmentierungsarten.
                                   Beispiel: {'basic': 0.4, 'burn': 0.2, 'oven': 0.1, 'mix': 0.15, 'progression': 0.15}
                                   Wenn None, Standardstrategie verwenden.
        save_format (str): 'jpg' oder 'png'.
        save_quality (int): Qualität für JPG (1-100).
        transparent_background_color (tuple): RGB für transparenten Hintergrund.
        burn_effect_params (dict): Parameter für PizzaBurningEffect.
        oven_effect_params (dict): Parameter für SimpleOvenEffect.
        progression_steps (int): Anzahl der Stufen für Verbrennungsprogression.
        mixup_alpha (float): Alpha-Parameter für MixUp.
        cutmix_alpha (float): Alpha-Parameter für CutMix.
    """
    start_time_pipeline = time.time()
    if not os.path.isdir(input_dir): print(f"FEHLER: Input '{input_dir}' nicht gefunden."); return
    if total_augmentations <= 0: print("FEHLER: total_augmentations muss > 0 sein."); return
    os.makedirs(output_dir, exist_ok=True)

    try:
        all_files = os.listdir(input_dir)
        image_files = [os.path.join(input_dir, f) for f in all_files
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')) and not f.startswith('.')]
        image_files = [f for f in image_files if os.path.isfile(f)]
    except Exception as e: print(f"FEHLER beim Lesen von '{input_dir}': {e}"); return

    if not image_files: print(f"FEHLER: Keine Bilder in '{input_dir}' gefunden!"); return
    print(f"Gefunden: {len(image_files)} Originalbilder in '{input_dir}'.")
    random.shuffle(image_files)

    # --- Strategie ---
    if strategy is None:
        # KORREKTUR: Standardstrategie mit getrennten burn/oven
        strategy = {'basic': 0.40, 'burn': 0.20, 'oven': 0.10, 'mix': 0.15, 'progression': 0.15}
        print("Verwende Standard-Strategie (burn/oven getrennt).")
    else: print(f"Verwende benutzerdefinierte Strategie: {strategy}")

    total_ratio = sum(s for s in strategy.values() if isinstance(s, (int, float)) and s > 0) # Nur positive Zahlen berücksichtigen
    if not np.isclose(total_ratio, 1.0) and total_ratio > 0:
        print(f"Warnung: Strategie-Anteile summieren sich zu {total_ratio:.2f} (nicht 1.0). Normalisiere...")
        strategy = {k: v / total_ratio for k, v in strategy.items()}
    elif total_ratio == 0: print("FEHLER: Strategie-Summe ist 0."); return

    counts = {k: int(total_augmentations * v) for k, v in strategy.items()}
    remainder = total_augmentations - sum(counts.values())
    if remainder != 0 and counts:
        largest_cat = max(counts, key=counts.get)
        counts[largest_cat] += remainder
        print(f"Rundungsdifferenz ({remainder}) zu '{largest_cat}' addiert.")

    print("Geplante Augmentierungszahlen:")
    for cat in ['basic', 'burn', 'oven', 'mix', 'progression']: # Definierte Reihenfolge
        print(f"  - {cat}: {counts.get(cat, 0)}")
    print("-" * 30)

    # --- Setup ---
    all_augmented_saved_count = 0
    current_save_index = 0
    max_cpu_workers = os.cpu_count() or 1
    num_workers = min(num_workers, max_cpu_workers)
    pin_memory = (device.type == 'cuda')
    prefetch_factor = 2 if num_workers > 0 else None
    print(f"Verwende {num_workers} DataLoader Worker. Pin Memory: {pin_memory}")

    # --- Augmentierungsphasen ---

    # --- 1. Basis Augmentierung ---
    basic_count = counts.get('basic', 0)
    if basic_count > 0:
        print("\n--- Starte Basis-Augmentierung ---")
        gen_start_time = time.time(); saved_count_step = 0
        try:
            basic_gen = pizza_basic_augmentation_generator(
                image_paths=image_files, num_total_augmentations=basic_count, batch_size=batch_size,
                target_size=target_size, num_workers=num_workers, pin_memory=pin_memory,
                prefetch_factor=prefetch_factor, transparent_background_color=transparent_background_color
            )
            for basic_batch in basic_gen: # Batch ist [0, 1] auf device
                saved = save_augmented_images(basic_batch.cpu(), output_dir, "basic", batch_size=batch_size,
                                             start_idx=current_save_index, quality=save_quality, save_format=save_format)
                saved_count_step += saved; current_save_index += saved
                del basic_batch; clean_memory() # Speicher nach Speichern freigeben
            print(f"Basis-Aug abgeschlossen: {saved_count_step} Bilder gespeichert. Dauer: {time.time() - gen_start_time:.2f}s")
        except Exception as e: print(f"FEHLER Basis-Aug: {type(e).__name__} - {e}")
        finally: all_augmented_saved_count += saved_count_step; clean_memory()

    # --- 2. Verbrennungs-Effekte ---
    burn_count = counts.get('burn', 0)
    if burn_count > 0:
        print("\n--- Starte Verbrennungs-Effekt-Augmentierung ---")
        gen_start_time = time.time(); saved_count_step = 0
        try:
            burn_gen = pizza_effects_augmentation_generator(
                image_paths=image_files, num_total_augmentations=burn_count, batch_size=batch_size,
                target_size=target_size, use_burning=True, burn_effect_params=burn_effect_params,
                use_oven=False, # Nur Burn hier
                burn_prob=0.9, oven_prob=0.0, # Hohe Wahrsch. für Burn
                num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor,
                transparent_background_color=transparent_background_color
            )
            # Da burn_prob hoch ist, sollte die Zielzahl ungefähr erreicht werden
            for burn_batch in burn_gen: # Batch ist [0, 1] auf device
                saved = save_augmented_images(burn_batch.cpu(), output_dir, "burn", batch_size=batch_size,
                                             start_idx=current_save_index, quality=save_quality, save_format=save_format)
                saved_count_step += saved; current_save_index += saved
                del burn_batch; clean_memory()
                if saved_count_step >= burn_count: break # Breche ab, wenn Ziel erreicht
            print(f"Verbrennungs-Aug abgeschlossen: {saved_count_step} Bilder gespeichert. Dauer: {time.time() - gen_start_time:.2f}s")
        except Exception as e: print(f"FEHLER Verbrennungs-Aug: {type(e).__name__} - {e}")
        finally: all_augmented_saved_count += saved_count_step; clean_memory()

    # --- 3. Ofen-Effekte ---
    oven_count = counts.get('oven', 0)
    if oven_count > 0:
        print("\n--- Starte Ofen-Effekt-Augmentierung ---")
        gen_start_time = time.time(); saved_count_step = 0
        try:
            oven_gen = pizza_effects_augmentation_generator(
                image_paths=image_files, num_total_augmentations=oven_count, batch_size=batch_size,
                target_size=target_size, use_burning=False, # Nur Oven hier
                use_oven=True, oven_effect_params=oven_effect_params,
                burn_prob=0.0, oven_prob=0.9, # Hohe Wahrsch. für Oven
                num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor,
                transparent_background_color=transparent_background_color
            )
            for oven_batch in oven_gen: # Batch ist [0, 1] auf device
                saved = save_augmented_images(oven_batch.cpu(), output_dir, "oven", batch_size=batch_size,
                                             start_idx=current_save_index, quality=save_quality, save_format=save_format)
                saved_count_step += saved; current_save_index += saved
                del oven_batch; clean_memory()
                if saved_count_step >= oven_count: break
            print(f"Ofen-Aug abgeschlossen: {saved_count_step} Bilder gespeichert. Dauer: {time.time() - gen_start_time:.2f}s")
        except Exception as e: print(f"FEHLER Ofen-Aug: {type(e).__name__} - {e}")
        finally: all_augmented_saved_count += saved_count_step; clean_memory()

    # --- 4. Gemischte Bilder (MixUp/CutMix) ---
    mix_count = counts.get('mix', 0)
    if mix_count > 0 and len(image_files) >= 2:
        print("\n--- Starte MixUp/CutMix-Augmentierung ---")
        gen_start_time = time.time(); saved_count_step = 0; generated_mix_count = 0
        try:
            # Einfache Transformation nur für das Laden ([0, 1] Tensor)
            mix_transform = transforms.Compose([transforms.Resize(target_size, antialias=True), transforms.ToTensor()])
            dataset_mix = PizzaAugmentationDataset(image_files, transform=mix_transform, target_size=target_size, transparent_background_color=transparent_background_color)

            # Hinweis: Verwendung von zwei Loadern kann bei kleinen Datasets suboptimal sein.
            loader1 = DataLoader(dataset_mix, batch_size=batch_size, shuffle=True, num_workers=num_workers//2 or 1, pin_memory=pin_memory, drop_last=True, persistent_workers=(num_workers//2 or 1)>0 and os.name!='nt')
            loader2 = DataLoader(dataset_mix, batch_size=batch_size, shuffle=True, num_workers=num_workers//2 or 1, pin_memory=pin_memory, drop_last=True, persistent_workers=(num_workers//2 or 1)>0 and os.name!='nt')

            mix_pbar = tqdm(total=mix_count, desc="Mix Aug", unit="img")
            loaders_iter = zip(iter(loader1), iter(loader2))

            while generated_mix_count < mix_count:
                 try: batch1_data, batch2_data = next(loaders_iter)
                 except StopIteration: print("Mix Dataloader neu starten..."); loaders_iter = zip(iter(loader1), iter(loader2)); batch1_data, batch2_data = next(loaders_iter)
                 except Exception as load_err: print(f"Fehler beim Laden für Mix: {load_err}"); break # Abbruch bei Ladefehler

                 batch1 = (batch1_data[0] if isinstance(batch1_data, (list, tuple)) else batch1_data).to(device)
                 batch2 = (batch2_data[0] if isinstance(batch2_data, (list, tuple)) else batch2_data).to(device)
                 if batch1.shape[0] == 0 or batch2.shape[0] == 0: continue

                 # MixUp oder CutMix anwenden (Ergebnis ist [0, 1])
                 with torch.no_grad(): # Sicherstellen, dass hier keine Gradienten berechnet werden
                     if random.random() < 0.5:
                         mixed_batch, _ = pizza_mixup(batch1, batch2, alpha=mixup_alpha)
                     else:
                         mixed_batch, _ = pizza_cutmix(batch1, batch2, alpha=cutmix_alpha)

                 saved = save_augmented_images(mixed_batch.cpu(), output_dir, "mixed", batch_size=batch_size,
                                              start_idx=current_save_index, quality=save_quality, save_format=save_format)
                 saved_count_step += saved; current_save_index += saved
                 generated_mix_count += mixed_batch.shape[0]
                 mix_pbar.update(mixed_batch.shape[0])

                 del batch1, batch2, mixed_batch # Speicher
                 # clean_memory() # Nicht unbedingt hier

            mix_pbar.close()
            del loader1, loader2, dataset_mix # Cleanup
            print(f"Mix/CutMix-Aug abgeschlossen: {saved_count_step} Bilder gespeichert. Dauer: {time.time() - gen_start_time:.2f}s")
        except Exception as e: print(f"FEHLER Mix/CutMix-Aug: {type(e).__name__} - {e}")
        finally: all_augmented_saved_count += saved_count_step; clean_memory()

    # --- 5. Verbrennungsprogression ---
    progression_count = counts.get('progression', 0)
    if progression_count > 0:
        print("\n--- Starte Verbrennungs-Progression ---")
        gen_start_time = time.time(); saved_count_step = 0; generated_prog_steps_count = 0
        try:
            images_per_original = progression_steps + 1
            num_originals_needed = (progression_count + progression_steps - 1) // progression_steps
            num_originals_to_process = min(num_originals_needed, len(image_files))

            if num_originals_to_process > 0:
                print(f"Wähle {num_originals_to_process} Originalbilder für Progression ({progression_steps} Stufen).")
                selected_files = random.sample(image_files, k=num_originals_to_process)

                prog_transform = transforms.Compose([transforms.Resize(target_size, antialias=True), transforms.ToTensor()]) # Ergibt [0, 1]
                prog_dataset = PizzaAugmentationDataset(selected_files, transform=prog_transform, target_size=target_size, transparent_background_color=transparent_background_color)
                prog_batch_size = max(1, batch_size // images_per_original)
                prog_loader = DataLoader(prog_dataset, batch_size=prog_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
                prog_pbar = tqdm(total=progression_count, desc="Progression Aug", unit="step")

                for batch_data in prog_loader:
                    original_batch = (batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data).to(device)
                    if original_batch.shape[0] == 0: continue

                    # Generiere Progression (Ergebnis ist [0, 1])
                    # KORREKTUR: Übergebe burn_color_mult
                    with torch.no_grad():
                         progression_batch = pizza_burning_progression(original_batch, num_steps=progression_steps, burn_color_mult=burn_effect_params.get('burn_color_mult', (0.6, 0.3, 0.1)))

                    if progression_batch is None or progression_batch.shape[0] <= original_batch.shape[0]:
                         print("Warnung: Progression hat keine zusätzlichen Stufen erzeugt."); continue

                    saved = save_augmented_images(progression_batch.cpu(), output_dir, "progression", batch_size=batch_size,
                                                 start_idx=current_save_index, quality=save_quality, save_format=save_format)
                    saved_count_step += saved; current_save_index += saved
                    new_steps_in_batch = progression_batch.shape[0] - original_batch.shape[0]
                    generated_prog_steps_count += new_steps_in_batch
                    prog_pbar.update(new_steps_in_batch)

                    del original_batch, progression_batch; clean_memory()
                    if generated_prog_steps_count >= progression_count: break

                prog_pbar.close()
                del prog_loader, prog_dataset
            else: print("Nicht genug Originalbilder für Progression verfügbar oder Zielanzahl ist 0.")
            print(f"Verbrennungs-Progression abgeschlossen: {saved_count_step} Bilder gespeichert ({generated_prog_steps_count} neue Stufen). Dauer: {time.time() - gen_start_time:.2f}s")
        except Exception as e: print(f"FEHLER Progression-Aug: {type(e).__name__} - {e}")
        finally: all_augmented_saved_count += saved_count_step; clean_memory()


    # --- Abschluss ---
    print("\n" + "="*50)
    print("Augmentierungs-Pipeline abgeschlossen!")
    print(f"Insgesamt {all_augmented_saved_count} augmentierte Bilder gespeichert.")
    print(f"Ziel war ca.: {total_augmentations} Bilder.")
    print(f"Ausgabe im Verzeichnis: '{os.path.abspath(output_dir)}'")
    pipeline_duration = time.time() - start_time_pipeline
    print(f"Gesamtdauer der Pipeline: {pipeline_duration:.2f} Sekunden ({pipeline_duration/60:.2f} Minuten)")
    print("WICHTIG: Überprüfen Sie die generierten Bilder visuell!")
    print("="*50)

# =============================================================================
# BLOCK 6: main.py - Hauptausführungsskript
# =============================================================================

def main():
    print("="*60)
    print("      Pizza-Verbrennungserkennung - Datensatz-Augmentierung (Korrigierte Version)")
    print("="*60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Verwendetes Gerät: {device} ({'CUDA verfügbar' if torch.cuda.is_available() else 'Kein CUDA'})")
    if str(device) == 'cuda':
        try:
            print(f"CUDA Version (PyTorch): {torch.version.cuda}")
            props = torch.cuda.get_device_properties(0)
            print(f"GPU Speicher: {props.total_memory / (1024**3):.2f} GB")
            print(f"Compute Capability: {props.major}.{props.minor}")
        except Exception as e: print(f"Konnte CUDA-Details nicht abrufen: {e}")

    # --- Konfiguration ---
    INPUT_IMAGE_DIRECTORY = "/home/emilio/Documents/ai/pizza/data/raw"
    OUTPUT_DIRECTORY = "augmented_pizza_output_repaired"

    TOTAL_AUGMENTATIONS_TARGET = 1000   # Zielanzahl
    PROCESSING_BATCH_SIZE = 32          # Batch-Größe GPU/Speichern
    TARGET_IMAGE_SIZE = (256, 256)      # Zielbildgröße (H, W)
    DATALOADER_NUM_WORKERS = 4          # CPU-Kerne für Datenladen

    # Augmentierungsstrategie (Anteile ~1.0) - Burn/Oven getrennt
    AUGMENTATION_STRATEGY = {
        'basic': 0.40,
        'burn': 0.20,  # Nur Verbrennung
        'oven': 0.10,  # Nur Ofeneffekte
        'mix': 0.15,   # MixUp / CutMix
        'progression': 0.15 # Verbrennungsstufen
    }

    # Speicheroptionen
    SAVE_FORMAT = 'jpg' # 'jpg' oder 'png'
    JPG_SAVE_QUALITY = 90 # 1-100 für JPG

    # Farbanpassungen
    TRANSPARENT_BACKGROUND_COLOR = (128, 128, 128) # Grau als Default für Transparenz
    BURN_EFFECT_COLOR_MULT = (0.6, 0.3, 0.1) # RGB Multiplikatoren für Brandfarbe

    # Effekt-Parameter (können hier überschrieben werden, sonst Defaults)
    BURN_EFFECT_PARAMS = {
        'burn_intensity_min': 0.2, 'burn_intensity_max': 0.8,
        'edge_focus': 2.5, # Leicht erhöht zum Testen
        'spot_radius_min': 0.05, 'spot_radius_max': 0.2,
        'spot_count_min': 3, 'spot_count_max': 8,
        'burn_color_mult': BURN_EFFECT_COLOR_MULT # Übergebe Farbparameter
    }
    OVEN_EFFECT_PARAMS = {
        'steam_prob': 0.3, 'warmth_prob': 0.3, 'shadow_prob': 0.2,
        'steam_blur_kernel_min': 9, 'steam_blur_kernel_max': 25,
        'steam_blur_sigma_min': 5, 'steam_blur_sigma_max': 15
    }
    PROGRESSION_STEPS = 4 # Anzahl neuer Stufen (total 5 Bilder pro Original)
    MIXUP_ALPHA = 0.4
    CUTMIX_ALPHA = 1.0
    # --- Ende Konfiguration ---

    if not os.path.isdir(INPUT_IMAGE_DIRECTORY):
        print(f"\n!!! FEHLER: Input '{INPUT_IMAGE_DIRECTORY}' nicht gefunden. Bitte Pfad anpassen. !!!\n")
        return

    print("\n--- Einstellungen ---")
    print(f"Input:                  {INPUT_IMAGE_DIRECTORY}")
    print(f"Output:                 {OUTPUT_DIRECTORY}")
    print(f"Ziel-Augmentierungen:   {TOTAL_AUGMENTATIONS_TARGET}")
    print(f"Batch-Größe:            {PROCESSING_BATCH_SIZE}")
    print(f"Bildgröße:              {TARGET_IMAGE_SIZE}")
    print(f"DataLoader Workers:     {DATALOADER_NUM_WORKERS}")
    print(f"Strategie:              {AUGMENTATION_STRATEGY}")
    print(f"Speicherformat:         {SAVE_FORMAT.upper()} (Qualität: {JPG_SAVE_QUALITY if SAVE_FORMAT=='jpg' else 'N/A'})")
    print(f"Transparenz-BG:         {TRANSPARENT_BACKGROUND_COLOR}")
    print(f"Brandfarb-Multiplik.:   {BURN_EFFECT_PARAMS['burn_color_mult']}")
    print("-" * 30)

    try:
        augment_pizza_dataset(
            input_dir=INPUT_IMAGE_DIRECTORY,
            output_dir=OUTPUT_DIRECTORY,
            total_augmentations=TOTAL_AUGMENTATIONS_TARGET,
            batch_size=PROCESSING_BATCH_SIZE,
            target_size=TARGET_IMAGE_SIZE,
            num_workers=DATALOADER_NUM_WORKERS,
            strategy=AUGMENTATION_STRATEGY,
            save_format=SAVE_FORMAT,
            save_quality=JPG_SAVE_QUALITY,
            transparent_background_color=TRANSPARENT_BACKGROUND_COLOR,
            burn_effect_params=BURN_EFFECT_PARAMS,
            oven_effect_params=OVEN_EFFECT_PARAMS,
            progression_steps=PROGRESSION_STEPS,
            mixup_alpha=MIXUP_ALPHA,
            cutmix_alpha=CUTMIX_ALPHA
        )
    except KeyboardInterrupt:
        print("\n!!! Augmentierung durch Benutzer abgebrochen (Strg+C) !!!")
    except Exception as e:
        print("\n!!! Ein unerwarteter Fehler ist in der Haupt-Pipeline aufgetreten !!!")
        import traceback
        traceback.print_exc()
    finally:
        print("\nFühre finale Speicherbereinigung durch...")
        clean_memory()
        print("Skript beendet.")
        print("="*60)

if __name__ == "__main__":
    main()