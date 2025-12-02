import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging
import numpy as np
from tqdm import tqdm
from collections import Counter
import random
from src.config import RP2040Config

logger = logging.getLogger(__name__)

class BasePizzaDataset(Dataset):
    """Base dataset without transforms for splitting train/val"""
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        # Find all classes (directories in the main directory)
        self.classes = [d for d in os.listdir(root_dir) 
                      if os.path.isdir(os.path.join(root_dir, d))
                      and not d.startswith('.')]
        self.classes.sort()  # For consistent indexing
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Collect all images and labels
        self.samples = self._collect_samples()
        
        # Compute class distribution for weighting
        self._compute_class_weights()
    
    def _collect_samples(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    samples.append((img_path, self.class_to_idx[class_name]))
        
        # Check if there are images
        if not samples:
            # Instead of raising error immediately, we might want to handle empty datasets gracefully
            # But for now, let's keep the behavior
            # raise RuntimeError(f"No images found in directory {self.root_dir}")
            pass
        
        return samples
    
    def _compute_class_weights(self):
        """Compute class weights for balanced training"""
        if not self.samples:
            self.class_weights = {}
            self.sample_weights = []
            return

        # Count images per class
        class_counts = Counter([label for _, label in self.samples])
        
        # Total number of samples and classes
        num_samples = len(self.samples)
        num_classes = len(self.classes)
        
        # Calculate weights: (N / (K * n_c)) where N=total, K=number of classes, n_c=count in class c
        self.class_weights = {c: num_samples / (num_classes * count) for c, count in class_counts.items()}
        
        # Create weights for each sample
        self.sample_weights = [self.class_weights[label] for _, label in self.samples]
    
    def __len__(self):
        return len(self.samples)

class TransformedPizzaDataset(Dataset):
    """Dataset that applies transforms to the base dataset"""
    def __init__(self, base_dataset, transform=None, indices=None, img_size=None, config=None):
        self.base_dataset = base_dataset
        self.transform = transform
        self.indices = indices if indices is not None else range(len(base_dataset))
        
        if img_size is not None:
            self.img_size = img_size
        elif config is not None:
            self.img_size = config.IMG_SIZE
        else:
            self.img_size = RP2040Config.IMG_SIZE
        
        # For balancing
        if hasattr(base_dataset, 'sample_weights') and base_dataset.sample_weights:
            self.sample_weights = [base_dataset.sample_weights[i] for i in self.indices]
        else:
            self.sample_weights = []
        
        # For class information
        self.classes = base_dataset.classes
        self.class_to_idx = base_dataset.class_to_idx
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the original image path and label
        img_path, label = self.base_dataset.samples[self.indices[idx]]
        
        try:
            img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
                
            return img, label
        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
            # Generate a black image as a last resort
            if self.transform:
                # Create a black PIL image and transform it
                black_img = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
                return self.transform(black_img), label
            else:
                # Create a black tensor directly
                return torch.zeros(3, self.img_size, self.img_size), label

class PizzaDatasetAnalysis:
    """Analysiert den Datensatz für optimale Vorverarbeitung und Klassenbalancierung"""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.class_dirs = [d for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d)) 
                          and not d.startswith('.')]
        
        self.stats = {
            'class_counts': {},
            'image_sizes': [],
            'mean_rgb': np.zeros(3),
            'std_rgb': np.zeros(3),
            'aspect_ratios': [],
            'total_images': 0
        }
        
    def analyze(self, sample_size=None):
        """Führt eine vollständige Analyse des Datensatzes durch"""
        logger.info(f"Analysiere Datensatz in {self.data_dir}...")
        
        # Sammle alle Bilder
        all_images = []
        for class_dir in self.class_dirs:
            class_path = os.path.join(self.data_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            image_files = [os.path.join(class_path, f) for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            self.stats['class_counts'][class_dir] = len(image_files)
            all_images.extend(image_files)
        
        self.stats['total_images'] = len(all_images)
        
        # Stichprobe für detaillierte Analyse
        if sample_size is not None and sample_size < len(all_images):
            sampled_images = random.sample(all_images, sample_size)
        else:
            sampled_images = all_images
        
        # Sammle RGB-Werte für Mittelwert- und Std-Berechnung
        rgb_values = []
        
        # Analysiere jedes Bild
        for img_path in tqdm(sampled_images, desc="Analysiere Bilder"):
            try:
                with Image.open(img_path) as img:
                    # Größe und Seitenverhältnis
                    width, height = img.size
                    self.stats['image_sizes'].append((width, height))
                    self.stats['aspect_ratios'].append(width / height)
                    
                    # Konvertiere zu RGB für Analyse
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Downsample für schnellere Verarbeitung
                    img_small = img.resize((50, 50))
                    img_array = np.array(img_small) / 255.0  # Normalisieren auf [0,1]
                    rgb_values.append(img_array.reshape(-1, 3))
            except Exception as e:
                logger.warning(f"Fehler beim Analysieren von {img_path}: {e}")
        
        # Berechne RGB-Mittelwerte und Standardabweichungen
        if rgb_values:
            all_rgb = np.vstack(rgb_values)
            self.stats['mean_rgb'] = np.mean(all_rgb, axis=0)
            self.stats['std_rgb'] = np.std(all_rgb, axis=0)
        else:
            # Fallback-Werte, wenn keine gültigen RGB-Werte gefunden wurden
            logger.warning("Keine gültigen Bilder für RGB-Analyse gefunden. Verwende Standardwerte.")
            self.stats['mean_rgb'] = np.array([0.5, 0.5, 0.5])  # Neutrale Werte
            self.stats['std_rgb'] = np.array([0.25, 0.25, 0.25])
        
        # Berechne durchschnittliche Bildgröße
        if self.stats['image_sizes']:
            widths, heights = zip(*self.stats['image_sizes'])
            self.stats['avg_width'] = sum(widths) / len(widths)
            self.stats['avg_height'] = sum(heights) / len(heights)
            self.stats['median_width'] = sorted(widths)[len(widths)//2]
            self.stats['median_height'] = sorted(heights)[len(heights)//2]
        else:
            # Fallback-Werte, wenn keine gültigen Bilder gefunden wurden
            logger.warning("Keine gültigen Bilder für die Größenanalyse gefunden. Verwende Standardwerte.")
            self.stats['avg_width'] = 320
            self.stats['avg_height'] = 240
            self.stats['median_width'] = 320
            self.stats['median_height'] = 240
        
        # Klassenverteilung und -gewichtung
        if self.stats['class_counts']:
            total = sum(self.stats['class_counts'].values())
            self.stats['class_distribution'] = {cls: count/total for cls, count in self.stats['class_counts'].items()}
            
            # Klassengewichte für Balancierung (inverses Verhältnis zur Häufigkeit)
            max_count = max(self.stats['class_counts'].values())
            self.stats['class_weights'] = {cls: max_count/count if count > 0 else 0.0 
                                for cls, count in self.stats['class_counts'].items()}
        else:
            # Fallback-Werte, wenn keine Klassen gefunden wurden
            logger.warning("Keine Klassen gefunden. Verwende Standardwerte.")
            self.stats['class_distribution'] = {'unknown': 1.0}
            self.stats['class_weights'] = {'unknown': 1.0}
        
        # Ausgabe der Ergebnisse
        logger.info("Datensatzanalyse abgeschlossen:")
        logger.info(f"Gesamtzahl der Bilder: {self.stats['total_images']}")
        logger.info(f"Klassenverteilung: {self.stats['class_counts']}")
        logger.info(f"Durchschnittliche Bildgröße: {self.stats['avg_width']:.1f} x {self.stats['avg_height']:.1f}")
        logger.info(f"RGB-Mittelwerte: [{self.stats['mean_rgb'][0]:.4f}, {self.stats['mean_rgb'][1]:.4f}, {self.stats['mean_rgb'][2]:.4f}]")
        logger.info(f"RGB-Standardabweichungen: [{self.stats['std_rgb'][0]:.4f}, {self.stats['std_rgb'][1]:.4f}, {self.stats['std_rgb'][2]:.4f}]")
        
        return self.stats
        
    def get_preprocessing_parameters(self):
        # Sicherstellen, dass die Analyse durchgeführt wurde, wenn nicht wurde sie bereits aufgerufen
        if not 'class_weights' in self.stats:
            self.analyze()
                
        # Round values for better readability
        mean_rgb = [round(float(x), 3) for x in self.stats['mean_rgb']]
        std_rgb = [round(float(x), 3) for x in self.stats['std_rgb']]
        
        # Ensure minimum std values
        std_rgb = [max(x, 0.1) for x in std_rgb]
        
        return {
            'mean': mean_rgb,  # Ensure this key exists
            'std': std_rgb,    # Ensure this key exists
            'mean_rgb': mean_rgb,  # Keep for backward compatibility
            'std_rgb': std_rgb,    # Keep for backward compatibility
            'class_weights': self.stats['class_weights']
        }

class BalancedPizzaDataset(Dataset):
    """Erweiterter Dataset mit Augmentierung und Klassenbalancierung für Pizza-Erkennung"""
    def __init__(self, root_dir, transform=None, split='train', config=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.config = config or RP2040Config()
        
        # Finde alle Klassen (Verzeichnisse im Hauptverzeichnis)
        self.classes = [d for d in os.listdir(root_dir) 
                         if os.path.isdir(os.path.join(root_dir, d))
                         and not d.startswith('.')]
        self.classes.sort()  # Für konsistente Indizierung
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Sammle alle Bilder und Labels
        self.samples = self._collect_samples()
        
        # Berechne Klassenverteilung für Gewichtung
        self._compute_class_weights()
    
    def _collect_samples(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    samples.append((img_path, self.class_to_idx[class_name]))
        
        # Überprüfe, ob es Bilder gibt
        if not samples:
            raise RuntimeError(f"Keine Bilder im Verzeichnis {self.root_dir} gefunden")
        
        return samples
    
    def _compute_class_weights(self):
        """Berechnet Klassengewichtungen für balanciertes Training"""
        # Zähle Bilder pro Klasse
        class_counts = Counter([label for _, label in self.samples])
        
        # Gesamtzahl der Samples und Klassen
        num_samples = len(self.samples)
        num_classes = len(self.classes)
        
        # Berechne Gewichte: (N / (K * n_c)) wobei N=Gesamtanzahl, K=Anzahl Klassen, n_c=Anzahl in Klasse c
        self.class_weights = {c: num_samples / (num_classes * count) for c, count in class_counts.items()}
        
        # Erstelle Gewichte für jedes Sample
        self.sample_weights = [self.class_weights[label] for _, label in self.samples]
        
        # Ausgabe der Klassenverteilung und Gewichte
        if self.split == 'train':
            logger.info(f"Klassenverteilung ({self.split}):")
            for cls_idx, count in class_counts.items():
                cls_name = self.classes[cls_idx]
                logger.info(f"  {cls_name}: {count} Bilder, Gewicht={self.class_weights[cls_idx]:.2f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a dataset item with robust error handling.
        Avoids infinite recursion and bias from failed images.
        """
        img_path, label = self.samples[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
                
            return img, label
        except Exception as e:
            # Log error but don't propagate
            logger.warning(f"Error loading {img_path}: {e}")
            
            # Track error statistics if this is the first access
            if not hasattr(self, 'error_stats'):
                self.error_stats = {
                    'total_errors': 0,
                    'class_errors': Counter(),
                    'error_paths': []
                }
            
            self.error_stats['total_errors'] += 1
            self.error_stats['class_errors'][label] += 1
            self.error_stats['error_paths'].append(img_path)
            
            # Log summary if error rate is high
            error_rate = self.error_stats['total_errors'] / len(self.samples)
            if error_rate > 0.05 and self.error_stats['total_errors'] % 10 == 0:
                logger.warning(f"High error rate detected: {error_rate:.1%} of images failed to load")
                logger.warning(f"Class distribution of errors: {dict(self.error_stats['class_errors'])}")
            
            # Create a placeholder tensor (black image) instead of recursion
            if self.transform:
                # Size depends on the transform, but most end with ToTensor
                channels = 3  # RGB
                h, w = self.config.IMG_SIZE, self.config.IMG_SIZE  # Default size
                return torch.zeros(channels, h, w), label
            else:
                # Return PIL image if no transform
                return Image.new('RGB', (self.config.IMG_SIZE, self.config.IMG_SIZE), (0, 0, 0)), label
