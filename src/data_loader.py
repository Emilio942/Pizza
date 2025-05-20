#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pizza-Datensatz-Lader

Dieses Modul enthält Klassen und Funktionen für das Laden und Vorverarbeiten des 
Pizza-Datensatzes für Training und Evaluierung.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from pathlib import Path

class PizzaDataset(Dataset):
    """
    Dataset-Klasse für den Pizza-Datensatz
    
    Unterstützt:
    - Laden von Bildern aus dem Dateisystem
    - Anwendung von Transformationen
    - Klassen-Mapping (Pizza-Zustände zu Klassen-Indizes)
    """
    def __init__(self, data_dir, transform=None, split='train'):
        """
        Initialisiert das PizzaDataset
        
        Args:
            data_dir (str): Verzeichnis mit den Bilddaten
            transform (callable): Transformationen für die Bilder
            split (str): 'train' oder 'val' für Trainings- oder Validierungsdaten
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        
        # Definiere Klassen-Mapping
        self.classes = ['basic', 'burnt', 'partially_cooked', 'ready']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Lade Dateipfade und Labels
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Lade alle Bilddateien mit ihren Labels"""
        samples = []
        
        # Suche in entsprechenden Unterverzeichnissen
        augmented_dirs = ['augmented_pizza/basic', 'augmented_pizza/burnt', 
                          'augmented_pizza/mixed', 'augmented_pizza/combined']
        
        for dir_path in augmented_dirs:
            class_dir = self.data_dir / dir_path
            if not class_dir.exists():
                continue
                
            # Bestimme, welche Klasse dieses Verzeichnis enthält
            class_name = class_dir.parts[-1]
            if class_name == 'mixed':
                class_name = 'partially_cooked'
            elif class_name == 'combined':
                class_name = 'ready'
                
            class_idx = self.class_to_idx.get(class_name, 0)
            
            # Sammle alle JPG-Dateien
            for img_path in class_dir.glob('*.jpg'):
                samples.append((str(img_path), class_idx))
        
        # Teile Datensatz für Training und Validierung auf
        random.seed(42)  # Für reproduzierbarkeit
        random.shuffle(samples)
        
        split_idx = int(len(samples) * 0.8)  # 80% Training, 20% Validierung
        
        if self.split == 'train':
            return samples[:split_idx]
        else:
            return samples[split_idx:]
    
    def __len__(self):
        """Gibt die Anzahl der Samples zurück"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Gibt ein Sample mit Index idx zurück"""
        img_path, label = self.samples[idx]
        
        # Lade und verarbeite das Bild
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

def create_dataloaders(data_dir='augmented_pizza', batch_size=32, img_size=48):
    """
    Erstellt DataLoader für Trainings- und Validierungsdaten
    
    Args:
        data_dir (str): Verzeichnis mit den Bilddaten
        batch_size (int): Batch-Größe für DataLoader
        img_size (int): Zielgröße für Bilder
        
    Returns:
        train_loader, val_loader: DataLoader für Training und Validierung
    """
    # Definiere Transformationen
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Erstelle Datasets
    train_dataset = PizzaDataset(
        data_dir=data_dir,
        transform=train_transform,
        split='train'
    )
    
    val_dataset = PizzaDataset(
        data_dir=data_dir,
        transform=val_transform,
        split='val'
    )
    
    # Erstelle DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader
