"""
Gemeinsam genutzte Hilfsfunktionen für das Pizza-Erkennungssystem.
"""

import os
import logging
import logging.config
import torch
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional

def setup_logging(config_path: str = 'config/logging.conf') -> None:
    """Konfiguriert das Logging-System."""
    if os.path.exists(config_path):
        logging.config.fileConfig(config_path)
    else:
        logging.basicConfig(level=logging.INFO)
    
def get_device() -> torch.device:
    """Ermittelt das beste verfügbare Gerät (CUDA/CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ensure_dir(path: Union[str, Path]) -> Path:
    """Stellt sicher, dass ein Verzeichnis existiert."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_model(model_path: str, device: Optional[torch.device] = None) -> torch.nn.Module:
    """Lädt ein PyTorch-Modell mit Fehlerbehandlung."""
    if device is None:
        device = get_device()
    
    try:
        model = torch.load(model_path, map_location=device)
        model.eval()
        return model
    except Exception as e:
        logging.error(f"Fehler beim Laden des Modells {model_path}: {e}")
        raise

def save_model(model: torch.nn.Module, save_path: str) -> None:
    """Speichert ein PyTorch-Modell mit Fehlerbehandlung."""
    try:
        ensure_dir(os.path.dirname(save_path))
        torch.save(model.state_dict(), save_path)
        logging.info(f"Modell gespeichert unter: {save_path}")
    except Exception as e:
        logging.error(f"Fehler beim Speichern des Modells unter {save_path}: {e}")
        raise

def estimate_inference_time(model: torch.nn.Module, 
                          input_size: Tuple[int, ...],
                          num_runs: int = 100) -> float:
    """Schätzt die durchschnittliche Inferenzzeit eines Modells."""
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)
    
    # Zeitmessung
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            model(dummy_input)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    
    return np.mean(times)

def format_size(size_bytes: int) -> str:
    """Formatiert Bytes in lesbare Größe."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"