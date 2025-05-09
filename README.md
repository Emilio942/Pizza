# RP2040 Pizza-Erkennungssystem

Ein minimalistisches Bilderkennungssystem für den RP2040 Mikrocontroller zur Erkennung von Pizza-Zuständen.

## Projektstruktur

```
.
├── config/           # Konfigurationsdateien
├── data/             # Datensätze
│   ├── augmented/    # Augmentierte Bilder
│   ├── classified/   # Klassifizierte Bilder
│   ├── processed/    # Verarbeitete Bilder
│   ├── raw/          # Original Bilder
│   ├── synthetic/    # Synthetische Bilder
│   └── videos/       # Video-Dateien
├── docs/             # Dokumentation
├── hardware/         # Hardware-Dateien
│   ├── datasheets/   # Datenblätter
│   ├── docs/         # Hardware-Dokumentation
│   ├── eda/          # Elektronische Design-Dateien
│   └── manufacturing/# Fertigungsunterlagen
├── models/           # Trainierte Modelle
│   ├── checkpoints/  # Trainings-Checkpoints
│   ├── exports/      # Exportierte Modelle
│   ├── pruned_model/ # Gestutzte Modelle
│   ├── rp2040_export/# RP2040-spezifische Modelle
│   └── visualizations/# Modell-Visualisierungen
├── output/           # Ausgabeverzeichnisse
│   ├── logs/         # Logdateien
│   └── temp/         # Temporäre Dateien
├── src/              # Quellcode
│   ├── augmentation/ # Code für Datenaugmentierung
│   ├── chatlist_ki/  # KI-Chat-Schnittstelle
│   ├── emulation/    # RP2040-Emulation
│   ├── integration/  # Integration mit anderen Systemen
│   └── utils/        # Hilfsfunktionen
└── tests/            # Testdateien
```

## Installation

1. Python-Umgebung einrichten (Python 3.8+ erforderlich):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate     # Windows
```

2. Abhängigkeiten installieren:
```bash
pip install -r requirements.txt
```

## Verwendung

1. Datenaufbereitung und Augmentierung:
```bash
python src/augmentation/enhanced_pizza_augmentation.py --input-dir data/raw --output-dir data/augmented
```

2. Modell trainieren:
```bash
python src/pizza_detector.py train
```

3. Modell für RP2040 exportieren:
```bash
python src/pizza_detector.py export
```

4. Emulator starten:
```bash
python src/emulation/emulator.py
```

## Hardware-Anforderungen

- RP2040 Mikrocontroller (z.B. Raspberry Pi Pico)
- OV2640 Kamerasensor
- CR123A Batterie mit LDO-Regler

Weitere Details finden Sie in der [ausführlichen Dokumentation](docs/RP2040%20Pizza-Erkennungssystem%20Dokumentation.pdf).