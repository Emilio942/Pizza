# RP2040 Pizza-Erkennungssystem

Ein minimalistisches Bilderkennungssystem für den RP2040 Mikrocontroller zur Erkennung von Pizza-Zuständen.

## Projektstruktur

```
.
├── config/           # Konfigurationsdateien
├── data/            # Datensätze
│   ├── augmented/   # Augmentierte Bilder
│   └── raw/         # Original Bilder
├── docs/            # Dokumentation
├── models/          # Trainierte Modelle
│   ├── checkpoints/ # Trainings-Checkpoints
│   ├── exports/     # Exportierte Modelle
│   └── rp2040_export/ # RP2040-spezifische Modelle
├── output/          # Ausgabeverzeichnisse
│   ├── evaluation/  # Evaluierungsergebnisse
│   ├── logs/       # Logdateien
│   └── temp/       # Temporäre Dateien
├── src/            # Quellcode
└── tests/          # Testdateien
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
pip install -e .
```

## Verwendung

1. Datenaufbereitung und Augmentierung:
```bash
python src/augmentation.py --input-dir data/raw --output-dir data/augmented
```

2. Modell trainieren:
```bash
python src/pizza_detector.py train
```

3. Modell für RP2040 exportieren:
```bash
python src/pizza_detector.py export
```

## Hardware-Anforderungen

- RP2040 Mikrocontroller (z.B. Raspberry Pi Pico)
- OV2640 Kamerasensor
- CR123A Batterie mit LDO-Regler

Weitere Details finden Sie in der [ausführlichen Dokumentation](docs/RP2040%20Pizza-Erkennungssystem%20Dokumentation.pdf).