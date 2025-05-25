# RP2040 Pizza-Erkennungssystem

Ein minimalistisches Bilderkennungssystem für den RP2040 Mikrocontroller zur Erkennung von Pizza-Zuständen.

## Projektstruktur

```
.
├── config/           # Konfigurationsdateien
│   └── *.json        # JSON-Konfigurationsdateien
├── data/             # Datensätze
│   ├── augmented/    # Augmentierte Bilder
│   ├── classified/   # Klassifizierte Bilder
│   ├── processed/    # Verarbeitete Bilder
│   ├── raw/          # Original Bilder
│   ├── synthetic/    # Synthetische Bilder
│   └── videos/       # Video-Dateien
├── docs/             # Dokumentation
│   ├── completed_tasks/ # Dokumentation abgeschlossener Aufgaben
│   ├── status/       # Projektstatusdateien
│   └── *.md          # Allgemeine Dokumentation
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
├── scripts/          # Skripte für verschiedene Aufgaben
│   ├── evaluation/   # Evaluierungsskripte
│   ├── processing/   # Verarbeitungsskripte
│   └── utility/      # Hilfsskripte
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
# Standard augmentation pipeline
python scripts/standard_augmentation.py path/to/sample_image.jpg  # Visualize augmentations

# Full dataset augmentation
python scripts/augment_dataset.py --input-dir data/raw --output-dir data/augmented --aug-types all
```

2. Modell trainieren (mit standard Augmentierungspipeline):
```bash
# Train with standard augmentation pipeline (medium intensity)
python scripts/train_with_augmentation.py --data-dir data/classified --aug-intensity medium

# Train with high intensity augmentations
python scripts/train_with_augmentation.py --data-dir data/classified --aug-intensity high
```

3. Oder das Modell mit dem ursprünglichen Trainingsskript trainieren:
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

5. Temperaturmessungs-Test ausführen:
```bash
python -m tests.test_temperature_logging
```

Die Temperaturmessung simuliert die Erfassung und Aufzeichnung von Temperaturdaten im RP2040. Die Logs werden im CSV-Format und über UART gespeichert und können für die Analyse der Temperaturentwicklung unter verschiedenen Lastbedingungen verwendet werden. Visualisierungen finden Sie im [Bildverzeichnis](docs/images/). Weitere Details finden Sie in der [Temperaturmessung-Dokumentation](docs/temperature_monitoring.md).

## Hardware-Anforderungen

- RP2040 Mikrocontroller (z.B. Raspberry Pi Pico)
- OV2640 Kamerasensor
- CR123A Batterie mit LDO-Regler

Weitere Details finden Sie in der [ausführlichen Dokumentation](docs/RP2040%20Pizza-Erkennungssystem%20Dokumentation.pdf).

## Dokumentation

- [Standard Augmentierungspipeline](docs/standard_augmentation_pipeline.md) - Detaillierte Beschreibung der implementierten Augmentierungstechniken und Parameter
- [RP2040 Hardware-Integration](docs/hardware-integration.md) - Hardware-Integrationsanleitung
- [Temperaturmessung](docs/temperature_monitoring.md) - Informationen zur Temperaturerfassung und -analyse

## Projektstatus und Aufgaben

Der aktuelle Projektstatus wird in folgenden Dateien dokumentiert:
- [Projektstatus](docs/status/PROJECT_STATUS.txt) - Aktuelle Projektstatus mit Details zu abgeschlossenen und laufenden Aufgaben
- [Aufgabenliste](docs/status/aufgaben.txt) - Liste aller Aufgaben des Projekts
- [Abgeschlossene Aufgaben](docs/status/COMPLETED_TASKS.md) - Detaillierte Dokumentation aller abgeschlossenen Aufgaben

Symlinks zu diesen Dateien sind im Hauptverzeichnis des Projekts verfügbar.