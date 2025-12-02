# Plan zur Verbesserung der Softwarearchitektur

**Datum:** 2025-11-16
**Autor:** Gemini CLI Agent

## 1. Zusammenfassung

Dieses Dokument beschreibt einen Plan zur strukturellen Verbesserung (Refactoring) des Pizza-Erkennungsprojekts. Die Kernanalyse hat ergeben, dass die primäre Geschäftslogik – von der Datenanalyse über das Training bis zum Export – in einem einzigen, monolithischen Skript (`src/pizza-baking-detection-final.py`) konzentriert ist. Dieser Zustand erschwert die Wartbarkeit, Testbarkeit und Wiederverwendbarkeit der einzelnen Komponenten erheblich.

Der vorgeschlagene Lösungsansatz ist ein schrittweises Refactoring hin zu einer modularen, paketbasierten Architektur mit einer klaren Trennung der Verantwortlichkeiten ("Separation of Concerns").

## 2. Analyse des aktuellen Zustands

- **Monolithische Struktur:** Das Skript `src/pizza-baking-detection-final.py` ist über 2000 Zeilen lang und enthält diverse, logisch getrennte Funktionalitäten.
- **Vermischte Verantwortlichkeiten:** In der Datei sind folgende Aufgaben vermischt:
  - Hardware- und Projektkonfiguration (`RP2040Config`)
  - Datenanalyse (`PizzaDatasetAnalysis`)
  - Speicher- und Leistungsabschätzung (`MemoryEstimator`)
  - Dataset-Definition (`BalancedPizzaDataset`, `BasePizzaDataset`)
  - Erstellung der Datenlader (`create_optimized_dataloaders`)
  - Modelldefinition (`MicroPizzaNet`)
  - Trainingslogik (`train_microcontroller_model`)
  - Hilfsklassen (`EarlyStopping`)
  - Quantisierung und Export (`calibrate_and_quantize`, `export_to_microcontroller`)
- **Fehlender zentraler Einstiegspunkt:** Es gibt keine klare Kommandozeilenschnittstelle (CLI), um gezielt einzelne Aktionen wie "nur trainieren" oder "nur exportieren" auszuführen.

## 3. Vorgeschlagener Refactoring-Plan

### Phase 1: Modularisierung des `src`-Verzeichnisses

Der wichtigste Schritt ist die Aufteilung des monolithischen Skripts in eine logische Paketstruktur.

**Neue Verzeichnisstruktur:**

```
src/
├── __init__.py
├── augmentation/
│   ├── __init__.py
│   ├── cheese_augment.py
│   └── ...
├── data/
│   ├── __init__.py
│   ├── analysis.py       # Enthält PizzaDatasetAnalysis
│   ├── datasets.py       # Enthält BasePizzaDataset, TransformedPizzaDataset
│   └── loaders.py        # Enthält create_optimized_dataloaders
├── models/
│   ├── __init__.py
│   └── micropizzanet.py  # Enthält MicroPizzaNet
├── training/
│   ├── __init__.py
│   ├── trainer.py        # Enthält train_microcontroller_model
│   └── utils.py          # Enthält EarlyStopping
├── export/
│   ├── __init__.py
│   ├── quantization.py   # Enthält calibrate_and_quantize
│   └── rp2040.py         # Enthält export_to_microcontroller
└── utils/
    ├── __init__.py
    └── memory.py         # Enthält MemoryEstimator
```

**Umsetzungsplan:**

1.  Erstelle die oben gezeigten neuen Verzeichnisse und `__init__.py`-Dateien.
2.  Verschiebe die jeweilige Klasse oder Funktion aus `src/pizza-baking-detection-final.py` in die entsprechende neue Datei.
3.  Passe die Import-Anweisungen in den neuen Dateien an, damit sie die neue modulare Struktur widerspiegeln (z.B. `from src.models.micropizzanet import MicroPizzaNet`).
4.  Das Skript `src/pizza-baking-detection-final.py` wird am Ende dieser Phase zu einem Orchestrierungs-Skript, das die Funktionen aus den neuen Modulen importiert und in der richtigen Reihenfolge aufruft.

### Phase 2: Zentralisierung der Konfiguration

Die Konfiguration ist derzeit eine Klasse innerhalb des Hauptskripts.

- **Vorschlag:** Verschiebe die `RP2040Config`-Klasse in eine eigene Datei, z.B. `src/config.py`.
- **Zukünftige Erweiterung:** Für mehr Flexibilität könnte die Konfiguration in eine externe Datei (z.B. `config.yaml`) ausgelagert und mit einer Bibliothek wie `Pydantic` oder `Hydra` geladen werden. Dies würde Experimente mit Hyperparametern ermöglichen, ohne den Code zu ändern.

### Phase 3: Einführung einer Kommandozeilenschnittstelle (CLI)

Nach der Modularisierung kann ein benutzerfreundlicher Einstiegspunkt geschaffen werden.

- **Vorschlag:** Erstelle eine neue Datei `run.py` im Projekt-Stammverzeichnis. Unter Verwendung einer Bibliothek wie `argparse` oder `click` kann dieses Skript verschiedene Aktionen steuern.

- **Beispielhafte Befehle:**
  ```bash
  # Nur das Training ausführen
  python run.py train

  # Ein trainiertes Modell evaluieren
  python run.py evaluate --model-path "models/best_model.pth"

  # Ein Modell für den Export vorbereiten
  python run.py export --model-path "models/best_model.pth"

  # Die 3D-Käse-Augmentierungsbilder generieren
  python run.py generate-renders
  ```

- **Implementierung:** Das `run.py`-Skript würde die Kommandozeilenargumente parsen und die entsprechenden Funktionen aus den neuen Modulen (z.B. `src.training.trainer.train_microcontroller_model`) aufrufen.

## 4. Vorteile des Refactorings

- **Wartbarkeit:** Kleinere, fokussierte Module sind leichter zu verstehen und zu ändern.
- **Testbarkeit:** Jedes Modul kann isoliert getestet werden (Unit-Tests).
- **Wiederverwendbarkeit:** Funktionen wie die Datenanalyse oder die Modellerstellung können leicht in anderen Kontexten wiederverwendet werden.
- **Skalierbarkeit:** Das Hinzufügen neuer Funktionen (z.B. ein anderes Modell, eine neue Exportmethode) wird einfacher, da nur ein neues Modul hinzugefügt werden muss, anstatt den Monolithen zu verändern.

## 5. Fazit

Dieser Plan bietet eine klare und strukturierte Roadmap, um die Codebasis des Projekts auf ein professionelles Niveau zu heben. Die Umsetzung dieses Plans wird die Langlebigkeit und Erweiterbarkeit des Projekts erheblich verbessern und die zukünftige Entwicklung beschleunigen. Es wird empfohlen, dass eine zukünftige KI oder ein Entwickler diesen Plan als Leitfaden für die Umstrukturierung verwendet.
