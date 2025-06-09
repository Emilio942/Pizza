# 📁 Projekt-Ordnerstruktur

Diese Datei dokumentiert die organisierte Ordnerstruktur des Pizza-Projekts.

## 🗂️ Hauptverzeichnisse

### `/src/` - Quellcode
- **Hauptmodule**: Kernfunktionalität der Pizza-Erkennung
- **`/rl/`**: Reinforcement Learning Komponenten
- **`/emulation/`**: Hardware-Emulation
- **`/verification/`**: Verifikations-Tools

### `/scripts/` - Utilities und Tools
- **Datenverarbeitung**: Augmentation, Preprocessing
- **Training**: Model Training Scripts
- **Evaluation**: Bewertungs- und Benchmark-Tools
- **`image_quality_control.py`**: Bildqualitätskontrolle-Tool

### `/config/` - Konfigurationsdateien
- **`requirements.txt`**: Standard Python Dependencies
- **`requirements_spatial.txt`**: Spatial Model Dependencies
- **`spatial_requirements.txt`**: Weitere Spatial Requirements

### `/docs/` - Dokumentation
- **`/aufgaben/`**: Aufgabenstellungen und Spezifikationen
- **`/reports/`**: Completion Reports und Analysen
- **`/status/`**: Projekt-Status und Task-Tracking

### `/logs/` - Log-Dateien
- **Training Logs**: `pizza_training_detailed.log`
- **System Logs**: `api_server.log`, `automated_test_suite.log`
- **Analysis Logs**: `compression_analysis.log`, `clustering_output.log`

### `/data/` - Datensätze
- **Training Data**: Original und verarbeitete Datensätze
- **Test Data**: Evaluierungs-Datensätze

### `/models/` - Trainierte Modelle
- **Standard Models**: Basis-Pizza-Erkennungsmodelle
- **Optimized Models**: Pruned/Quantized Versionen

### `/tests/` - Test-Suite
- **Unit Tests**: Komponentenspezifische Tests
- **Integration Tests**: End-to-End Tests

### `/augmented_pizza/` - Augmentierte Datensätze
- **`/basic/`**: Basis-Pizza-Bilder
- **`/burnt/`**: Verbrannte Pizza-Bilder
- **`/mixed/`**: Gemischte Kategorien
- **`/synthetic/`**: Synthetisch generierte Bilder

## 🧹 Aufräumaktion (Juni 2025)

### Verschobene Dateien:
- ✅ **Log-Dateien** → `/logs/`
- ✅ **Aufgaben-Dateien** → `/docs/aufgaben/`
- ✅ **Requirements** → `/config/`
- ✅ **Reports** → `/docs/reports/`
- ✅ **Status-Dateien** → `/docs/status/`

### Symlinks erstellt:
- `PROJECT_STATUS.txt` → `/docs/status/PROJECT_STATUS.txt`
- `COMPLETED_TASKS.md` → `/docs/status/COMPLETED_TASKS.md`

## 📋 Vorteile der neuen Struktur:

1. **Klarere Trennung** zwischen Code, Daten und Dokumentation
2. **Bessere Auffindbarkeit** von spezifischen Dateitypen
3. **Sauberes Hauptverzeichnis** mit nur essentiellen Dateien
4. **Logische Gruppierung** verwandter Komponenten
5. **Einfachere Navigation** für neue Entwickler

## 🔗 Wichtige Einstiegspunkte:

- **Training starten**: `/scripts/train_pizza_model.py`
- **Bildqualität prüfen**: `/scripts/image_quality_control.py`
- **Projekt-Status**: `/docs/status/PROJECT_STATUS.txt`
- **API starten**: `/src/pizza-baking-detection-final.py`
