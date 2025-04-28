# Pizza-Erkennungsmodell für RP2040

Trainiert und optimiert: 2025-04-04 16:09

## Modelldetails

- **Klassen**: basic, burnt, combined, mixed, progression, segment
- **Bildgröße**: 48x48
- **Parameter**: 582
- **Modellgröße**: 9.24 KB (quantisiert)
- **Genauigkeit**: 0.00%
- **F1-Score**: 0.0000

## Verzeichnisstruktur

- `pizza_model_int8.pth`: Quantisiertes PyTorch-Modell
- `rp2040_export/`: C-Code und Dokumentation für RP2040-Integration
- `visualizations/`: Trainings- und Leistungsvisualisierungen
- `evaluation_report.json`: Detaillierter Evaluierungsbericht

## Nutzung

Siehe `rp2040_export/README.md` für Anweisungen zur Integration in RP2040-Projekte.

# Modelle für das Pizza-Erkennungssystem

Dieses Verzeichnis enthält die trainierten Modelle für das Pizza-Erkennungssystem.

## Modellversionen

### micro_pizza_model.pth
- Optimiertes Modell für RP2040
- Eingabegröße: 48x48x3
- Quantisiert: Int8
- Flash-Größe: ~180KB
- RAM-Bedarf: ~100KB

### pizza_model_float32.pth
- Nicht-quantisiertes Basismodell
- Eingabegröße: 48x48x3
- Datentyp: Float32

### pizza_model_int8.pth  
- Quantisiertes Modell
- Eingabegröße: 48x48x3
- Datentyp: Int8

## Verzeichnisstruktur

- `checkpoints/` - Trainings-Checkpoints für verschiedene Epochen
- `exports/` - Exportierte Modelle in verschiedenen Formaten (ONNX, TFLite)
- `rp2040_export/` - C-Code und Dokumentation für RP2040-Implementierung
- `visualizations/` - Trainings- und Evaluierungsvisualisierungen

## Evaluierungsmetriken

Siehe `evaluation_report.json` für detaillierte Leistungsmetriken für jedes Modell, einschließlich:
- Genauigkeit pro Klasse
- Konfusionsmatrix
- F1-Scores
- Inferenzzeiten
- RAM/Flash-Nutzung
