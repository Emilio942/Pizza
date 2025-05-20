# Pizza-Erkennungsmodell für RP2040

Trainiert und optimiert: 2025-05-14 12:54

## Modelldetails

- **Klassen**: basic, burnt, combined, mixed, processed, progression, raw, segment, synthetic
- **Bildgröße**: 48x48
- **Parameter**: 633
- **Modellgröße**: 9.43 KB (quantisiert)
- **Genauigkeit**: 50.00%
- **F1-Score**: 0.0741

## Verzeichnisstruktur

- `pizza_model_int8.pth`: Quantisiertes PyTorch-Modell
- `rp2040_export/`: C-Code und Dokumentation für RP2040-Integration
- `visualizations/`: Trainings- und Leistungsvisualisierungen
- `evaluation_report.json`: Detaillierter Evaluierungsbericht

## Nutzung

Siehe `rp2040_export/README.md` für Anweisungen zur Integration in RP2040-Projekte.
