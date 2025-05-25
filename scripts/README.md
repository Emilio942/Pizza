# Skripte für das Pizza-Erkennungssystem

Dieses Verzeichnis enthält verschiedene Skripte für das Pizza-Erkennungssystem, organisiert nach Funktionalität.

## Verzeichnisstruktur

- `evaluation/` - Skripte zur Evaluierung von Modellen und Ergebnissen
  - Evaluierungsskripte (evaluate_*.py)
  - Benchmarks und Testskripte
  - Verifikationsskripte

- `processing/` - Skripte zur Datenverarbeitung und -generierung
  - Datensatz-Balancierung (balance_*.py)
  - Bildgenerierung (create_*.py, generate_*.py)
  - Datensatzerweiterung (extend_*.py)

- `utility/` - Hilfsskripte und Tools
  - Analyse-Tools (analyze_*.py)
  - Visualisierungsskripte (visualize_*.py)
  - Ausführungsskripte (run_*.py, run_*.sh)

- `early_exit/` - Skripte speziell für Early-Exit-Modelle
  - Training und Evaluierung von Early-Exit-Netzwerken
  - Metriken und Benchmarks

- `model_optimization/` - Skripte zur Modelloptimierung
  - Pruning-Tools
  - Quantisierung
  - Modellkompression

- `backup/` - Backup-Skripte und Sicherungswerkzeuge

- `ci/` - Continuous Integration Skripte

## Verwendung

Die meisten Skripte können direkt ausgeführt werden und haben integrierte Hilfe:

```bash
python scripts/utility/visualize_pruning_results.py --help
```

Allgemeine Skripte für häufige Aufgaben sind im Hauptverzeichnis des Projekts in der README.md dokumentiert.
