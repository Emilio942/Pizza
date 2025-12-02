# Aufgabenlisten-Review – 2025-10-03

## Zusammenfassung
- Schwerpunkt auf Funktionsprüfungen der wichtigsten Skripte aus `aufgaben.txt`, inkl. kleiner End-to-End-Läufe.
- Mehrere Hilfs-Skripte laufen erfolgreich, aber einige Aufgaben verweisen auf veraltete Pfade (`scripts/analyze_*` → `scripts/utility/`).
- Vollständiger `pytest`-Durchlauf bricht mit 28 Fehlern und 1 Fehler in Diffusions-Pipeline-Einbindung ab; zentrale Speicher-/Validierungsaufgaben offenbar ungelöst.

## Geprüfte Skripte
| Skript | Aufruf | Ergebnis |
| --- | --- | --- |
| `scripts/classify_images.py` | `python scripts/classify_images.py --input test_verification_images --output temp_output/classify_smoke --no-display` | ✅ Klassifikation erzeugt 4 Ausgabebilder + JSON, keine Fehler. |
| `scripts/augment_dataset.py` | `python scripts/augment_dataset.py --input-dir test_verification_images --output-dir temp_output/augment_smoke --num-per-image 1 --aug-types basic --target-size 48 --batch-size 2 --save-stats` | ✅ Augmentierung läuft auf GPU, Statistikdatei erzeugt. |
| `scripts/utility/analyze_performance_logs.py` | `python scripts/utility/analyze_performance_logs.py --help` | ✅ Hilfeausgabe erscheint, Pfad korrekt. |
| `scripts/analyze_performance_logs.py` | `python scripts/analyze_performance_logs.py --help` | ❌ Datei fehlt (Pfad nicht mehr gültig). |
| `scripts/utility/analyze_energy_measurements.py` | `python scripts/utility/analyze_energy_measurements.py --help` | ✅ Hilfeausgabe erscheint. |
| `scripts/analyze_energy_measurements.py` | `python scripts/analyze_energy_measurements.py --help` | ❌ Datei fehlt (Pfad nicht mehr gültig). |
| `scripts/simulate_battery_life.py` | `python scripts/simulate_battery_life.py --list-scenarios` | ✅ Szenarien werden gelistet.
| `scripts/simulate_power_consumption.py` | `python scripts/simulate_power_consumption.py --help` | ⚠️ Skript führt sofort Simulationen aus, erzeugt PNG + JSON, keine Hilfe-Ausgabe.

## Pytest-Ergebnisse
- Befehl: ``python -m pytest -q`` (Dauer ~4 min, abgebrochen mit Exit-Code 130).
- 28 Tests fehlgeschlagen, 1 Fehler (Diffusions-Pipeline lädt Stable Diffusion XL, führt zu Netzwerk-/Downloadversuch), 7 übersprungen.
- Häufigste Fehlerbilder:
	- **RAM-Überläufe** beim Laden der Firmware im Emulator (`tests/test_power_manager.py`, `tests/test_framebuffer_ram.py`, `tests/test_emulator_limits.py`).
	- **Validierungsfunktionen** schlagen erwartete Exceptions nicht oder zu hart zu (`tests/test_validation.py`).
	- **Metrik-Berechnungen** liefern falsche Formen/Daten (`tests/test_metrics.py`).
	- **Pizza-Model Tests** scheitern wegen Modell-API-Änderungen (`load_model` kennt `quantized` nicht, Accuracy = 0 %).
	- **Diffusions-Pipeline** (`tests/test_verification_algorithms.py`) versucht umfangreiche Downloads und endet mit KeyboardInterrupt.
- Warnungen: zahlreiche `PytestReturnNotNoneWarning` → Tests geben Werte zurück statt Assertions.

## Beobachtungen & offene Punkte
- Aufgabenverweise in `aufgaben.txt` müssen auf den tatsächlichen Speicherort `scripts/utility/…` aktualisiert werden (Performance-/Energie-Analyse).
- `simulate_power_consumption.py` ignoriert `--help` und führt direkt Simulationen aus → Falsches CLI-Verhalten gemessen am Erwartungstext.
- `pytest`-Befunde zeigen, dass viele Speicher- und Validierungsaufgaben faktisch nicht erfüllt sind (RAM-Abschätzungen, Tensor-Arena-Checks, Modell-Genauigkeit).
- CI/Workflow-Aufgaben (`PERF-4.1`) weiterhin offen; keine Hinweise auf automatisierte Regressionstests.

## Nächste Schritte (Empfehlungen)
1. Aufgabenliste und zugehörige Dokumente auf `scripts/utility/…`-Pfade korrigieren.
2. CLI-Verhalten von `simulate_power_consumption.py` anpassen (separater `--help`-Modus, optional). 
3. Speicher-/Validierungs-Tests priorisieren: RAM-Berechnungen, Metrikfunktionen, Pizza-Inferenzskripte.
4. Diffusions-Test isoliert ausführen oder Skip-Strategie definieren, wenn Downloads lokal nicht gewünscht.
5. CI-Workflow für Regressionstests (`PERF-4.1`) konkretisieren und implementieren.
