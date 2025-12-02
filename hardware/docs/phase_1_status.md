# Phase 1 – Analyse des aktuellen Status (DFM-Check & Fertigungsfreigabe)

Stand: 2025-08-25

## Zusammenfassung
- EDA-Projektdateien vorhanden: `.kicad_pcb`, `.kicad_sch`, `.kicad_pro`.
- KiCad CLI installiert: 7.0.11. In KiCad 7 bietet die CLI keine direkten ERC/DRC-Befehle → ERC/DRC noch nicht ausgeführt (GUI oder KiBot nötig).
- BOM/CPL vorhanden; Prüfung inkl. Normalisierung für JLC durchgeführt.

## Details

### EDA-Projekt öffnen
Vorhandene Dateien in `hardware/eda/`:
- `PizzaBoard-RP2040.kicad_pcb`
- `PizzaBoard-RP2040.kicad_sch`
- `PizzaBoard-RP2040.kicad_pro`

KiCad-CLI auf dem System: gefunden (`kicad-cli` 7.0.11).

### ERC & DRC
- Aktueller Status: Nicht ausgeführt. Hinweis: KiCad 7 CLI bietet keine direkten ERC/DRC-Kommandos.
- Optionen:
  - Manuell in GUI: Eeschema → Tools → Electrical Rules Checker; Pcbnew → Inspect → Design Rules Checker; Reports als Datei exportieren nach `hardware/manufacturing/output/corrected_jlcpcb/`.
  - Alternativ: KiBot nutzen (falls installiert), um ERC/DRC headless zu erzeugen.

### BOM- und CPL-Prüfung (JLCPCB)
Dateien in `hardware/manufacturing/`:
- BOM: `bom_jlcpcb.csv`
- CPL: `cpl_jlcpcb.csv`

Ergebnis Normalisierung (automatisiert mit `hardware/manufacturing/normalize_mfg_files.py`):
- Ausgaben: `hardware/manufacturing/output/corrected_jlcpcb/bom_jlcpcb.csv`, `.../cpl_jlcpcb.csv`.
- BOM auf JLC-Spalten gemappt (`Comment,Designator,Footprint,LCSC Part #,Quantity`).
  - Befunde: 1 Position ohne LCSC-Nummer, 2 Positionen mit unbekanntem/ungenauem Package.
    - Missing LCSC: Designator = `BATTERYPOWER_1, CR123A_1, CAMERA_1, DEBUGPROG_1, AUDIO_1, ONOFF_1, R1_1` (Value `BATTERY POWER`).
    - Unklare Package-Angaben: obige Gruppe (`Unknown`), `USB_1` (`SMD`).
- CPL normalisiert: mm-Suffixe entfernt, Rotation numerisch formatiert; weitere Rotation/Orientierung noch zu verifizieren.
  - Befunde: 10 Einträge mit entfernten mm-Suffixen; 10 Einträge mit Rotation 0.00° (prüfen!).

Hinweis: Der vorhandene `.kicad_pcb` ist nicht vollständig KiCad-kompatibel (Parserfehler/Netz-IDs). Für verlässliche CPL/DRC/GERBER-Exporte sollte das echte Board aus KiCad exportiert oder KiBot/Projektquellen aktualisiert werden.

## Nächste Schritte
1) ERC/DRC in KiCad-GUI ausführen und Reports nach `hardware/manufacturing/output/corrected_jlcpcb/` exportieren.
2) Fehlende LCSC-Nummer ergänzen oder DNP kennzeichnen; Packages präzisieren (USB-C, BATTERY POWER-Gruppe).
3) CPL verifizieren: Koordinaten/Rotation/Side pro Bauteil prüfen, Board-Origin anpassen falls nötig.
4) Ergebnisse dokumentieren und Blocking-Issues markieren.

## Checkliste (Phase 1)
- [x] Sicherstellen, dass KiCad-Dateien vorliegen
- [x] KiCad-Version ≥ 7.x verifiziert (7.0.11)
- [ ] ERC-Report erzeugt und bewertet
- [ ] DRC-Report erzeugt und bewertet
- [x] BOM/CPL erste Plausibilitätsprüfung durchgeführt
