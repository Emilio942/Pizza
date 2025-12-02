# Phase 2 – Export & Artefakt-Erzeugung

Stand: 2025-08-25

## Ergebnisübersicht
- Automatisierter Export durchgeführt (korrigierter Exporter).
- Artefakte vorbereitet für JLCPCB-Upload.

## Generierte Artefakte
- Gerber (korrigiert): `hardware/manufacturing/output/corrected_jlcpcb/gerber/`
- Drill: `hardware/manufacturing/output/corrected_jlcpcb/gerber/PizzaBoard-RP2040.drl`
- Gerber Job: `hardware/manufacturing/output/corrected_jlcpcb/gerber/PizzaBoard-RP2040-job.gbrjob`
- BOM (JLC-Format, normalisiert): `hardware/manufacturing/output/corrected_jlcpcb/bom_jlcpcb.csv`
- CPL (normalisiert): `hardware/manufacturing/output/corrected_jlcpcb/cpl_jlcpcb.csv`
- Upload-Bundle: `hardware/manufacturing/JLCPCB_UPLOAD/`
  - `gerber_jlcpcb_FIXED.zip` (Gerber-only ZIP)
  - `bom_jlcpcb.csv`, `cpl_jlcpcb.csv`
  - `CORRECTED_PCB_REPORT.md`
  - `fresh_gerber/` mit Einzeldateien

## Kurzcheck (Gerber-Viewer)
- Layer: F.Cu, B.Cu, F.Mask, B.Mask, F.Silkscreen, B.Silkscreen, Edge_Cuts vorhanden.
- Bohrungen: `.drl` generiert.
- Fiducials: nicht explizit vorhanden → bei Assembly ggf. hinzufügen.
- CPL: Koordinaten ohne „mm“-Suffix; Rotationen teils 0.00° → visuell prüfen.

## Hinweise
- Repo-`.kicad_pcb` ist nicht vollständig KiCad-CLI-kompatibel. Für echte ERC/DRC/CPL direkt aus KiCad bitte Board in KiCad öffnen und neu exportieren.
- BOM/CPL normalisiert. Offene Punkte:
  - 1 Position ohne LCSC (BATTERY POWER-Gruppe) → ergänzen oder DNP.
  - Package bei `USB_1` zu grob ("SMD").

## JLCPCB Smoke-Test (manuell)
1) https://cart.jlcpcb.com/quote → `gerber_jlcpcb_FIXED.zip` hochladen.
2) Assembly aktivieren → `bom_jlcpcb.csv` und `cpl_jlcpcb.csv` laden.
3) Fehlerbericht prüfen/speichern; Screenshots optional nach `hardware/docs/`.
4) Rotationen/Layer im CPL bei Warnungen anpassen und erneut exportieren.
