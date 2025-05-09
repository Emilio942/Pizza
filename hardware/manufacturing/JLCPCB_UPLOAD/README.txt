# JLCPCB UPLOAD-DATEIEN
# ====================

Dieser Ordner enthält alle notwendigen Dateien für den JLCPCB-Fertigungsprozess
für das RP2040 Pizza Detection System.

## DATEIEN FÜR UPLOAD

1. gerber_jlcpcb.zip
   - Upload bei: JLCPCB "Quote Now" → Gerber-Datei-Upload
   - Enthält alle Leiterplatten-Layer (8 Layer-Design)
   - Wichtig: Bei den PCB-Einstellungen "8 Layers" auswählen!

2. bom_jlcpcb.csv
   - Upload bei: "SMT Assembly" → "Add BOM File"
   - Enthält alle Komponenten mit LCSC-Teilenummern
   - Format: JLCPCB-kompatibel

3. cpl_jlcpcb.csv
   - Upload bei: "SMT Assembly" → "Add CPL File"
   - Enthält Bauteilpositionen und Rotationen
   - Alle Komponenten auf "Top Side"
   - Format: JLCPCB-kompatibel

## UPLOAD-REIHENFOLGE BEI JLCPCB

1. Gehen Sie zu https://jlcpcb.com/ und klicken Sie auf "Quote Now"
2. Laden Sie zuerst gerber_jlcpcb.zip hoch
3. Konfigurieren Sie die PCB-Einstellungen:
   - Layers: 8
   - PCB Thickness: 1.6mm
   - Andere Einstellungen nach Bedarf
4. Aktivieren Sie die "SMT Assembly" Option
5. Wählen Sie "Top Side" für die Bestückung
6. Laden Sie bom_jlcpcb.csv als BOM-Datei hoch
7. Laden Sie cpl_jlcpcb.csv als CPL-Datei hoch
8. Überprüfen Sie die Komponenten und bestätigen Sie die Bestellung

Stand: 8. Mai 2025
RP2040 Pizza Detection System
