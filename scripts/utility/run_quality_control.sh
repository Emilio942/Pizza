#!/bin/bash
# 
# Bildqualitätskontrolle für Pizza-Klassifikation
#
# Dieses Skript führt den manuellen Qualitätskontrollprozess durch:
# 1. Generiert einen HTML-Report mit zufälligen Bildstichproben
# 2. Ermöglicht das Markieren von Bildern zur Löschung
# 3. Löscht die markierten Bilder aus dem Datensatz
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Verzeichnis für die Ausgabe
OUTPUT_DIR="$SCRIPT_DIR/output/quality_control"
mkdir -p "$OUTPUT_DIR"

# Startet den Qualitätskontrollprozess
# Argumente:
#   --samples: Anzahl der zufälligen Stichproben pro Kategorie (Standard: 50)
#   --image-dirs: Verzeichnisse mit den zu prüfenden Bildern (Standard: augmented_pizza)
#   --check-existing: Prüft vorhandene Kontrolldaten, zeigt nur unverarbeitete Bilder an
#
echo "Starte Bildqualitätskontrollprozess..."
python "$SCRIPT_DIR/scripts/image_quality_check.py" \
    --samples 50 \
    --image-dirs augmented_pizza \
    --output "$OUTPUT_DIR"

echo ""
echo "Nach dem Markieren der Bilder im Browser, führen Sie folgenden Befehl aus, um markierte Bilder zu löschen:"
echo "python $SCRIPT_DIR/scripts/image_quality_check.py --process-marked --output $OUTPUT_DIR"
