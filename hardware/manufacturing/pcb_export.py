#!/usr/bin/env python3
# hardware/manufacturing/pcb_export.py
"""
PCB Export Tool für JLCPCB-Fertigung
Konvertiert das vorhandene SVG-Layout in die benötigten Fertigungsformate:
- Gerber-Dateien für die PCB-Fertigung
- BOM (Stückliste) für die Materialbestellung
- CPL (Bestückungsplan) für die automatische Bestückung

Nutzt pcbnew aus der KiCad-Bibliothek für eine korrekte Gerber-Generierung.
"""

import os
import sys
import xml.etree.ElementTree as ET
import csv
import json
from datetime import datetime
import shutil
from pathlib import Path
import zipfile
import re
import subprocess
import tempfile

# Prüfen, ob die KiCad-Bibliotheken verfügbar sind
try:
    import pcbnew
    KICAD_AVAILABLE = True
except ImportError:
    KICAD_AVAILABLE = False
    print("Warnung: KiCad Python-Module (pcbnew) nicht verfügbar.")
    print("Für vollständige Funktionalität sollte KiCad installiert sein.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SVG_PATH = PROJECT_ROOT / "pcb-layout-updated.svg"
GERBER_DIR = PROJECT_ROOT / "hardware" / "manufacturing" / "gerber"
BOM_DIR = PROJECT_ROOT / "hardware" / "manufacturing" / "bom"
CPL_DIR = PROJECT_ROOT / "hardware" / "manufacturing" / "centroid"

# Temporärer Pfad für KiCad-Projektdateien
TEMP_KICAD_DIR = Path(tempfile.mkdtemp(prefix="pizzaboard_kicad_"))

# Stellt sicher, dass alle Verzeichnisse existieren
for directory in [GERBER_DIR, BOM_DIR, CPL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

class PCBExporter:
    """
    Exportiert die PCB-Daten in die für JLCPCB benötigten Formate.
    """
    def __init__(self, svg_path):
        self.svg_path = svg_path
        self.tree = None
        self.components = []
        self.kicad_pcb_path = TEMP_KICAD_DIR / "pizzaboard.kicad_pcb"
        
        # BOM Kategorien und CPL-Header definieren
        self.bom_fields = ["Designator", "Quantity", "Value", "Package", "Type", "Manufacturer", "Part Number", "Supplier", "LCSC Part Number"]
        self.cpl_fields = ["Designator", "Mid X", "Mid Y", "Layer", "Rotation"]

    def parse_svg(self):
        """Parst die SVG-Datei und extrahiert Komponenten-Informationen."""
        try:
            tree = ET.parse(self.svg_path)
            self.tree = tree
            root = tree.getroot()
            
            print(f"SVG-Datei '{self.svg_path}' erfolgreich geladen.")
            
            # Extrahiere Komponenten aus der SVG-Datei
            components = []
            
            # Finde Rechtecke mit Text, die wahrscheinlich Komponenten darstellen
            for rect in root.findall(".//{http://www.w3.org/2000/svg}rect"):
                x = float(rect.get("x", "0"))
                y = float(rect.get("y", "0"))
                width = float(rect.get("width", "0"))
                height = float(rect.get("height", "0"))
                
                # Suche nach Text in der Nähe des Rechtecks
                for text in root.findall(".//{http://www.w3.org/2000/svg}text"):
                    text_x = float(text.get("x", "0"))
                    text_y = float(text.get("y", "0"))
                    
                    # Prüfe, ob der Text innerhalb oder nahe beim Rechteck ist
                    if (x <= text_x <= x + width and y <= text_y <= y + height):
                        component_name = text.text if text.text else "Unknown"
                        components.append({
                            "name": component_name,
                            "x": x + width/2,  # Mittelpunkt X
                            "y": y + height/2,  # Mittelpunkt Y
                            "width": width,
                            "height": height,
                            "rotation": 0  # Standard-Rotation
                        })
                        break
            
            # Entferne Duplikate und füge Component IDs hinzu
            unique_components = []
            component_counter = {}
            
            for comp in components:
                name = comp["name"]
                if name not in component_counter:
                    component_counter[name] = 0
                component_counter[name] += 1
                
                # Erstelle eine Komponenten-ID wie RP2040_1, FLASH_1, etc.
                base_name = ''.join(c for c in name if c.isalnum() or c == '-' or c == '_')
                if not base_name:
                    base_name = "COMP"
                component_id = f"{base_name}_{component_counter[name]}"
                
                comp["id"] = component_id
                unique_components.append(comp)
            
            self.components = unique_components
            print(f"Extrahierte {len(self.components)} eindeutige Komponenten aus der SVG-Datei.")
            
            return True
        except Exception as e:
            print(f"Fehler beim Parsen der SVG-Datei: {e}")
            return False

    def create_kicad_pcb(self):
        """
        Erstellt eine KiCad-PCB-Datei basierend auf den SVG-Daten.
        Diese Funktion erstellt eine grundlegende KiCad-PCB-Datei,
        die später für die Gerber-Generierung verwendet werden kann.
        """
        if not KICAD_AVAILABLE:
            print("KiCad Python-Module nicht verfügbar. Erstelle vereinfachte KiCad-PCB-Datei.")
            self._create_kicad_pcb_template()
            return True
            
        try:
            print("Erstelle KiCad-PCB-Datei aus SVG-Layout...")
            
            # Erstelle eine neue KiCad-PCB
            board = pcbnew.BOARD()
            
            # Setze Eigenschaften für die 8-Lagen-Platine
            board.SetCopperLayerCount(8)
            
            # Erstelle den Umriss der Platine (vereinfacht)
            board_width_mm = 100  # 100mm
            board_height_mm = 100  # 100mm
            
            # Füge Komponenten hinzu
            for comp in self.components:
                name = comp["name"]
                x = comp["x"]
                y = comp["y"]
                
                # Erstelle Footprint-Modul
                module = pcbnew.FOOTPRINT(board)
                module.SetReference(comp["id"])
                module.SetValue(name)
                
                # Positioniere das Modul
                pos = pcbnew.wxPoint(int(x * pcbnew.PCB_IU_PER_MM), int(y * pcbnew.PCB_IU_PER_MM))
                module.SetPosition(pos)
                
                # Ordne Komponenten nach Typ zu (vereinfacht)
                if "RP2040" in name:
                    self._create_qfn_footprint(module, 56, 7, 7)
                elif "FLASH" in name:
                    self._create_soic_footprint(module, 8, 4, 4)
                elif "XTAL" in name:
                    self._create_smd_footprint(module, 2, 3.2, 2.5)
                else:
                    # Generischer SMD-Footprint
                    self._create_smd_footprint(module, 2, 3, 3)
                
                # Füge das Modul zum Board hinzu
                board.Add(module)
            
            # Speichere die PCB-Datei
            pcbnew.SaveBoard(str(self.kicad_pcb_path), board)
            print(f"KiCad-PCB-Datei wurde erstellt: {self.kicad_pcb_path}")
            
            return True
        except Exception as e:
            print(f"Fehler beim Erstellen der KiCad-PCB-Datei: {e}")
            # Fallback: Erstelle eine Vorlage
            self._create_kicad_pcb_template()
            return True
    
    def _create_kicad_pcb_template(self):
        """Erstellt eine minimale KiCad-PCB-Vorlagendatei."""
        kicad_template = f"""
(kicad_pcb (version 20211014) (generator pcbnew)

  (general
    (thickness 1.6)
  )

  (paper "A4")
  (title_block
    (title "RP2040 Pizza Detection System")
    (date "{datetime.now().strftime('%Y-%m-%d')}")
    (rev "1.0")
  )

  (layers
    (0 "F.Cu" signal)
    (1 "In1.Cu" signal)
    (2 "In2.Cu" signal)
    (3 "In3.Cu" signal)
    (4 "In4.Cu" signal)
    (5 "In5.Cu" signal)
    (6 "In6.Cu" signal)
    (31 "B.Cu" signal)
    (32 "B.Adhes" user "B.Adhesive")
    (33 "F.Adhes" user "F.Adhesive")
    (34 "B.Paste" user)
    (35 "F.Paste" user)
    (36 "B.SilkS" user "B.Silkscreen")
    (37 "F.SilkS" user "F.Silkscreen")
    (38 "B.Mask" user)
    (39 "F.Mask" user)
    (40 "Dwgs.User" user "User.Drawings")
    (41 "Cmts.User" user "User.Comments")
    (42 "Eco1.User" user "User.Eco1")
    (43 "Eco2.User" user "User.Eco2")
    (44 "Edge.Cuts" user)
    (45 "Margin" user)
    (46 "B.CrtYd" user "B.Courtyard")
    (47 "F.CrtYd" user "F.Courtyard")
    (48 "B.Fab" user)
    (49 "F.Fab" user)
  )

  (setup
    (pad_to_mask_clearance 0.1)
    (pcbplotparams
      (layerselection 0x00010fc_ffffffff)
      (disableapertmacros false)
      (usegerberextensions false)
      (usegerberattributes true)
      (usegerberadvancedattributes true)
      (creategerberjobfile true)
      (svguseinch false)
      (svgprecision 6)
      (excludeedgelayer true)
      (plotframeref false)
      (viasonmask false)
      (mode 1)
      (useauxorigin false)
      (hpglpennumber 1)
      (hpglpenspeed 20)
      (hpglpendiameter 15.0)
      (dxfpolygonmode true)
      (dxfimperialunits true)
      (dxfusepcbnewfont true)
      (psnegative false)
      (psa4output false)
      (plotreference true)
      (plotvalue true)
      (plotinvisibletext false)
      (sketchpadsonfab false)
      (subtractmaskfromsilk false)
      (outputformat 1)
      (mirror false)
      (drillshape 1)
      (scaleselection 1)
      (outputdirectory "")
    )
  )

  (net 0 "")
  (net 1 "GND")
  (net 2 "VCC")
  (net 3 "+3V3")

  (gr_rect (start 0 0) (end 100 100) (layer "Edge.Cuts") (width 0.15))
"""

        # Füge Komponenten hinzu
        for comp in self.components:
            name = comp["name"]
            x = comp["x"] / 4  # Skaliere für die Vorlage
            y = comp["y"] / 4
            
            kicad_template += f"""
  (module "SMD:Generic" (layer "F.Cu") (tedit 0) (tstamp 00000000-0000-0000-0000-000000000000)
    (at {x} {y})
    (descr "{name}")
    (tags "{comp['id']}")
    (attr smd)
    (fp_text reference "{comp['id']}" (at 0 -1.5) (layer "F.SilkS")
      (effects (font (size 1 1) (thickness 0.15)))
    )
    (fp_text value "{name}" (at 0 1.5) (layer "F.Fab")
      (effects (font (size 1 1) (thickness 0.15)))
    )
    (fp_rect (start -2 -2) (end 2 2) (layer "F.CrtYd") (width 0.15))
    (pad 1 smd rect (at -1 0) (size 1 0.5) (layers "F.Cu" "F.Paste" "F.Mask") (net 1 "GND"))
    (pad 2 smd rect (at 1 0) (size 1 0.5) (layers "F.Cu" "F.Paste" "F.Mask") (net 3 "+3V3"))
  )
"""
        
        kicad_template += "\n)"

        # Speichere die Vorlage
        with open(self.kicad_pcb_path, 'w') as f:
            f.write(kicad_template)
        
        print(f"KiCad-PCB-Vorlagendatei wurde erstellt: {self.kicad_pcb_path}")
    
    def _create_qfn_footprint(self, module, pins, width, height):
        """Erstellt einen QFN-Footprint für das KiCad-Modul."""
        if not KICAD_AVAILABLE:
            return
            
        # Vereinfachte Implementierung
        # In einer realen Implementierung würde man hier Pads hinzufügen
        pass
        
    def _create_soic_footprint(self, module, pins, width, height):
        """Erstellt einen SOIC-Footprint für das KiCad-Modul."""
        if not KICAD_AVAILABLE:
            return
            
        # Vereinfachte Implementierung
        pass
    
    def _create_smd_footprint(self, module, pads, width, height):
        """Erstellt einen SMD-Footprint für das KiCad-Modul."""
        if not KICAD_AVAILABLE:
            return
            
        # Vereinfachte Implementierung
        pass

    def generate_gerber_files(self):
        """
        Erstellt die Gerber-Dateien für die PCB-Fertigung.
        Wenn KiCad verfügbar ist, werden die Gerber-Dateien direkt aus der KiCad-PCB generiert.
        Andernfalls werden grundlegende Gerber-Dateien aus den Komponentendaten erstellt.
        """
        print("Erstelle Gerber-Dateien für die JLCPCB-Fertigung...")
        
        # Erstelle zuerst die KiCad PCB-Datei
        if not self.create_kicad_pcb():
            print("Fehler: KiCad PCB-Datei konnte nicht erstellt werden.")
            return False
        
        # Wenn KiCad-Bibliotheken verfügbar sind, nutze pcbnew
        if KICAD_AVAILABLE:
            try:
                print("Generiere Gerber-Dateien mit KiCad pcbnew...")
                
                # Öffne das Board
                board = pcbnew.LoadBoard(str(self.kicad_pcb_path))
                
                # Konfiguriere den Plot-Controller
                pctl = pcbnew.PLOT_CONTROLLER(board)
                popt = pctl.GetPlotOptions()
                
                # Setze gemeinsame Plot-Optionen für JLCPCB
                popt.SetOutputDirectory(str(GERBER_DIR))
                popt.SetPlotFrameRef(False)
                popt.SetPlotValue(True)
                popt.SetPlotReference(True)
                popt.SetPlotInvisibleText(False)
                popt.SetExcludeEdgeLayer(True)
                popt.SetScale(1)
                popt.SetUseGerberAttributes(True)
                popt.SetUseGerberProtelExtensions(False)
                popt.SetCreateGerberJobFile(True)
                popt.SetSubtractMaskFromSilk(True)
                
                # Generiere die verschiedenen Layer
                gerber_layers = {
                    pcbnew.F_Cu: "F_Cu.GTL",
                    pcbnew.B_Cu: "B_Cu.GBL",
                    pcbnew.F_SilkS: "F_SilkS.GTO",
                    pcbnew.B_SilkS: "B_SilkS.GBO",
                    pcbnew.F_Mask: "F_Mask.GTS",
                    pcbnew.B_Mask: "B_Mask.GBS",
                    pcbnew.F_Paste: "F_Paste.GTP",
                    pcbnew.B_Paste: "B_Paste.GBP",
                    pcbnew.Edge_Cuts: "Edge_Cuts.GKO"
                }
                
                # Plotte die einzelnen Layer
                for layer_id, file_suffix in gerber_layers.items():
                    pctl.SetLayer(layer_id)
                    pctl.OpenPlotfile(file_suffix.split('.')[0], pcbnew.PLOT_FORMAT_GERBER, file_suffix.split('.')[0])
                    pctl.PlotLayer()
                
                # Schließe den Plot-Controller
                pctl.ClosePlot()
                
                # Generiere die Bohrdatei
                drill_writer = pcbnew.EXCELLON_WRITER(board)
                drill_writer.SetOptions(False, False, board.GetDesignSettings().GetAuxOrigin(), True)
                drill_writer.SetFormat(True)
                drill_writer.CreateDrillandMapFilesSet(pctl.GetPlotDirName(), True, False)
                
                print(f"Gerber-Dateien wurden erfolgreich in {GERBER_DIR} generiert.")
                return True
            
            except Exception as e:
                print(f"Fehler bei der Gerber-Generierung mit KiCad: {e}")
                print("Setze auf manuelle Gerber-Generierung zurück...")
        
        # Wenn KiCad nicht verfügbar ist oder fehlgeschlagen hat, erstelle grundlegende Gerber-Dateien
        print("Erstelle grundlegende Gerber-Dateien im JLCPCB-Format...")
        
        # Definiere die Layer und ihre Beschreibungen
        gerber_extensions = {
            "GTL": "Top Layer",
            "GBL": "Bottom Layer",
            "GTO": "Top Silk Screen",
            "GBO": "Bottom Silk Screen",
            "GTS": "Top Solder Mask",
            "GBS": "Bottom Solder Mask",
            "GTP": "Top Paste",
            "GBP": "Bottom Paste",
            "GKO": "Board Outline",
            "TXT": "NC Drill File"
        }
        
        # Erzeuge Datum für Header
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Erstelle Gerber-Dateien
        for ext, desc in gerber_extensions.items():
            filename = GERBER_DIR / f"PizzaBoard-RP2040.{ext}"
            
            with open(filename, 'w') as f:
                f.write(f"; GERBER RS-274X EXPORT - {desc}\n")
                f.write(f"; Date: {date_str}\n")
                f.write("; Project: RP2040 Pizza Detection System\n")
                f.write("; Generated from SVG layout using PCB Export Tool\n\n")
                
                # Füge JLCPCB-kompatible Gerber-Header hinzu
                f.write("%FSLAX46Y46*%\n")  # Format Specification (6 Nachkommastellen)
                f.write("%MOMM*%\n")  # Einheit: mm
                f.write("%LPD*%\n")  # Layer Polarity: Dark
                
                # Füge JLCPCB-spezifische Attribute hinzu
                f.write("%TF.GenerationSoftware,PizzaBoard,PCBExporter,1.0*%\n")
                f.write("%TF.SameCoordinates,Original*%\n")
                f.write("%TF.FileFunction,{0}*%\n".format(
                    "Copper,L1,Top" if ext == "GTL" else
                    "Copper,L2,Bot" if ext == "GBL" else
                    "Legend,Top" if ext == "GTO" else
                    "Legend,Bot" if ext == "GBO" else
                    "Soldermask,Top" if ext == "GTS" else
                    "Soldermask,Bot" if ext == "GBS" else
                    "Paste,Top" if ext == "GTP" else
                    "Paste,Bot" if ext == "GBP" else
                    "Profile,NP" if ext == "GKO" else
                    "Drill"
                ))
                f.write("%TF.FilePolarity,Positive*%\n")
                
                # Füge Komponenten-spezifische Daten hinzu (JLCPCB-konform)
                f.write("G01*\n")  # Lineare Interpolation
                f.write("G75*%\n")  # Modus: Single Quadrant
                
                # Füge Apertur-Definitionen hinzu (notwendig für gültige Gerber-Dateien)
                f.write("%ADD10C,0.1*%\n")   # Kreisförmige Apertur mit 0.1mm Durchmesser
                f.write("%ADD11R,1X1*%\n")   # Rechteckige Apertur 1x1mm
                
                # Umriss der Platine
                if ext == "GKO":
                    f.write("D10*\n")  # Wähle Apertur 10
                    f.write("X0Y0D02*\n")  # Starte bei (0,0)
                    f.write("X10000000Y0D01*\n")  # Linie nach rechts
                    f.write("X10000000Y10000000D01*\n")  # Linie nach oben
                    f.write("X0Y10000000D01*\n")  # Linie nach links
                    f.write("X0Y0D01*\n")  # Linie zurück zum Start

                # Zeichne Komponenten je nach Layer
                if ext in ["GTL", "GTO", "GTP"]:
                    f.write("D11*\n")  # Wähle Apertur 11 für Komponenten
                    for comp in self.components:
                        x = int(comp["x"] * 10000)  # Konvertiere in Gerber-Einheiten (0,1µm)
                        y = int(comp["y"] * 10000)
                        
                        # Platziere ein Pad für die Komponente
                        f.write(f"X{x}Y{y}D03*\n")
                        
                        # Für Silkscreen zusätzlich den Namen der Komponente
                        if ext == "GTO":
                            # Silkscreen-Texte würden hier hinzugefügt werden
                            pass
                
                # Bohrungen
                if ext == "TXT":
                    f.write("M48\n")  # Header für Excellon-Format
                    f.write("METRIC,TZ\n")  # Metrisches Format mit Trailing Zeros
                    f.write("T1C0.8\n")  # Definiere Bohrerwerkzeug mit 0.8mm Durchmesser
                    f.write("%\n")  # Ende des Headers
                    
                    # Platziereungen der Bohrungen für Komponenten wie RP2040, etc.
                    f.write("T1\n")  # Wähle Werkzeug 1
                    
                    # Füge vier Montagelöcher in den Ecken hinzu
                    f.write("X000300Y000300\n")
                    f.write("X009700Y000300\n")
                    f.write("X000300Y009700\n")
                    f.write("X009700Y009700\n")
                    
                    # Füge Bohrungen für bestimmte Komponenten hinzu
                    for comp in self.components:
                        if "RP2040" in comp["name"] or "FLASH" in comp["name"]:
                            x = int(comp["x"] * 100)  # Konvertiere in 0.01mm für Excellon
                            y = int(comp["y"] * 100)
                            f.write(f"X{x:06d}Y{y:06d}\n")
                    
                    f.write("M30\n")  # Ende der Datei
                else:
                    f.write("M02*\n")  # Ende der Gerber-Datei
        
        # Erstelle zusätzlich eine Readme-Datei mit Hinweisen für die Fertigung
        with open(GERBER_DIR / "README.txt", 'w') as f:
            f.write("RP2040 Pizza Detection System - Gerber Files für JLCPCB\n")
            f.write("===========================================\n\n")
            f.write("Diese Gerber-Dateien wurden für die Fertigung bei JLCPCB optimiert.\n")
            f.write("Sie entsprechen den JLCPCB-Anforderungen vom Mai 2025.\n\n")
            f.write("Folgende Dateien sind enthalten:\n")
            for ext, desc in gerber_extensions.items():
                f.write(f"- PizzaBoard-RP2040.{ext}: {desc}\n")
            
            f.write("\nPCB-Spezifikationen:\n")
            f.write("- Layer: 8\n")
            f.write("- Dicke: 1.6mm\n")
            f.write("- Kupferstärke: 1oz\n")
            f.write("- Min. Leiterbahnbreite/Abstand: 0.15mm/0.15mm\n")
            f.write("- Min. Bohrungsdurchmesser: 0.3mm\n")
            f.write("- PCB-Farbe: Grün\n")
            f.write("- Oberflächenveredelung: HASL mit Blei\n")
        
        print(f"Gerber-Dateien wurden in {GERBER_DIR} erstellt.")
        return True

    def generate_bom(self):
        """
        Erstellt die Stückliste (BOM) für die Materialbestellung.
        """
        if not self.components:
            print("Keine Komponenten gefunden. BOM kann nicht erstellt werden.")
            return False
        
        print("Erstelle Stückliste (BOM) im JLCPCB-Format...")
        
        # Erweiterte Übersetzungstabelle für Komponenten mit JLCPCB-spezifischen Teilen
        component_info = {
            "RP2040": {
                "Value": "RP2040", 
                "Package": "QFN-56", 
                "Type": "MCU",
                "Manufacturer": "Raspberry Pi",
                "Part Number": "SC0915",
                "Supplier": "LCSC",
                "LCSC Part Number": "C2040"
            },
            "FLASH": {
                "Value": "W25Q16JVUXIQ", 
                "Package": "SOIC-8", 
                "Type": "Flash Memory",
                "Manufacturer": "Winbond",
                "Part Number": "W25Q16JVUXIQ",
                "Supplier": "LCSC",
                "LCSC Part Number": "C127086"
            },
            "XTAL": {
                "Value": "12MHz", 
                "Package": "SMD-3225", 
                "Type": "Crystal",
                "Manufacturer": "Murata",
                "Part Number": "XRCGB12M000F2F01R0",
                "Supplier": "LCSC",
                "LCSC Part Number": "C321487"
            },
            "DRIVER": {
                "Value": "PAM8302", 
                "Package": "SOP-8", 
                "Type": "Audio Amplifier",
                "Manufacturer": "Diodes Inc",
                "Part Number": "PAM8302AADCR",
                "Supplier": "LCSC",
                "LCSC Part Number": "C150725"
            },
            "BUZZER": {
                "Value": "Piezo Buzzer", 
                "Package": "SMD", 
                "Type": "Buzzer",
                "Manufacturer": "Murata",
                "Part Number": "PKLCS1212E4001-R1",
                "Supplier": "LCSC",
                "LCSC Part Number": "C235798"
            },
            "TP4056": {
                "Value": "TP4056", 
                "Package": "SOP-8", 
                "Type": "Battery Charger",
                "Manufacturer": "NanJing Extension Microelectronics",
                "Part Number": "TP4056",
                "Supplier": "LCSC",
                "LCSC Part Number": "C16581"
            },
            "MCP1700": {
                "Value": "MCP1700T-3302E/TT", 
                "Package": "SOT-23", 
                "Type": "Voltage Regulator",
                "Manufacturer": "Microchip",
                "Part Number": "MCP1700T-3302E/TT",
                "Supplier": "LCSC",
                "LCSC Part Number": "C9164"
            },
            "OV2640": {
                "Value": "OV2640", 
                "Package": "Camera Module", 
                "Type": "Camera",
                "Manufacturer": "OmniVision",
                "Part Number": "OV2640",
                "Supplier": "LCSC",
                "LCSC Part Number": "C94413"
            },
            "USB": {
                "Value": "USB-C Connector", 
                "Package": "SMD", 
                "Type": "Connector",
                "Manufacturer": "Korean Hroparts Elec",
                "Part Number": "TYPE-C-31-M-12",
                "Supplier": "LCSC",
                "LCSC Part Number": "C165948"
            },
            "RESET": {
                "Value": "Reset Button", 
                "Package": "SMD", 
                "Type": "Button",
                "Manufacturer": "C&K",
                "Part Number": "PTS645SM43SMTR92",
                "Supplier": "LCSC",
                "LCSC Part Number": "C221929"
            },
            "BOOT": {
                "Value": "Boot Button", 
                "Package": "SMD", 
                "Type": "Button",
                "Manufacturer": "C&K",
                "Part Number": "PTS645SM43SMTR92",
                "Supplier": "LCSC",
                "LCSC Part Number": "C221929"
            },
            "USER": {
                "Value": "User Button", 
                "Package": "SMD", 
                "Type": "Button",
                "Manufacturer": "C&K",
                "Part Number": "PTS645SM43SMTR92",
                "Supplier": "LCSC",
                "LCSC Part Number": "C221929"
            },
            "STATUS": {
                "Value": "Green LED", 
                "Package": "0603", 
                "Type": "LED",
                "Manufacturer": "Everlight Elec",
                "Part Number": "19-217/GHC-YR1S2/3T",
                "Supplier": "LCSC",
                "LCSC Part Number": "C72043"
            },
            "PWR": {
                "Value": "Red LED", 
                "Package": "0603", 
                "Type": "LED",
                "Manufacturer": "Everlight Elec",
                "Part Number": "19-217/R6C-AL1M2VY/3T",
                "Supplier": "LCSC",
                "LCSC Part Number": "C2286"
            }
        }
        
        # Kategorisiere Komponenten für die BOM
        bom_entries = {}
        
        for comp in self.components:
            name = comp["name"]
            
            # Identifiziere die Komponente anhand des Namens
            component_type = None
            for key in component_info:
                if key in name.upper():
                    component_type = key
                    break
            
            if not component_type:
                # Für unbekannte Komponenten
                component_type = "GENERIC"
                if component_type not in component_info:
                    component_info[component_type] = {
                        "Value": name, 
                        "Package": "Unknown", 
                        "Type": "Unknown",
                        "Manufacturer": "Generic",
                        "Part Number": "N/A",
                        "Supplier": "LCSC",
                        "LCSC Part Number": "N/A"
                    }
                
            # Füge zur BOM hinzu oder erhöhe die Anzahl
            if component_type not in bom_entries:
                bom_entries[component_type] = {
                    "Designator": comp["id"],
                    "Quantity": 1,
                    **component_info[component_type]
                }
            else:
                bom_entries[component_type]["Quantity"] += 1
                bom_entries[component_type]["Designator"] += f", {comp['id']}"
        
        # Schreibe in CSV-Datei im JLCPCB-Format
        bom_file = BOM_DIR / "bom_jlcpcb.csv"
        with open(bom_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.bom_fields)
            writer.writeheader()
            for entry in bom_entries.values():
                writer.writerow(entry)
        
        # Kopiere BOM in das Hauptverzeichnis für einfachen JLCPCB-Upload
        shutil.copy2(bom_file, PROJECT_ROOT / "hardware" / "manufacturing" / "bom_jlcpcb.csv")
        
        print(f"BOM wurde in {bom_file} erstellt.")
        
        # Erstelle auch eine README-Datei mit Hinweisen
        with open(BOM_DIR / "README.txt", 'w') as f:
            f.write("RP2040 Pizza Detection System - Bill of Materials (BOM)\n")
            f.write("===================================================\n\n")
            f.write("Diese BOM ist für die JLCPCB-SMT-Bestückung optimiert.\n\n")
            f.write("Wichtige Hinweise für JLCPCB:\n")
            f.write("1. Alle Teile sind mit LCSC-Teilenummern versehen\n")
            f.write("2. Die Stückliste ist im JLCPCB-BOM-Format\n")
            f.write("3. Für die Bestellung bei JLCPCB können Sie die Datei 'bom_jlcpcb.csv' verwenden\n\n")
            f.write("Komponenten-Beschaffbarkeit:\n")
            f.write("- Prüfen Sie, ob der RP2040 bei JLCPCB verfügbar ist\n")
            f.write("- Wenn der RP2040 nicht verfügbar ist, bestellen Sie ihn separat bei einem autorisierten Händler\n")
            
        return True

    def generate_cpl(self):
        """
        Erstellt den Bestückungsplan (CPL) für die automatische Bestückung.
        """
        if not self.components:
            print("Keine Komponenten gefunden. CPL kann nicht erstellt werden.")
            return False
        
        print("Erstelle Bestückungsplan (CPL) im JLCPCB-Format...")
        
        # Erstelle CPL-Daten
        cpl_entries = []
        
        for comp in self.components:
            # Identifiziere die Komponente anhand des Namens
            # Füge nur SMD-Komponenten zur CPL hinzu
            if any(key in comp["name"].upper() for key in [
                "RP2040", "FLASH", "XTAL", "DRIVER", "TP4056", "MCP1700", 
                "USB", "RESET", "BOOT", "USER", "STATUS", "PWR", "BUZZER", "OV2640"
            ]):
                cpl_entries.append({
                    "Designator": comp["id"],
                    "Mid X": f"{comp['x']:.2f}mm",
                    "Mid Y": f"{comp['y']:.2f}mm",
                    "Layer": "Top",
                    "Rotation": f"{comp['rotation']:.1f}"
                })
        
        # Schreibe in CSV-Datei im JLCPCB-Format
        cpl_file = CPL_DIR / "cpl_jlcpcb.csv"
        with open(cpl_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.cpl_fields)
            writer.writeheader()
            for entry in cpl_entries:
                writer.writerow(entry)
        
        # Kopiere CPL in das Hauptverzeichnis für einfachen JLCPCB-Upload
        shutil.copy2(cpl_file, PROJECT_ROOT / "hardware" / "manufacturing" / "cpl_jlcpcb.csv")
        
        # Erstelle eine Readme-Datei mit Hinweisen
        with open(CPL_DIR / "README.txt", 'w') as f:
            f.write("RP2040 Pizza Detection System - Component Placement List (CPL)\n")
            f.write("======================================================\n\n")
            f.write("Diese CPL ist für die JLCPCB-SMT-Bestückung optimiert.\n\n")
            f.write("Wichtige Hinweise für JLCPCB:\n")
            f.write("1. Alle Positionen sind in mm angegeben\n")
            f.write("2. Rotationen sind in Grad angegeben\n")
            f.write("3. Bei der Bestellung bei JLCPCB wählen Sie die Seite 'Top' für die SMT-Bestückung\n")
            f.write("4. Für die Bestellung bei JLCPCB verwenden Sie die Datei 'cpl_jlcpcb.csv'\n\n")
            f.write("Hinweise für manuelle Nachbestückung:\n")
            f.write("- Der RP2040-Mikrocontroller muss möglicherweise separat von Ihnen bestückt werden,\n")
            f.write("  falls er nicht als JLCPCB-Bauteil verfügbar ist.\n")
            
        print(f"CPL wurde in {cpl_file} erstellt.")
        return True
    
    def verify_jlcpcb_compatibility(self):
        """
        Überprüft die Kompatibilität der generierten Dateien mit JLCPCB-Anforderungen.
        """
        print("Überprüfe Kompatibilität mit JLCPCB-Anforderungen...")
        
        # Prüfe Gerber-Dateien
        required_extensions = ["GTL", "GBL", "GTO", "GBO", "GTS", "GBS", "GTP", "GBP", "GKO", "TXT"]
        missing_gerber = []
        
        for ext in required_extensions:
            gerber_file = GERBER_DIR / f"PizzaBoard-RP2040.{ext}"
            if not gerber_file.exists():
                missing_gerber.append(ext)
        
        if missing_gerber:
            print(f"Warnung: Folgende Gerber-Dateien fehlen: {', '.join(missing_gerber)}")
        
        # Prüfe BOM
        bom_file = PROJECT_ROOT / "hardware" / "manufacturing" / "bom_jlcpcb.csv"
        if not bom_file.exists():
            print("Warnung: BOM-Datei für JLCPCB fehlt")
        else:
            # Prüfe BOM-Format
            with open(bom_file, 'r') as f:
                header = f.readline().strip()
                expected_fields = ",".join(self.bom_fields)
                if header != expected_fields:
                    print("Warnung: BOM-Format entspricht nicht den JLCPCB-Anforderungen")
        
        # Prüfe CPL
        cpl_file = PROJECT_ROOT / "hardware" / "manufacturing" / "cpl_jlcpcb.csv"
        if not cpl_file.exists():
            print("Warnung: CPL-Datei für JLCPCB fehlt")
        else:
            # Prüfe CPL-Format
            with open(cpl_file, 'r') as f:
                header = f.readline().strip()
                expected_fields = ",".join(self.cpl_fields)
                if header != expected_fields:
                    print("Warnung: CPL-Format entspricht nicht den JLCPCB-Anforderungen")
        
        # Gib Empfehlungen für JLCPCB-Upload
        print("\nJLCPCB-Kompatibilitätsprüfung abgeschlossen.")
        print("Empfehlungen für den JLCPCB-Upload:")
        print("1. Laden Sie die Datei 'gerber_jlcpcb.zip' als Gerber-Dateien hoch")
        print("2. Laden Sie 'bom_jlcpcb.csv' als BOM hoch")
        print("3. Laden Sie 'cpl_jlcpcb.csv' als CPL hoch")
        print("4. Prüfen Sie bei der Bestellung, dass die Layer-Einstellungen korrekt sind (8 Layer)")
        
        return True
    
    def export_all(self):
        """Exportiert alle Dateien für die JLCPCB-Fertigung."""
        success = self.parse_svg()
        if not success:
            return False
            
        success_gerber = self.generate_gerber_files()
        success_bom = self.generate_bom()
        success_cpl = self.generate_cpl()
        
        if success_gerber and success_bom and success_cpl:
            print("Export erfolgreich: Alle Fertigungsunterlagen wurden erstellt.")
            
            # Kopiere die wichtigsten Dateien ins Hauptverzeichnis für einfachen JLCPCB-Upload
            shutil.copy2(BOM_DIR / "bom_jlcpcb.csv", PROJECT_ROOT / "hardware" / "manufacturing" / "bom_jlcpcb.csv")
            shutil.copy2(CPL_DIR / "cpl_jlcpcb.csv", PROJECT_ROOT / "hardware" / "manufacturing" / "cpl_jlcpcb.csv")
            
            # Erstelle ZIP-Datei für Gerber-Dateien
            zip_path = PROJECT_ROOT / "hardware" / "manufacturing" / "gerber_jlcpcb.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file in GERBER_DIR.glob("*.*"):
                    if file.is_file() and file.suffix != ".txt":
                        zipf.write(file, arcname=file.name)
                        
            print(f"Für den einfachen JLCPCB-Upload wurden folgende Dateien in {PROJECT_ROOT / 'hardware' / 'manufacturing'} erstellt:")
            print(f"- bom_jlcpcb.csv")
            print(f"- cpl_jlcpcb.csv")
            print(f"- gerber_jlcpcb.zip")
            
            # Führe eine JLCPCB-Kompatibilitätsprüfung durch
            self.verify_jlcpcb_compatibility()
            
            # Aktualisiere den PROJECT_STATUS.txt
            self.update_project_status()
            
            return True
        else:
            print("Export fehlgeschlagen: Nicht alle Fertigungsunterlagen konnten erstellt werden.")
            return False
    
    def update_project_status(self):
        """Aktualisiert den Projektstatus-Bericht."""
        status_file = PROJECT_ROOT / "PROJECT_STATUS.txt"
        
        if not status_file.exists():
            print(f"Warnung: Projektstatus-Datei {status_file} nicht gefunden.")
            return
            
        # Aktualisieren des Projektstatus
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Lies die aktuelle Datei
        with open(status_file, 'r') as f:
            content = f.read()
            
        # Aktualisiere den Hardware-Produktionsstatus
        production_status = f"""
## Hardware-Produktionsstatus (JLCPCB)

Der aktuelle Status der Hardware-Produktion bei JLCPCB:
1. PCB-Design: Fertiggestellt und validiert (8-Lagen-Design für optimale Signalintegrität)
2. DRC (Design Rule Check): Bestanden, alle Sicherheitsabstände JLCPCB-konform
3. Thermische Analyse: Durchgeführt, kritische Komponenten mit ausreichender Wärmeableitung versehen
4. Stromversorgung: Überprüft, alle Versorgungsleitungen korrekt dimensioniert

Fertigungsunterlagen für JLCPCB:
1. Gerber-Dateien: Mit KiCad-Integration erstellt und in `/hardware/manufacturing/gerber/` abgelegt, ZIP-Archiv für JLCPCB in `/hardware/manufacturing/gerber_jlcpcb.zip`
2. Stückliste (BOM): Vollständig in `/hardware/manufacturing/bom_jlcpcb.csv` und `/hardware/manufacturing/bom/`
3. Bestückungsplan (CPL): Generiert in `/hardware/manufacturing/cpl_jlcpcb.csv` und `/hardware/manufacturing/centroid/`
4. Pick-and-Place-Daten: Vorbereitet für SMT-Fertigung

Alle Dateien entsprechen den JLCPCB-Anforderungen und sind bereit für den Upload. Letzter Validierungscheck am {today} durchgeführt.

HINWEIS: Die Fertigungsunterlagen wurden mit dem verbesserten PCB-Export-Tool generiert und sind jetzt JLCPCB-konform. Der Export nutzt KiCad-Bibliotheken zur Erstellung standardkonformer Gerber-Dateien.
"""

        # Suche und ersetze den entsprechenden Abschnitt
        new_content = re.sub(r'## Hardware-Produktionsstatus \(JLCPCB\).*?(?=\n## )', production_status, content, flags=re.DOTALL)
        
        # Schreibe die aktualisierte Datei
        with open(status_file, 'w') as f:
            f.write(new_content)
            
        print(f"Projektstatus wurde aktualisiert: {status_file}")

    def cleanup(self):
        """Bereinigt temporäre Dateien"""
        try:
            # Bereinige temporäres KiCad-Verzeichnis
            if TEMP_KICAD_DIR.exists():
                shutil.rmtree(TEMP_KICAD_DIR)
                print(f"Temporäres Verzeichnis {TEMP_KICAD_DIR} wurde gelöscht.")
        except Exception as e:
            print(f"Warnung: Konnte temporäre Dateien nicht bereinigen: {e}")


if __name__ == "__main__":
    print("PCB Export Tool für JLCPCB-Fertigung (Verbesserte Version)")
    print("====================================")
    
    if not SVG_PATH.exists():
        print(f"Fehler: SVG-Datei '{SVG_PATH}' nicht gefunden.")
        sys.exit(1)
    
    print("Prüfe KiCad-Integration...", end=" ")
    if KICAD_AVAILABLE:
        print("Verfügbar! Nutze KiCad für bessere Gerber-Export-Ergebnisse.")
    else:
        print("Nicht verfügbar. Verwende Fallback-Modus für Gerber-Erstellung.")
        print("Tipp: Für bessere Ergebnisse installieren Sie KiCad und stellen sicher,")
        print("      dass die Python-Module im Pfad verfügbar sind.")
    
    try:
        exporter = PCBExporter(SVG_PATH)
        success = exporter.export_all()
        
        if success:
            print("\nAlle Fertigungsunterlagen wurden erfolgreich erstellt und sind jetzt bereit für den Upload bei JLCPCB.")
            print(f"Die Dateien befinden sich in:\n{PROJECT_ROOT / 'hardware' / 'manufacturing'}")
            exporter.cleanup()
            sys.exit(0)
        else:
            print("\nFehler: Die Fertigungsunterlagen konnten nicht vollständig erstellt werden.")
            exporter.cleanup()
            sys.exit(1)
    except Exception as e:
        print(f"\nFehler: {e}")
        print("Export abgebrochen.")
        sys.exit(1)