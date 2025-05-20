#!/usr/bin/env python3
# hardware/manufacturing/pcb_export.py
"""
PCB Export Tool für verschiedene Fertigungsanbieter
Unterstützt mehrere PCB-Fertigungsdienste und nutzt bestehende EDA-Tools.

Funktionen:
- Import von SVG-Layouts als Referenz für Komponentenpositionen
- Export von Fertigungsdateien für mehrere PCB-Hersteller (JLCPCB, PCBWay, OSH Park, Eurocircuits)
- Integration mit KiCad, Eagle oder direkte Nutzung von FlatCAM
- Erstellung von BOM und CPL im herstellerspezifischen Format
- Automatische Preisabfrage und Angebotsvergleich über Hersteller-APIs

Besser nutzbar als eigenständiges Tool oder als Ergänzung im KiCad-Workflow.
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
import argparse
import webbrowser
import requests
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define API client interface
class ApiClientInterface:
    """Interface for API client functionality to handle import issues."""
    
    def __init__(self):
        self.available = False
        
        # Try to import the module
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import pcb_api_clients
            self._get_pcb_client = pcb_api_clients.get_pcb_client
            self._format_price = pcb_api_clients.format_price
            self._compare_quotes = pcb_api_clients.compare_quotes
            self.available = True
            logger.info("PCB API clients module imported successfully.")
        except ImportError as e:
            logger.warning(f"PCB API clients module could not be imported: {e}")
            logger.warning("API quote functionality will not be available.")
    
    def get_pcb_client(self, manufacturer: str) -> Any:
        """Get a PCB API client for the specified manufacturer."""
        if not self.available:
            logger.error(f"PCB API clients module not available. Cannot get client for {manufacturer}.")
            return None
        return self._get_pcb_client(manufacturer)
    
    def format_price(self, price: float, currency: str = "USD") -> str:
        """Format a price with currency."""
        if not self.available:
            return f"{price} {currency}"
        return self._format_price(price, currency)
    
    def compare_quotes(self, quotes_list: List[Dict]) -> Dict:
        """Compare multiple quotes and return the best options."""
        if not self.available:
            logger.error("PCB API clients module not available. Cannot compare quotes.")
            return {}
        return self._compare_quotes(quotes_list)

# Initialize the API client interface
api_client = ApiClientInterface()

# Konfiguration für das Logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pcb_export")

# Projekt-Verzeichnisse
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SVG_PATH = PROJECT_ROOT / "docs" / "pcb-layout-updated.svg"
MANUFACTURING_DIR = PROJECT_ROOT / "hardware" / "manufacturing"
OUTPUT_DIR = MANUFACTURING_DIR / "output"

# Stellt sicher, dass das Ausgabeverzeichnis existiert
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Unterstützte PCB-Hersteller
PCB_MANUFACTURERS = {
    "jlcpcb": {
        "name": "JLCPCB",
        "url": "https://jlcpcb.com/",
        "quote_calculator_url": "https://cart.jlcpcb.com/quote",
        "api_docs_url": "https://jlcpcb.com/api/",
        "bom_format": ["Comment", "Designator", "Footprint", "Quantity"], # Standard KiCad BOM + Quantity
        "cpl_format": ["Designator", "Val", "Package", "Mid X", "Mid Y", "Rotation", "Layer"], # Standard KiCad CPL
        "gerber_naming": {"top_copper": "gtl", "bottom_copper": "gbl", "drill": "txt"}, # Common JLCPCB names
        "api_available": True,
        "description": "Beliebter chinesischer Hersteller für Prototypen und Kleinserien.",
        "assembly_service_url": "https://jlcpcb.com/assembly",
        "technical_data_url": "https://jlcpcb.com/capabilities/pcb-capabilities",
        "typical_quote_parameters": {
            "pcb": [
                "Platinen-Typ: (PCB / PCBA / Schablone)",
                "Basis-Material: (FR-4 Standard, Aluminium, Rogers, etc.)",
                "Lagenanzahl: {layers} (aus Kommandozeile)",
                "Platinenabmessungen (B x H): {width_mm:.2f}mm x {height_mm:.2f}mm (aus SVG)",
                "Stückzahl: {quantity} (aus Kommandozeile)",
                "Lieferformat: (Einzelplatine / Nutzen durch JLCPCB / Nutzen durch Kunde)",
                "Leiterplattendicke: (z.B. 0.6mm, 1.0mm, 1.2mm, 1.6mm (Standard), 2.0mm)",
                "Kupferaußendicke: (z.B. 1oz (35µm Standard), 2oz (70µm))",
                "Kupferinnendicke (für >2 Lagen): (z.B. 0.5oz, 1oz)",
                "Oberfläche: (z.B. HASL bleifrei (Standard), ENIG (Gold), OSP)",
                "Lötstopplackfarbe: (z.B. Grün, Rot, Gelb, Blau, Weiß, Schwarz matt)",
                "Bestückungsdruckfarbe: (z.B. Weiß, Schwarz)",
                "Goldfinger: (Ja/Nein)",
                "Castellated Holes (Randmetallisierung): (Ja/Nein)",
                "Entfernen der Auftragsnummer: (Ja/Nein)",
                "Testmethode: (AOI Standard, Fliegende Sonde Test)",
                "Papier zwischen Platinen: (Ja/Nein)"
            ],
            "assembly": [
                "Bestückungsseite(n): (Oben / Unten / Beide)",
                "Anzahl eindeutiger Bauteile: {unique_components} (aus BOM)",
                "Anzahl gesamter SMD-Bauteile: {total_components} (aus BOM, ggf. SMD/THT präzisieren)",
                "Anzahl THT-Bauteile: (Manuell angeben, falls vorhanden)",
                "Schablone (Stencil) benötigt: (Ja/Nein, Typ)",
                "Bestätigung der Bauteilplatzierung: (Ja/Nein)"
            ]
        }
    },
    "pcbway": {
        "name": "PCBWay",
        "url": "https://www.pcbway.com/",
        "quote_calculator_url": "https://www.pcbway.com/pcb-quote.html",
        "api_docs_url": "https://www.pcbway.com/api.html",
        "bom_format": ["Designator", "Quantity", "Manufacturer", "Manufacturer Part Number", "Type", "Value", "Package", "Description"],
        "cpl_format": ["Designator", "X (mm)", "Y (mm)", "Side", "Rotation"],
        "gerber_naming": {"top_copper": "gtl", "bottom_copper": "gbl", "drill": "txt"},
        "api_available": True,
        "description": "Umfassende Fertigungsdienstleistungen mit Prototypen und Serienproduktion",
        "assembly_service_url": "https://www.pcbway.com/assembly.html",
        "technical_data_url": "https://www.pcbway.com/capabilities.html",
        "typical_quote_parameters": {
            "pcb": [
                "Platinengröße (B x H): {width_mm:.2f}mm x {height_mm:.2f}mm (aus SVG)",
                "Stückzahl: {quantity} (aus Kommandozeile)",
                "Lagenanzahl: {layers} (aus Kommandozeile)",
                "Material: (FR-4 Standard, Aluminium, Rogers, etc.)",
                "Leiterplattendicke (FR4): (z.B. 0.8mm - 2.0mm (1.6mm Standard))",
                "Min. Leiterbahn/Abstand: (z.B. 3.5/3.5mil, 4/4mil, 5/5mil (Standard))",
                "Min. Bohrung: (z.B. 0.2mm, 0.25mm, 0.3mm (Standard))",
                "Lötstopplackfarbe: (z.B. Grün, Rot, Gelb, Blau, Weiß, Schwarz)",
                "Bestückungsdruckfarbe: (z.B. Weiß, Schwarz)",
                "Oberfläche: (HASL bleifrei (Standard), ENIG, Immersion Silber/Zinn, OSP)",
                "Kupferaußendicke: (1oz (Standard), 2oz)",
                "Gold Fingers: (Ja/Nein)",
                "Castellated Holes: (Ja/Nein)",
                "Impedanzkontrolle: (Ja/Nein)",
                "Panelisierung: (Einzeln / Panel by PCBWay / Panel by Customer)",
                "Flying Probe Test: (Standard)"
            ],
            "assembly": [
                "Bestückungsseite(n): (Oben / Unten / Beide)",
                "Anzahl eindeutiger Bauteile: {unique_components} (aus BOM)",
                "Anzahl SMD-Bauteile gesamt: {total_components} (aus BOM, ggf. SMD/THT präzisieren)",
                "Anzahl THT-Bauteile gesamt: (Manuell angeben, falls vorhanden)",
                "Schablone (Stencil): (Ja/Nein, Typ)"
            ]
        }
    },
    "oshpark": {
        "name": "OSH Park",
        "url": "https://oshpark.com/",
        "quote_calculator_url": "https://oshpark.com/pricing/", 
        "api_docs_url": None,
        "bom_format": ["Reference", "Quantity", "Value", "Footprint", "DNP"],
        "cpl_format": ["Ref", "Val", "Package", "PosX", "PosY", "Rot", "Side"],
        "gerber_naming": {"top_copper": "GTL", "bottom_copper": "GBL", "drill": "XLN"},
        "api_available": False,
        "description": "US-basierter Hersteller mit hochwertigen PCBs in charakteristischer violetter Farbe",
        "technical_data_url": "https://docs.oshpark.com/services/",
        "assembly_service_url": None, # OSH Park bietet keinen eigenen Bestückungsservice an
        "typical_quote_parameters": {
            "pcb": [
                "Lagenanzahl: (2 Lagen Standard / 4 Lagen)",
                "Platinenabmessungen (B x H): {width_mm:.2f}mm x {height_mm:.2f}mm (aus SVG, wird in inch umgerechnet)",
                "Stückzahl: {quantity} (meist in 3er-Batches für 2-Lagen)",
                "Service: (Standard (USA Fertigung) / Super Swift (USA) / After Dark (EU))",
                "Kupferdicke: (Standard 1oz/ft² (35µm))",
                "Material: (FR-4 für 2-Lagen, FR408 für 4-Lagen)",
                "Finish: (ENIG für alle Platinen)",
                "Hinweis: OSH Park hat feste Spezifikationen pro Service. Der Preis wird pro Quadratzoll berechnet."
            ],
            "assembly": [] # Keine Assembly-Parameter, da kein Service angeboten
        }
    },
    "eurocircuits": {
        "name": "Eurocircuits",
        "url": "https://www.eurocircuits.com/",
        "quote_calculator_url": "https://www.eurocircuits.com/pcb-calculator/",
        "api_docs_url": "https://www.eurocircuits.com/discover-our-apis-for-pcb-services/",
        "bom_format": ["Component", "Values", "Package", "Layer", "Rotation", "Populated"],
        "cpl_format": ["Component", "X", "Y", "Layer", "Rotation"],
        "gerber_naming": {"top_copper": "cmp", "bottom_copper": "sol", "drill": "drd"},
        "api_available": True,
        "description": "Europäischer Hersteller mit Fokus auf Qualität und Zuverlässigkeit",
        "assembly_service_url": "https://www.eurocircuits.com/pcb-assembly/",
        "technical_data_url": "https://www.eurocircuits.com/pcb-manufacturing-technology/",
        "typical_quote_parameters": {
            "pcb": [
                "Service: (PCB Proto / Standard Pool / etc.)",
                "Lagenanzahl: {layers} (aus Kommandozeile)",
                "Platinenabmessungen (B x H): {width_mm:.2f}mm x {height_mm:.2f}mm (aus SVG)",
                "Stückzahl: {quantity} (aus Kommandozeile)",
                "Lieferformat: (Einzelplatine / E- όχι Nutzen)",
                "Material: (FR-4 Standard, etc.)",
                "Leiterplattendicke: (z.B. 1.0mm, 1.6mm (Standard))",
                "Kupferlagenaufbau: (Standard / Spezifisch)",
                "Oberfläche: (Chem. Ni/Au (ENIG Standard), bleifrei HAL, etc.)",
                "Lötstoppmaske Farbe: (Grün Standard, andere Farben)",
                "Bestückungsdruck Farbe: (Weiß Standard, andere Farben)",
                "E-Test: (Standard)",
                "Kontur: (Fräsen Standard)"
            ],
            "assembly": [
                "Bestückungsseite(n): (Oben / Unten / Beide)",
                "Anzahl eindeutiger Bauteile: {unique_components} (aus BOM)",
                "Gesamtzahl Lötstellen (SMD/THT): (Manuell oder aus detaillierter BOM)",
                "Schablone (Stencil): (Ja/Nein, Edelstahl)"
            ]
        }
    }
}

# Unterstützte EDA-Tools für die Integration
EDA_TOOLS = {
    "kicad": {
        "name": "KiCad",
        "check_command": ["kicad-cli", "--version"],
        "export_gerber_command": ["kicad-cli", "pcb", "export", "gerbers"],
        "export_bom_command": ["kicad-cli", "pcb", "export", "bom"],
        "export_cpl_command": ["kicad-cli", "pcb", "export", "pos"],
        "description": "Open-Source-EDA-Tool mit umfassenden Funktionen",
        "website": "https://www.kicad.org/"
    },
    "flatcam": {
        "name": "FlatCAM",
        "check_command": ["flatcam", "--version"],
        "export_command": ["flatcam", "--run_script"],
        "description": "Open-Source-Tool zur Erzeugung von Gerber-Dateien aus verschiedenen Eingabeformaten",
        "website": "http://flatcam.org/"
    },
    "gerbv": {
        "name": "Gerbv",
        "check_command": ["gerbv", "--version"],
        "export_command": ["gerbv", "-x", "png"],
        "description": "Open-Source-Gerber-Viewer mit Export-Funktionen",
        "website": "http://gerbv.geda-project.org/"
    }
}

class PCBExportTool:
    """
    Werkzeug zum Export von PCB-Fertigungsdaten für verschiedene Hersteller.
    Bietet Flexibilität bei der Integration mit bestehenden EDA-Tools.
    """
    def __init__(self, svg_path: Path = SVG_PATH, 
                 output_dir: Path = OUTPUT_DIR,
                 manufacturer: str = "jlcpcb"):
        """
        Initialisiert das PCB-Export-Tool.
        
        Args:
            svg_path: Pfad zur SVG-Layout-Datei
            output_dir: Verzeichnis für die Ausgabedateien
            manufacturer: PCB-Hersteller (Standard: jlcpcb)
        """
        self.svg_path = svg_path
        self.output_dir = output_dir
        
        # Setze Hersteller
        if manufacturer.lower() not in PCB_MANUFACTURERS:
            logger.warning(f"Hersteller '{manufacturer}' nicht unterstützt. Verwende JLCPCB als Standard.")
            manufacturer = "jlcpcb"
        
        self.manufacturer = manufacturer.lower()
        self.manufacturer_info = PCB_MANUFACTURERS[self.manufacturer]
        
        # Erstelle herstellerspezifische Ausgabeverzeichnisse
        self.mfg_dir = output_dir / self.manufacturer
        self.gerber_dir = self.mfg_dir / "gerber"
        self.bom_dir = self.mfg_dir / "bom"
        self.cpl_dir = self.mfg_dir / "cpl"
        
        for directory in [self.mfg_dir, self.gerber_dir, self.bom_dir, self.cpl_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialisiere Komponentendaten
        self.components = []
        self.svg_data = None
        self.board_dimensions = {"width": 0, "height": 0} # Hinzugefügt
        self.bom_summary = None # Hinzugefügt für Preis-Guidance
        
        # Erkenne verfügbare EDA-Tools
        self.available_tools = self._detect_available_tools()

    def _detect_available_tools(self) -> Dict[str, bool]:
        """Erkennt, welche EDA-Tools auf dem System verfügbar sind."""
        available = {}
        
        for tool_id, tool_info in EDA_TOOLS.items():
            try:
                # Prüfe, ob das Tool verfügbar ist
                result = subprocess.run(
                    tool_info["check_command"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    timeout=5
                )
                available[tool_id] = result.returncode == 0
                
                if available[tool_id]:
                    logger.info(f"{tool_info['name']} gefunden und verfügbar.")
                else:
                    logger.info(f"{tool_info['name']} nicht verfügbar (Exit-Code: {result.returncode}).")
            
            except (subprocess.SubprocessError, FileNotFoundError):
                available[tool_id] = False
                logger.info(f"{tool_info['name']} nicht installiert oder nicht im PATH.")
        
        return available

    def parse_svg(self) -> bool:
        """
        Parst die SVG-Datei und extrahiert Komponenteninformationen.
        Einfachere und robustere Implementation als zuvor.
        Ermittelt auch die Platinenabmessungen.
        """
        if not self.svg_path.exists():
            logger.error(f"SVG-Datei nicht gefunden: {self.svg_path}")
            return False
            
        try:
            tree = ET.parse(self.svg_path)
            root = tree.getroot()
            self.svg_data = tree
            
            logger.info(f"SVG-Datei '{self.svg_path}' erfolgreich geladen.")
            
            # Extrahiere den Namespace aus dem Root-Element (falls vorhanden)
            ns = {"svg": "http://www.w3.org/2000/svg"}
            
            components = []
            component_counter = {}
            
            min_x, max_x = float('inf'), float('-inf')
            min_y, max_y = float('inf'), float('-inf')

            # Finde alle Rechtecke (potenzielle Komponenten)
            for rect in root.findall(".//svg:rect", ns):
                x = float(rect.get("x", "0"))
                y = float(rect.get("y", "0"))
                width = float(rect.get("width", "0"))
                height = float(rect.get("height", "0"))
                
                # Update board dimensions
                min_x = min(min_x, x)
                max_x = max(max_x, x + width)
                min_y = min(min_y, y)
                max_y = max(max_y, y + height)

                # Suche nach Text in der Nähe des Rechtecks
                for text in root.findall(".//svg:text", ns):
                    text_x = float(text.get("x", "0"))
                    text_y = float(text.get("y", "0"))
                    
                    # Prüfe, ob der Text innerhalb oder nahe beim Rechteck ist
                    if (x-5 <= text_x <= x + width+5 and y-5 <= text_y <= y + height+5):
                        component_name = text.text if text.text else "Unknown"
                        
                        # Zähle Komponenten gleichen Typs
                        if component_name not in component_counter:
                            component_counter[component_name] = 0
                        component_counter[component_name] += 1
                        
                        # Erstelle ID aus dem Namen (z.B. RP2040_1)
                        base_name = ''.join(c for c in component_name if c.isalnum() or c == '-' or c == '_')
                        if not base_name:
                            base_name = "COMP"
                        component_id = f"{base_name}_{component_counter[component_name]}"
                        
                        # Gerundete Position (Mittelpunkt der Komponente)
                        components.append({
                            "id": component_id,
                            "name": component_name,
                            "x": round(x + width/2, 2),  # Mittelpunkt X
                            "y": round(y + height/2, 2),  # Mittelpunkt Y
                            "width": round(width, 2),
                            "height": round(height, 2),
                            "rotation": 0.0  # Standard-Rotation
                        })
                        break
            
            self.components = components
            logger.info(f"Extrahierte {len(self.components)} Komponenten aus der SVG-Datei.")
            
            if self.components: # Nur wenn Komponenten gefunden wurden
                self.board_dimensions["width"] = round(max_x - min_x, 2)
                self.board_dimensions["height"] = round(max_y - min_y, 2)
                logger.info(f"Ermittelte Platinengröße: {self.board_dimensions['width']}mm x {self.board_dimensions['height']}mm")
            else:
                logger.warning("Keine Komponenten gefunden, Platinengröße konnte nicht ermittelt werden.")

            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Parsen der SVG-Datei: {e}")
            return False

    def export_to_kicad_pcb(self, output_path: Optional[Path] = None) -> bool:
        """
        Exportiert die Komponenten in eine KiCad PCB-Datei.
        Nutzt ein einfacheres Format ohne komplexe KiCad-Bibliotheksintegrationen.
        
        Args:
            output_path: Pfad für die KiCad PCB-Datei (optional)
        """
        if not self.components:
            logger.error("Keine Komponenten zum Exportieren vorhanden.")
            return False
            
        if output_path is None:
            output_path = self.mfg_dir / "converted_design.kicad_pcb"
            
        try:
            # Erstelle eine Basis-KiCad-PCB-Datei
            with open(output_path, 'w') as f:
                # KiCad-PCB-Header
                f.write("""(kicad_pcb (version 20211014) (generator pcb_export.py)
  (general
    (thickness 1.6)
  )
  (paper "A4")
  (title_block
    (title "RP2040 Pizza Detection System")
    (date "%s")
    (rev "1.0")
  )
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
    (34 "B.Paste" user)
    (35 "F.Paste" user)
    (36 "B.SilkS" user "B.Silkscreen")
    (37 "F.SilkS" user "F.Silkscreen")
    (38 "B.Mask" user)
    (39 "F.Mask" user)
    (44 "Edge.Cuts" user)
  )
""" % (datetime.now().strftime("%Y-%m-%d")))
                
                # Füge Komponenten hinzu
                for comp in self.components:
                    f.write(f"""  (footprint "Converted:Generic" (layer "F.Cu")
    (at {comp["x"]} {comp["y"]})
    (descr "{comp["name"]}")
    (attr smd)
    (fp_text reference "{comp["id"]}" (at 0 -2) (layer "F.SilkS") (effects (font (size 1 1) (thickness 0.15))))
    (fp_text value "{comp["name"]}" (at 0 2) (layer "F.Fab") (effects (font (size 1 1) (thickness 0.15))))
    (fp_rect (start -{comp["width"]/2} -{comp["height"]/2}) (end {comp["width"]/2} {comp["height"]/2}) (layer "F.SilkS") (width 0.1))
  )
""")
                
                # Ende der KiCad-PCB-Datei
                f.write(")")
                
            logger.info(f"KiCad PCB-Datei erfolgreich exportiert: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Exportieren als KiCad PCB: {e}")
            return False

    def generate_bom(self) -> bool:
        """
        Erstellt eine BOM (Stückliste) im herstellerspezifischen Format.
        Vereinfachte Version, die keine detaillierten Komponenteninformationen erfordert.
        Speichert auch eine Zusammenfassung für die Preisabschätzung.
        """
        if not self.components:
            logger.error("Keine Komponenten für BOM vorhanden.")
            return False
            
        try:
            # Erzeuge eine zusammengefasste BOM (gruppiere gleiche Komponenten)
            bom_entries = {}
            
            for comp in self.components:
                name = comp["name"]
                component_id = comp["id"]
                
                # Bestimme Komponententyp aus dem Namen
                component_type = "Unknown"
                for key in ["RP2040", "FLASH", "XTAL", "DRIVER", "BUZZER", "USB", "BUTTON", "LED", "SENSOR"]:
                    if key.lower() in name.lower():
                        component_type = key
                        break
                
                # Standardwerte für alle Komponenten
                component_info = {
                    "Value": name,
                    "Package": "SMD", # Annahme: Alle sind SMD für diese einfache BOM
                    "Type": component_type,
                    "DNP": "No"
                }
                
                # Füge herstellerspezifische Felder hinzu
                if self.manufacturer == "jlcpcb":
                    component_info.update({
                        "Manufacturer": "Generic",
                        "Part Number": f"GENERIC-{component_type}",
                        "Supplier": "LCSC",
                        "LCSC Part Number": "N/A" # Muss manuell ergänzt werden für echte Bestellung
                    })
                
                # Gruppiere nach Komponentenname
                if name not in bom_entries:
                    bom_entries[name] = {
                        "Designator": component_id,
                        "Quantity": 1,
                        **component_info
                    }
                else:
                    bom_entries[name]["Quantity"] += 1
                    bom_entries[name]["Designator"] += f", {component_id}"
            
            # Speichere BOM-Zusammenfassung
            self.bom_summary = {
                "unique_components": len(bom_entries),
                "total_components": sum(item["Quantity"] for item in bom_entries.values())
            }
            logger.info(f"BOM Zusammenfassung: {self.bom_summary['unique_components']} eindeutige Bauteile, {self.bom_summary['total_components']} Bauteile gesamt.")

            # Schreibe BOM-Datei im herstellerspezifischen Format
            bom_fields = self.manufacturer_info["bom_format"]
            bom_file = self.bom_dir / f"bom_{self.manufacturer}.csv"
            
            with open(bom_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=bom_fields)
                writer.writeheader()
                
                for entry in bom_entries.values():
                    # Konvertiere in herstellerspezifisches Format
                    formatted_entry = {}
                    for field in bom_fields:
                        if field in entry:
                            formatted_entry[field] = entry[field]
                        elif field in ["Reference", "Ref", "Comment"]: # OSH Park, JLCPCB Kompatibilität
                            formatted_entry[field] = entry.get("Designator", "")
                        elif field in ["Footprint"]: 
                            formatted_entry[field] = entry.get("Package", "")
                        elif field in ["Val", "Values"]:
                            formatted_entry[field] = entry.get("Value", "")
                        elif field in ["Component"]: # Eurocircuits Kompatibilität
                            formatted_entry[field] = entry.get("Designator", "")
                        elif field in ["Populated"]: # Eurocircuits Kompatibilität
                            formatted_entry[field] = "Yes" if entry.get("DNP", "No").lower() == "no" else "No"
                        elif field == "LCSC Part #" and self.manufacturer == "jlcpcb": # Spezifisch für JLCPCB
                             formatted_entry[field] = entry.get("LCSC Part Number", "N/A")
                        elif field == "Quantity":
                            formatted_entry[field] = entry.get("Quantity", 0)
                        else:
                            formatted_entry[field] = "N/A" # Fallback
                    
                    writer.writerow(formatted_entry)
            
            # Kopiere BOM-Datei für einfachen Zugriff
            shutil.copy2(bom_file, self.mfg_dir / f"bom_{self.manufacturer}.csv")
            
            # Erstelle und speichere BOM-Zusammenfassung
            unique_parts_count = len(bom_entries)
            total_parts_count = sum(entry['Quantity'] for entry in bom_entries.values())
            # Annahme: Aktuell keine explizite THT-Erkennung in der BOM-Logik
            tht_parts_count = 0 # TODO: Erweitern, falls THT-Komponenten identifiziert werden können
            
            self.bom_summary = {
                "unique_parts": unique_parts_count,
                "total_parts": total_parts_count,
                "tht_parts": tht_parts_count 
            }
            logger.info(f"BOM-Zusammenfassung aktualisiert: {self.bom_summary}")
            
            logger.info(f"BOM erfolgreich für {self.manufacturer_info['name']} erstellt: {bom_file}")
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei der BOM-Erstellung: {e}")
            return False

    def _get_bom_summary(self) -> Dict[str, int]:
        """
        Stellt sicher, dass die BOM-Zusammenfassung verfügbar ist und gibt sie zurück.
        Generiert die BOM, falls noch nicht geschehen oder self.bom_summary leer ist.
        Initialisiert self.bom_summary, falls es nicht existiert.
        """
        if not hasattr(self, 'bom_summary') or not self.bom_summary:
            logger.info("BOM-Zusammenfassung nicht initialisiert oder leer, versuche BOM zu generieren...")
            if not self.components: # Komponenten werden für BOM benötigt
                logger.info("Keine Komponenten geparst, versuche SVG zu parsen...")
                if not self.parse_svg():
                    logger.error("SVG konnte nicht geparst werden. BOM-Zusammenfassung nicht möglich.")
                    # Initialisiere mit Standardwerten, falls parse_svg fehlschlägt
                    self.bom_summary = {"unique_parts": 0, "total_parts": 0, "tht_parts": 0}
                    return self.bom_summary
            
            if not self.generate_bom(): # generate_bom sollte self.bom_summary füllen
                logger.error("BOM konnte nicht generiert werden. BOM-Zusammenfassung nicht möglich.")
                # Initialisiere mit Standardwerten, falls generate_bom fehlschlägt
                self.bom_summary = {"unique_parts": 0, "total_parts": 0, "tht_parts": 0}
                return self.bom_summary
        
        # Stelle sicher, dass die erwarteten Schlüssel vorhanden sind, mit Standardwerten
        # Dies ist nützlich, falls bom_summary zwar existiert, aber unvollständig ist.
        summary = {
            "unique_parts": self.bom_summary.get('unique_parts', 0),
            "total_parts": self.bom_summary.get('total_parts', 0),
            "tht_parts": self.bom_summary.get('tht_parts', 0) 
        }
        # Aktualisiere self.bom_summary, um Konsistenz sicherzustellen
        self.bom_summary = summary
        return summary

    def generate_cpl(self) -> bool:
        """
        Erstellt eine CPL (Bestückungsplan) im herstellerspezifischen Format.
        """
        if not self.components:
            logger.error("Keine Komponenten für CPL vorhanden.")
            return False
            
        try:
            # Schreibe CPL-Datei (Bestückungsplan)
            cpl_fields = self.manufacturer_info["cpl_format"]
            cpl_file = self.cpl_dir / f"cpl_{self.manufacturer}.csv"
            
            with open(cpl_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=cpl_fields)
                writer.writeheader()
                
                for comp in self.components:
                    # Konvertiere in herstellerspezifisches Format
                    cpl_entry = {}
                    
                    for field in cpl_fields:
                        if field in ["Designator", "Ref", "Reference", "Component"]:
                            cpl_entry[field] = comp["id"]
                        elif field in ["Mid X", "X", "X (mm)", "PosX"]:
                            cpl_entry[field] = f"{comp['x']:.3f}mm"
                        elif field in ["Mid Y", "Y", "Y (mm)", "PosY"]:
                            cpl_entry[field] = f"{comp['y']:.3f}mm"
                        elif field in ["Layer", "Side"]:
                            cpl_entry[field] = "top"  # Standard: Oberseite
                        elif field in ["Rotation", "Rot"]:
                            cpl_entry[field] = f"{comp.get('rotation', 0):.1f}"
                        elif field in ["Val", "Value"]:
                            cpl_entry[field] = comp["name"]
                        elif field in ["Package"]:
                            cpl_entry[field] = "SMD"  # Standard-Package
                        else:
                            cpl_entry[field] = ""
                    
                    writer.writerow(cpl_entry)
            
            # Kopiere CPL-Datei für einfachen Zugriff
            shutil.copy2(cpl_file, self.mfg_dir / f"cpl_{self.manufacturer}.csv")
            
            logger.info(f"CPL erfolgreich für {self.manufacturer_info['name']} erstellt: {cpl_file}")
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei der CPL-Erstellung: {e}")
            return False

    def provide_price_estimation_guidance(self, quantity: int = 10, layers: int = 2) -> None:
        """
        Stellt detaillierte Informationen und Links für eine manuelle Preisabschätzung bereit,
        angepasst an die Webseiten der Hersteller.
        
        Args:
            quantity: Die gewünschte Stückzahl für die Preisanfrage.
            layers: Die Anzahl der Kupferlagen (Standard: 2).
        """
        if not self.components:
            if not self.parse_svg(): # Versuche SVG zu parsen, falls noch nicht geschehen
                logger.error("SVG konnte nicht geparst werden. Preisabschätzung nicht möglich.")
                return
        
        bom_info = self._get_bom_summary() # Stellt sicher, dass BOM-Daten aktuell sind

        board_width_mm = self.board_dimensions.get("width", 0)
        board_height_mm = self.board_dimensions.get("height", 0)
        board_width_str = f"{board_width_mm:.2f} mm" if board_width_mm > 0 else "Unbekannt (aus SVG/Design ableiten)"
        board_height_str = f"{board_height_mm:.2f} mm" if board_height_mm > 0 else "Unbekannt (aus SVG/Design ableiten)"

        print("\n" + "="*80)
        print("=== LEITFADEN ZUR PREISABSCHÄTZUNG FÜR PCB-FERTIGUNG & BESTÜCKUNG ===")
        print("="*80)
        print("Dieser Leitfaden hilft Ihnen, die notwendigen Parameter für die Online-Kalkulatoren der Hersteller zu sammeln.")
        print("Die Genauigkeit der Schätzung hängt von den eingegebenen Daten und den spezifischen Optionen des Herstellers ab.")
        
        print("\n--- Basisdaten Ihres Projekts ---")
        print(f"  - SVG-Datei: {self.svg_path.name}")
        print(f"  - Geschätzte Platinengröße (B x H): {board_width_str} x {board_height_str}")
        print(f"  - Gewünschte Stückzahl: {quantity}")
        print(f"  - Anzahl Kupferlagen: {layers}")
        if bom_info["total_parts"] > 0:
            print(f"  - Einzigartige Bauteile (SMD): {bom_info['unique_parts']}")
            print(f"  - Gesamtzahl Bauteil-Platzierungen (SMD): {bom_info['total_parts']}")
            if bom_info['tht_parts'] > 0: # Falls THT-Zählung implementiert wird
                 print(f"  - Anzahl THT-Bauteile: {bom_info['tht_parts']}")
        else:
            print("  - BOM-Daten: Noch nicht verfügbar oder keine Bauteile für Bestückung.")

        format_map = {
            "width_mm": board_width_mm,
            "height_mm": board_height_mm,
            "quantity": quantity,
            "layers": layers,
            "unique_components": bom_info['unique_parts'],
            "total_components": bom_info['total_parts'],
            "tht_components": bom_info.get('tht_parts', 0)
        }

        for mfg_id, mfg_info in PCB_MANUFACTURERS.items():
            print("\n" + "-"*60)
            print(f"Hersteller: {mfg_info['name']}")
            print("-"*60)
            
            if mfg_info.get("quote_calculator_url"):
                print(f"  Online-Kalkulator: {mfg_info['quote_calculator_url']}")
            else:
                print(f"  Website (für Kalkulator-Suche): {mfg_info['url']}")

            print("\n  Empfohlene Parameter für die Eingabe:")
            
            # 1. Parameter für Leiterplattenfertigung
            print("    --- 1. Parameter für Leiterplattenfertigung ---")
            printed_pcb_params = set() 

            for param_template in mfg_info.get("typical_quote_parameters", {}).get("pcb", []):
                # Standardwerte setzen
                final_display_name = param_template
                final_output_value = "Prüfen Sie die Optionen auf der Webseite des Herstellers."
                has_special_handling = False

                # Versuche, jeden Parameter zu formatieren mit den verfügbaren Projektdaten
                try:
                    formatted_param = param_template.format_map(format_map)
                    
                    # Erfolgreich formatiert und anders als das Original
                    if formatted_param != param_template:
                        if ":" in formatted_param:
                            parts = formatted_param.split(":", 1)
                            final_display_name = parts[0].strip()
                            final_output_value = parts[1].strip()
                        else:
                            final_display_name = param_template
                            final_output_value = formatted_param
                    # Keine Formatierung angewandt oder nicht anders als Original
                    elif ":" in param_template:
                        final_display_name = param_template.split(":", 1)[0].strip()
                        # Versuche einen Standardwert aus dem Parameter-String zu extrahieren
                        param_value_part = param_template.split(":", 1)[1].strip()
                        default_value = self._extract_default_from_param_string(param_value_part)
                        if default_value:
                            final_output_value = default_value
                        else:
                            final_output_value = param_value_part
                except (KeyError, ValueError):
                    # Formatierung fehlgeschlagen wegen fehlender Platzhalter oder Formatierungsfehler
                    logger.debug(f"Formatierung fehlgeschlagen für '{param_template}'. Versuche Schlüsselwort-Erkennung.")
                    if ":" in param_template:
                        final_display_name = param_template.split(":", 1)[0].strip()
                        param_value_part = param_template.split(":", 1)[1].strip()
                        default_value = self._extract_default_from_param_string(param_value_part)
                        if default_value:
                            final_output_value = default_value
                        else:
                            final_output_value = param_value_part
                
                # Schlüsselwortbasierte Spezialverarbeitung für bestimmte Parameter
                param_lower_check = param_template.lower()
                
                # OSH Park-spezifische Überschreibungen für Kupferdicke
                if mfg_id == "oshpark" and "kupferdicke" in param_lower_check:
                    final_display_name = param_template.split(":", 1)[0].strip() if ":" in param_template else "Kupferdicke" 
                    final_output_value = "1 oz/ft² (35µm) (Standard bei OSH Park für alle Lagen)"
                    has_special_handling = True
                # OSH Park-spezifische Überschreibungen für Material
                elif mfg_id == "oshpark" and "material" in param_lower_check:
                    final_display_name = "Material"
                    if layers <= 2:
                        final_output_value = "FR-4 (Standard für 2-Lagen bei OSH Park)"
                    else:
                        final_output_value = "FR408 (High-Speed Material für 4+ Lagen bei OSH Park)"
                    has_special_handling = True
                    printed_pcb_params.add("material")
                # PCBWay-spezifische Überschreibungen
                elif mfg_id == "pcbway" and "kupferaußendicke" in param_lower_check:
                    final_display_name = param_template.split(":", 1)[0].strip()
                    final_output_value = "1 oz/ft² (35µm) (Standard). Für höhere Ströme ggf. 2oz (70µm) wählen."
                    has_special_handling = True
                # Kupferinnendicke-Behandlung für 2-Lagen-Designs
                elif param_lower_check == "kupferinnendicke (für >2 lagen): (z.b. 0.5oz, 1oz)" or "kupferinnendicke" in param_lower_check:
                    final_display_name = param_template.split(":", 1)[0].strip() if ":" in param_template else "Kupferinnendicke"
                    if layers <= 2:
                        final_output_value = "Nur relevant für Platinen mit mehr als 2 Lagen."
                    else:
                        final_output_value = "0.5 oz/ft² (17.5µm) or 1 oz/ft² (35µm) (Standard für Innenlagen)."
                    has_special_handling = True
                # JLCPCB-spezifische Überschreibungen
                elif mfg_id == "jlcpcb" and param_lower_check in ["kupferaußendicke: (z.b. 1oz (35µm standard), 2oz (70µm))", "kupferdicke: (standard 1oz)"]:
                    final_display_name = param_template.split(":", 1)[0].strip()
                    final_output_value = "1 oz/ft² (35µm) (Standard). Für höhere Ströme ggf. 2oz (70µm) wählen."
                    has_special_handling = True
                # Allgemeine Behandlung nach Schlüsselwörtern
                elif any(p in param_lower_check for p in ["dimension", "size", "width", "height", "größe", "platinenabmessungen"]):
                    final_display_name = param_template.split(":", 1)[0].strip() if ":" in param_template else "Platinengröße"
                    final_output_value = f"{board_width_str} x {board_height_str}"
                    printed_pcb_params.add("dimensions")
                    has_special_handling = True
                elif any(p in param_lower_check for p in ["quantity", "menge", "stückzahl"]):
                    final_display_name = param_template.split(":", 1)[0].strip() if ":" in param_template else "Stückzahl"
                    final_output_value = str(quantity)
                    printed_pcb_params.add("quantity")
                    has_special_handling = True
                elif any(p in param_lower_check for p in ["layers", "lagen", "lagenanzahl"]):
                    final_display_name = param_template.split(":", 1)[0].strip() if ":" in param_template else "Lagenanzahl"
                    final_output_value = str(layers)
                    printed_pcb_params.add("layers")
                    has_special_handling = True
                elif "basis-material" in param_lower_check or ("material" in param_lower_check and not "basis-material" in param_lower_check and mfg_id != "oshpark"):
                    # Material-Spezifikationen
                    final_display_name = param_template.split(":", 1)[0].strip() if ":" in param_template else "Material"
                    final_output_value = "FR-4 (Standard). Prüfen Sie Alternativen falls benötigt."
                    has_special_handling = True
                    printed_pcb_params.add("material")
                elif any(p in param_lower_check for p in ["leiterplattendicke", "thickness", "dicke"]):
                    final_display_name = param_template.split(":", 1)[0].strip() if ":" in param_template else "Leiterplattendicke"
                    final_output_value = "1.6mm (Standard). Gängige Optionen: 0.8mm, 1.0mm, 1.2mm, 2.0mm."
                    has_special_handling = True
                # OSH Park-spezifische Überschreibungen
                elif mfg_id == "oshpark" and "kupferdicke" in param_lower_check:
                    final_display_name = param_template.split(":", 1)[0].strip() if ":" in param_template else "Kupferdicke" 
                    final_output_value = "1 oz/ft² (35µm) (Standard bei OSH Park für alle Lagen)"
                    has_special_handling = True
                # Material für OSH Park
                elif mfg_id == "oshpark" and "material" in param_lower_check:
                    final_display_name = "Material"
                    if layers <= 2:
                        final_output_value = "FR-4 (Standard für 2-Lagen bei OSH Park)"
                    else:
                        final_output_value = "FR408 (High-Speed Material für 4+ Lagen bei OSH Park)"
                    has_special_handling = True
                    printed_pcb_params.add("material")
                # Handle all manufacturers for copper parameters
                elif mfg_id in ["jlcpcb", "pcbway", "eurocircuits"] and "kupfer" in param_lower_check and "mm" not in final_output_value:
                    if "außen" in param_lower_check or (("dicke" in param_lower_check or "thickness" in param_lower_check) and "innen" not in param_lower_check):
                        final_display_name = param_template.split(":", 1)[0].strip() if ":" in param_template else "Kupferaußendicke"
                        final_output_value = "1 oz/ft² (35µm) (Standard). Für höhere Ströme ggf. 2oz (70µm) wählen."
                        has_special_handling = True
                elif "kupferinnendicke" in param_lower_check:
                    final_display_name = param_template.split(":", 1)[0].strip() if ":" in param_template else "Kupferinnendicke"
                    final_output_value = "0.5 oz/ft² (17.5µm) oder 1 oz/ft² (35µm) (Standard für Innenlagen)."
                    has_special_handling = True
                    if mfg_id == "jlcpcb" and layers <= 2:
                        final_output_value = "Nur relevant für Platinen mit mehr als 2 Lagen."
                elif any(p in param_lower_check for p in ["oberfläche", "surface finish", "finish"]):
                    final_display_name = param_template.split(":", 1)[0].strip() if ":" in param_template else "Oberfläche"
                    if mfg_id == "oshpark":
                        final_output_value = "ENIG (Immersion Gold, Standard bei OSH Park)"
                    else:
                        final_output_value = "HASL (Lead-Free) (kostengünstig) oder ENIG (Goldoberfläche, gut für feine Pitchs, teurer)."
                    has_special_handling = True
                elif any(p in param_lower_check for p in ["lötstopplack", "solder mask", "stoppmask"]):
                    final_display_name = param_template.split(":", 1)[0].strip() if ":" in param_template else "Lötstopplackfarbe"
                    if mfg_id == "oshpark":
                        final_output_value = "Violett (charakteristisch für OSH Park)"
                    else:
                        final_output_value = "Grün (oft Standard/günstigst). Andere: Rot, Blau, Schwarz, Weiß, Gelb."
                    has_special_handling = True
                elif any(p in param_lower_check for p in ["bestückungsdruckfarbe", "silkscreen"]):
                    final_display_name = param_template.split(":", 1)[0].strip() if ":" in param_template else "Bestückungsdruckfarbe"
                    final_output_value = "Weiß (Standard). Andere: Schwarz (je nach Lötstoppfarbe)."
                    has_special_handling = True
                elif "goldfinger" in param_lower_check or "gold fingers" in param_lower_check:
                    final_display_name = param_template.split(":", 1)[0].strip() if ":" in param_template else "Goldfinger"
                    final_output_value = "Nein (Standard, nur wenn Ihr Design Kantensteckverbinder hat)."
                    has_special_handling = True
                elif "castellated holes" in param_lower_check:
                    final_display_name = param_template.split(":", 1)[0].strip() if ":" in param_template else "Castellated Holes"
                    final_output_value = "Nein (Standard, nur wenn Ihr Design Module mit randmetallisierten Löchern sind)."
                    has_special_handling = True
                elif any(p in param_lower_check for p in ["lieferformat", "delivery format", "panelization", "nutzen"]):
                    final_display_name = param_template.split(":", 1)[0].strip() if ":" in param_template else "Lieferformat"
                    final_output_value = "Single Pieces (Standard für kleine Mengen). Bei größeren Mengen 'Panel by Manufacturer' or 'Panel by Customer' prüfen."
                    has_special_handling = True
                
                # Wenn der Parameter weder formatiert noch speziell behandelt wurde, versuche einen Standardwert zu finden
                if not has_special_handling and "{" not in final_output_value and final_output_value == param_template.split(":", 1)[1].strip() if ":" in param_template else "Prüfen Sie die Optionen auf der Webseite des Herstellers.":
                    if ":" in param_template:
                        value_part = param_template.split(":", 1)[1].strip()
                        default_val = self._extract_default_from_param_string(value_part)
                        if default_val:
                            final_output_value = default_val
                
                print(f"      - {final_display_name}: {final_output_value}")

            # Füge wichtige Parameter hinzu, falls sie nicht schon gedruckt wurden
            if "dimensions" not in printed_pcb_params: print(f"      - Platinengröße (Board Size): {board_width_str} x {board_height_str}")
            if "quantity" not in printed_pcb_params: print(f"      - Stückzahl (Quantity): {quantity}")
            if "layers" not in printed_pcb_params: print(f"      - Lagen (Layers): {layers}")

            # Assembly-Parameter nur anzeigen, wenn der Hersteller Bestückungsservice anbietet
            if bom_info["total_parts"] > 0 and mfg_info.get("assembly_service_url"):
                print("\n    --- 2. Parameter für Bestückungsservice (Assembly) ---")
                bom_filename = self.bom_dir / f"bom_{mfg_id}.csv" 
                cpl_filename = self.cpl_dir / f"cpl_{mfg_id}.csv"
                print(f"      (Stellen Sie sicher, dass Sie Ihre BOM- ({bom_filename}) und CPL-Dateien ({cpl_filename}) bereithalten)")
                
                printed_assembly_params = set()
                for param_template in mfg_info.get("typical_quote_parameters", {}).get("assembly", []):
                    # Standardwerte setzen
                    final_display_name = param_template
                    final_output_value = "Prüfen Sie die Optionen auf der Webseite des Herstellers."
                    has_special_handling = False

                    # Versuche, Parameter mit Projektdaten zu formatieren
                    try:
                        formatted_param = param_template.format_map(format_map)
                        if formatted_param != param_template:
                            if ":" in formatted_param:
                                parts = formatted_param.split(":", 1)
                                final_display_name = parts[0].strip()
                                final_output_value = parts[1].strip()
                            else:
                                final_display_name = param_template
                                final_output_value = formatted_param
                        elif ":" in param_template:
                            final_display_name = param_template.split(":", 1)[0].strip()
                            param_value_part = param_template.split(":", 1)[1].strip()
                            default_value = self._extract_default_from_param_string(param_value_part)
                            if default_value:
                                final_output_value = default_value
                            else:
                                final_output_value = param_value_part
                    except (KeyError, ValueError):
                        if ":" in param_template:
                            final_display_name = param_template.split(":", 1)[0].strip()
                            param_value_part = param_template.split(":", 1)[1].strip()
                            default_value = self._extract_default_from_param_string(param_value_part)
                            if default_value:
                                final_output_value = default_value
                            else:
                                final_output_value = param_value_part

                    # Schlüsselwortbasierte Spezialverarbeitung
                    param_lower_check = param_template.lower()

                    if any(p in param_lower_check for p in ["bestückungsseite", "assembly sides", "sides", "seiten"]):
                        final_display_name = param_template.split(":", 1)[0].strip() if ":" in param_template else "Bestückungsseite(n)"
                        final_output_value = "Top Side (Standard). 'Both Sides', falls Ihr Design Bauteile auf beiden Seiten hat."
                        has_special_handling = True
                    elif any(p in param_lower_check for p in ["anzahl eindeutiger bauteile", "unique components", "unique smt"]):
                        final_display_name = param_template.split(":", 1)[0].strip() if ":" in param_template else "Anzahl eindeutiger Bauteile"
                        final_output_value = f"{bom_info['unique_parts']} (aus Ihrer BOM)"
                        printed_assembly_params.add("unique_parts")
                        has_special_handling = True
                    elif any(p in param_lower_check for p in ["anzahl gesamter smd-bauteile", "total components", "total smt", "gesamtzahl"]):
                        final_display_name = param_template.split(":", 1)[0].strip() if ":" in param_template else "Anzahl gesamter SMD-Bauteile"
                        final_output_value = f"{bom_info['total_parts']} (aus Ihrer BOM)"
                        printed_assembly_params.add("total_parts")
                        has_special_handling = True
                    elif any(p in param_lower_check for p in ["anzahl tht-bauteile", "tht components"]):
                        final_display_name = param_template.split(":", 1)[0].strip() if ":" in param_template else "Anzahl THT-Bauteile"
                        final_output_value = f"{bom_info.get('tht_parts', 0)} (aus Ihrer BOM, falls vorhanden, sonst 0 oder manuell zählen)"
                        printed_assembly_params.add("tht_parts")
                        has_special_handling = True
                    elif any(p in param_lower_check for p in ["schablone", "stencil"]):
                        final_display_name = param_template.split(":", 1)[0].strip() if ":" in param_template else "Schablone (Stencil)"
                        final_output_value = "Ja, benötigt für SMD-Bestückung. Der Hersteller bietet dies meist an."
                        has_special_handling = True
                    
                    print(f"      - {final_display_name}: {final_output_value}")

                # Füge wichtige Assembly-Parameter hinzu, falls sie nicht schon gedruckt wurden
                if "unique_parts" not in printed_assembly_params: 
                    print(f"      - Anzahl einzigartige SMD-Bauteile: {bom_info['unique_parts']} (aus Ihrer BOM)")
                if "total_parts" not in printed_assembly_params: 
                    print(f"      - Gesamtzahl SMD-Platzierungen: {bom_info['total_parts']} (aus Ihrer BOM)")
                if "tht_parts" not in printed_assembly_params and bom_info.get('tht_parts',0) >= 0: 
                    print(f"      - Anzahl THT-Bauteile: {bom_info.get('tht_parts',0)} (aus Ihrer BOM, falls vorhanden)")

            # Zusätzliche herstellerspezifische Informationen
            if mfg_info.get("technical_data_url"):
                print(f"\n  Technische Daten/Fertigungsmöglichkeiten: {mfg_info['technical_data_url']}")
            if mfg_info.get("assembly_service_url"):
                print(f"  Informationen zum Bestückungsservice: {mfg_info['assembly_service_url']}")
            
            # Herstellerspezifische Tipps
            if mfg_id == "jlcpcb":
                print("  Tipp für JLCPCB: Nutzen Sie deren Online Gerber Viewer zum Überprüfen Ihrer hochgeladenen Daten. Achten Sie auf LCSC-Teilenummern in der BOM für die Bestückung.")
            elif mfg_id == "pcbway":
                print("  Tipp für PCBWay: Achten Sie auf Sonderangebote für Prototypen. Der Online-Chat kann bei Fragen helfen.")
            elif mfg_id == "oshpark":
                print("  Tipp für OSH Park: Ideal für kleine Stückzahlen und Prototypen in den USA. Preise sind oft pro Quadratzoll. Kein Bestückungsservice.")
        
        # Allgemeine Tipps für alle Hersteller
        print("\n" + "="*80)
        print("=== Allgemeine Tipps für Online-Preisangebote ===")
        print("="*80)
        print("  - **Einheiten prüfen**: Achten Sie darauf, ob Maße in mm oder Zoll (inch) erwartet werden.")
        print("  - **Dateien hochladen**: Für die genauesten Preise laden Sie Ihre Gerber-, BOM- und CPL-Dateien hoch.")
        print("  - **Lieferzeit beachten**: Kürzere Lieferzeiten erhöhen den Preis erheblich.")
        print("  - **Angebote vergleichen**: Holen Sie Angebote von mehreren Herstellern ein, wenn möglich.")
        print("  - **Sonderangebote**: Achten Sie auf Rabatte für Neukunden oder spezielle Prototypen-Angebote.")
        print("  - **Dokumentation**: Speichern Sie eine Kopie (Screenshot oder PDF) Ihrer finalen Konfiguration und des Preises.")
        print("  - **Mindestbestellmengen**: Manche Hersteller haben Mindestmengen oder -preise.")
        print("  - **Versandkosten**: Berücksichtigen Sie die Versandkosten und -zeiten, besonders bei internationalen Anbietern.")
        print("--- Ende des Leitfadens ---")

    def generate_gerber_with_external_tool(self, input_file: Optional[Path] = None) -> bool:
        """
        Erzeugt Gerber-Dateien mit externen Tools wie KiCad oder FlatCAM.
        Nutzt verfügbare Werkzeuge statt eigener Implementierung.
        
        Args:
            input_file: Eingabedatei für die Gerber-Generierung (optional)
        """
        if not input_file:
            # Erstelle eine temporäre KiCad PCB Datei als Eingabe
            temp_file = self.mfg_dir / "temp_design.kicad_pcb"
            if not self.export_to_kicad_pcb(temp_file):
                return False
            input_file = temp_file
        
        # Prüfe verfügbare Tools und wähle das beste aus
        if self.available_tools.get("kicad", False):
            logger.info("Verwende KiCad für die Gerber-Erzeugung...")
            return self._export_gerber_with_kicad(input_file)
        elif self.available_tools.get("flatcam", False):
            logger.info("Verwende FlatCAM für die Gerber-Erzeugung...")
            return self._export_gerber_with_flatcam(input_file)
        else:
            logger.warning("Keine kompatiblen EDA-Tools gefunden. Erstelle vereinfachte Gerber-Dateien...")
            return self._create_simplified_gerber()
    
    def _export_gerber_with_kicad(self, kicad_pcb_file: Path) -> bool:
        """Exportiert Gerber-Dateien mit KiCad-CLI."""
        try:
            # Erstelle KiCad-Plot-Konfiguration
            plot_config = self.mfg_dir / "plot_config.kibot.yaml"
            
            with open(plot_config, 'w') as f:
                f.write(f"""kibot:
  version: 1

outputs:
  - name: gerbers
    comment: "Gerber files for {self.manufacturer_info['name']}"
    type: gerber
    dir: {self.gerber_dir}
    options:
      exclude_edge_layer: true
      exclude_pads_from_silkscreen: true
      plot_sheet_reference: false
      plot_footprint_refs: true
      plot_footprint_values: true
      force_plot_invisible_refs_vals: false
      tent_vias: true
      line_width: 0.1
    layers:
      - F.Cu
      - B.Cu
      - F.SilkS
      - B.SilkS
      - F.Mask
      - B.Mask
      - F.Paste
      - B.Paste
      - Edge.Cuts

  - name: drill
    comment: "Drill files for {self.manufacturer_info['name']}"
    type: excellon
    dir: {self.gerber_dir}
    options:
      map: true
""")
            
            # Führe KiCad-CLI aus
            command = [
                "kicad-cli", "pcb", "export", "gerbers",
                "--output", str(self.gerber_dir),
                str(kicad_pcb_file)
            ]
            
            process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate(timeout=60)
            
            if process.returncode != 0:
                logger.error(f"KiCad Gerber-Export fehlgeschlagen: {stderr.decode('utf-8')}")
                return False
            
            logger.info(f"Gerber-Dateien erfolgreich mit KiCad erstellt in: {self.gerber_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Export mit KiCad: {e}")
            return False
    
    def _export_gerber_with_flatcam(self, input_file: Path) -> bool:
        """Exportiert Gerber-Dateien mit FlatCAM."""
        try:
            # Erstelle FlatCAM-Skript
            flatcam_script = self.mfg_dir / "flatcam_export.txt"
            
            with open(flatcam_script, 'w') as f:
                f.write(f"""# FlatCAM Gerber Export Skript
open_gerber("{input_file}")
set_sys("{self.gerber_dir}")
export_gerber("top_copper", "{self.manufacturer_info['gerber_naming']['top_copper']}")
export_gerber("bottom_copper", "{self.manufacturer_info['gerber_naming']['bottom_copper']}")
export_gerber("drill", "{self.manufacturer_info['gerber_naming']['drill']}")
save_project("{self.mfg_dir}/flatcam_project.flatcam")
quit()
""")
            
            # Führe FlatCAM aus
            command = [
                "flatcam",
                "--shellmode",
                "--script", str(flatcam_script)
            ]
            
            process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate(timeout=120)
            
            if process.returncode != 0:
                logger.error(f"FlatCAM Gerber-Export fehlgeschlagen: {stderr.decode('utf-8')}")
                return False
            
            logger.info(f"Gerber-Dateien erfolgreich mit FlatCAM erstellt in: {self.gerber_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Export mit FlatCAM: {e}")
            return False
    
    def _create_simplified_gerber(self) -> bool:
        """
        Erstellt vereinfachte Gerber-Dateien für ein Platinenlayout.
        Nur als Notlösung gedacht, wenn keine EDA-Tools verfügbar sind.
        """
        logger.warning("Erstelle vereinfachte Gerber-Dateien (nicht ideal für Produktion).")
        
        try:
            # Gerber-Layer definieren
            gerber_layers = {
                "top_copper": f"PizzaBoard.{self.manufacturer_info['gerber_naming']['top_copper']}",
                "bottom_copper": f"PizzaBoard.{self.manufacturer_info['gerber_naming']['bottom_copper']}",
                "top_silkscreen": "PizzaBoard.GTO",
                "bottom_silkscreen": "PizzaBoard.GBO",
                "top_mask": "PizzaBoard.GTS",
                "bottom_mask": "PizzaBoard.GBS",
                "outline": "PizzaBoard.GKO",
                "drill": f"PizzaBoard.{self.manufacturer_info['gerber_naming']['drill']}"
            }
            
            # Erzeuge einfache Gerber-Dateien mit rechteckigem Board-Outline
            board_width = 100  # 100mm
            board_height = 100  # 100mm
            
            # Erstelle Layer-Dateien mit Mindestinhalt
            for layer_name, filename in gerber_layers.items():
                filepath = self.gerber_dir / filename
                
                with open(filepath, 'w') as f:
                    # Gerber-Header (RS-274X Format)
                    f.write(f"G04 Simplified Gerber from PCB Export Tool (Layer: {layer_name})*\n")
                    f.write("%FSLAX46Y46*%\n")  # Format Statement: Leading zeros suppressed, Absolute coords, 4 integer digits, 6 fractional digits
                    f.write("%MOMM*%\n")  # Mode: Millimeters
                    f.write("%LPD*%\n")  # Layer Polarity: Dark
                    
                    # Define apertures
                    f.write("%ADD10C,0.15*%\n")  # Circular aperture with 0.15mm diameter for outlines
                    f.write("%ADD11C,0.5*%\n")   # Circular aperture for pads
                    
                    # Draw board outline for all layers
                    if layer_name == "outline":
                        f.write("G01*\n")  # Linear interpolation mode
                        f.write("D10*\n")  # Select aperture 10
                        f.write("X0Y0D02*\n")  # Move to 0,0
                        f.write(f"X{board_width*10000}Y0D01*\n")  # Draw to width,0
                        f.write(f"X{board_width*10000}Y{board_height*10000}D01*\n")  # Draw to width,height
                        f.write(f"X0Y{board_height*10000}D01*\n")  # Draw to 0,height
                        f.write("X0Y0D01*\n")  # Close the rectangle
                    
                    # Add component pads to copper and mask layers
                    if layer_name in ["top_copper", "top_mask"]:
                        f.write("G01*\n")  # Linear interpolation mode
                        f.write("D11*\n")  # Select aperture 11
                        
                        # Place pads for components
                        for comp in self.components:
                            x = int(comp["x"] * 10000)  # Convert to 0.1µm units
                            y = int(comp["y"] * 10000)
                            f.write(f"X{x}Y{y}D03*\n")  # Flash pad at component position
                    
                    # Add component labels to silkscreen
                    if layer_name == "top_silkscreen":
                        # In simplified Gerber, we can't easily add text
                        # We would usually draw rectangles representing components
                        f.write("G01*\n")  # Linear interpolation mode
                        f.write("D10*\n")  # Select thin aperture for outlines
                        
                        for comp in self.components:
                            x = int(comp["x"] * 10000)  # Convert to 0.1µm units
                            y = int(comp["y"] * 10000)
                            w = int(comp["width"] * 10000 / 2)  # Half width
                            h = int(comp["height"] * 10000 / 2)  # Half height
                            
                            # Draw rectangle for component outline
                            f.write(f"X{x-w}Y{y-h}D02*\n")  # Move to top-left
                            f.write(f"X{x+w}Y{y-h}D01*\n")  # Draw to top-right
                            f.write(f"X{x+w}Y{y+h}D01*\n")  # Draw to bottom-right
                            f.write(f"X{x-w}Y{y+h}D01*\n")  # Draw to bottom-left
                            f.write(f"X{x-w}Y{y-h}D01*\n")  # Close the rectangle
                    
                    # End of file
                    f.write("M02*\n")
            
            # Erzeuge eine README-Datei mit Erklärungen
            with open(self.gerber_dir / "README.txt", 'w') as f:
                f.write("VEREINFACHTE GERBER-DATEIEN - NUR ZUR REFERENZ\n")
                f.write("==========================================\n\n")
                f.write("Diese Gerber-Dateien wurden automatisch aus dem SVG-Layout erzeugt.\n")
                f.write("HINWEIS: Diese Dateien sind stark vereinfacht und nicht für die direkte Fertigung geeignet!\n\n")
                f.write("Für eine professionelle Fertigung sollten Sie:\n")
                f.write("1. Ein geeignetes EDA-Tool wie KiCad oder Eagle verwenden\n")
                f.write("2. Das Layout entsprechend IPC-Standards überarbeiten\n")
                f.write("3. Eine DRC (Design Rule Check) durchführen\n\n")
                f.write("Diese Dateien dienen nur als Ausgangspunkt für die Weiterverarbeitung.")
            
            logger.info("Vereinfachte Gerber-Dateien erstellt. HINWEIS: Nicht für direkte Fertigung geeignet!")
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei der Erstellung der vereinfachten Gerber-Dateien: {e}")
            return False

    def create_manufacturing_package(self) -> bool:
        """
        Erstellt ein komplettes Fertigungspaket mit Gerber-Dateien, BOM und CPL.
        Komprimiert alle Dateien in eine ZIP-Datei für einfachen Upload.
        """
        try:
            # Erstelle ZIP-Datei mit allen Fertigungsdateien
            zip_path = self.mfg_dir / f"manufacturing_package_{self.manufacturer}.zip"
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # Füge Gerber-Dateien hinzu
                for file in self.gerber_dir.glob("*.*"):
                    if file.is_file() and file.suffix != ".txt":
                        zipf.write(file, arcname=f"gerber/{file.name}")
                
                # Füge BOM hinzu
                bom_file = self.mfg_dir / f"bom_{self.manufacturer}.csv"
                if bom_file.exists():
                    zipf.write(bom_file, arcname=f"bom_{self.manufacturer}.csv")
                
                # Füge CPL hinzu
                cpl_file = self.mfg_dir / f"cpl_{self.manufacturer}.csv"
                if cpl_file.exists():
                    zipf.write(cpl_file, arcname=f"cpl_{self.manufacturer}.csv")
                
                # Füge README hinzu
                readme_path = self.mfg_dir / "README.txt"
                with open(readme_path, 'w') as f:
                    f.write(f"PCB-Fertigungspaket für {self.manufacturer_info['name']}\n")
                    f.write("=======================================\n\n")
                    f.write(f"Erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write("Enthaltene Dateien:\n")
                    f.write("- Gerber-Dateien: Enthält alle notwendigen Layer für die Fertigung\n")
                    f.write(f"- BOM: Stückliste im {self.manufacturer_info['name']}-Format\n")
                    f.write(f"- CPL: Bestückungsplan im {self.manufacturer_info['name']}-Format\n\n")
                    f.write("Anleitung für den Upload bei %s:\n" % self.manufacturer_info['name'])
                    f.write(f"1. Besuchen Sie {self.manufacturer_info['url']}\n")
                    f.write("2. Laden Sie im Bestellprozess die ZIP-Datei mit den Gerber-Dateien hoch\n")
                    f.write("3. Laden Sie die BOM- und CPL-Dateien separat hoch, falls Sie SMT-Bestückung wünschen\n")
                    
                zipf.write(readme_path, arcname="README.txt")
            
            logger.info(f"Fertigungspaket erstellt: {zip_path}")
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei der Erstellung des Fertigungspakets: {e}")
            return False

    def launch_manufacturer_website(self) -> bool:
        """Öffnet die Website des ausgewählten PCB-Herstellers."""
        url = self.manufacturer_info["url"]
        logger.info(f"Öffne Website von {self.manufacturer_info['name']}: {url}")
        
        try:
            webbrowser.open(url)
            return True
        except Exception as e:
            logger.error(f"Fehler beim Öffnen der Website: {e}")
            return False

    def suggest_alternative_approaches(self) -> List[Dict[str, str]]:
        """
        Gibt Vorschläge für alternative Ansätze zur PCB-Fertigung zurück.
        Nützlich für Benutzer, die Probleme mit dem Export-Prozess haben.
        """
        alternatives = [
            {
                "title": "KiCad direkt verwenden",
                "description": "Importieren Sie Ihr Design in KiCad und nutzen Sie dessen leistungsstarke Export-Funktionen.",
                "steps": [
                    "1. Installieren Sie KiCad von https://www.kicad.org/",
                    "2. Erstellen Sie ein neues PCB-Projekt",
                    "3. Importieren Sie Ihr SVG als Referenz-Layer",
                    "4. Zeichnen Sie die Komponenten und Verbindungen nach",
                    "5. Exportieren Sie Gerber-Dateien über 'Datei > Platinenherstellung > Gerber...'"
                ],
                "url": "https://www.kicad.org/"
            },
            {
                "title": "Online PCB-Tools nutzen",
                "description": "Online-Tools wie EasyEDA bieten einfache Möglichkeiten zur PCB-Erstellung und direkten Bestellung.",
                "steps": [
                    "1. Registrieren Sie sich bei EasyEDA (https://easyeda.com/)",
                    "2. Importieren Sie Ihr Design als SVG oder zeichnen Sie es neu",
                    "3. Platzieren Sie Komponenten aus der Bibliothek",
                    "4. Bestellen Sie direkt bei JLCPCB (integriert)"
                ],
                "url": "https://easyeda.com/"
            },
            {
                "title": "Lokale PCB-Fertigung",
                "description": "Suchen Sie nach lokalen Prototyping-Diensten oder Fab Labs in Ihrer Nähe.",
                "steps": [
                    "1. Suchen Sie nach 'Fab Lab' oder 'Makerspace' in Ihrer Region",
                    "2. Fragen Sie nach PCB-Fertigungsmöglichkeiten",
                    "3. Bringen Sie Ihr Design in einem kompatiblen Format mit"
                ],
                "url": "https://www.fablabs.io/labs"
            }
        ]
        
        return alternatives

    def export_all(self) -> bool:
        """
        Führt den kompletten Export-Prozess durch:
        1. Parst die SVG-Datei
        2. Erstellt KiCad PCB (falls möglich)
        3. Generiert Gerber-Dateien
        4. Erstellt BOM und CPL
        5. Erstellt ein komplettes Fertigungspaket
        """
        if not self.parse_svg():
            logger.error("Export abgebrochen: Fehler beim Parsen der SVG-Datei.")
            return False
        
        logger.info(f"Starte Export-Prozess für {self.manufacturer_info['name']}...")
        
        # Exportiere zu KiCad PCB (für bessere Kompatibilität)
        if not self.export_to_kicad_pcb():
            logger.warning("Warnung: KiCad PCB konnte nicht erstellt werden.")
        
        # Generiere Gerber-Dateien (mit externen Tools oder vereinfacht)
        if not self.generate_gerber_with_external_tool():
            logger.warning("Warnung: Gerber-Dateien konnten nicht mit externen Tools erstellt werden.")
            # Nutze Fallback-Methode
            if not self._create_simplified_gerber():
                logger.error("Fehler: Gerber-Dateien konnten nicht erstellt werden.")
                return False
        
        # Erstelle BOM und CPL
        if not self.generate_bom():
            logger.warning("Warnung: BOM konnte nicht erstellt werden.")
        
        if not self.generate_cpl():
            logger.warning("Warnung: CPL konnte nicht erstellt werden.")
        
        # Erstelle fertiges Paket für den Upload
        if not self.create_manufacturing_package():
            logger.warning("Warnung: Fertigungspaket konnte nicht erstellt werden.")
        
        logger.info(f"Export abgeschlossen. Ergebnisse in: {self.mfg_dir}")
        return True

    def _extract_default_from_param_string(self, param_string: str) -> Optional[str]:
        """
        Versucht, den Standardwert aus einem Parametertext zu extrahieren.
        Beispiele:
        - "(z.B. 1oz (35µm Standard), 2oz (70µm))" -> "1oz (35µm)"
        - "(Standard: 1.6mm)" -> "1.6mm"
        
        Args:
            param_string: Der Text, der einen Standardwert enthalten könnte.
            
        Returns:
            Den extrahierten Standardwert oder None, wenn keiner gefunden wurde.
        """
        if not param_string:
            return None
            
        # Versuche, Standard-Werte aus verschiedenen Formaten zu extrahieren
        standard_patterns = [
            r'Standard[:\)]*\s*([^,\)]+)',  # Standard: Wert oder Standard) Wert
            r'Standard[^\(]*\(([^)]+)\)',   # Standard(Wert)
            r'\(([^)]+)\s+Standard\)',      # (Wert Standard)
            r'([^,]+)\s+Standard[,\)]',     # Wert Standard, oder Wert Standard) 
            r'Standard[:\s]+([^,\)]+)',     # Standard: Wert oder Standard Wert
        ]
        
        for pattern in standard_patterns:
            match = re.search(pattern, param_string, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Wenn kein expliziter Standardwert gefunden wurde, versuche den ersten Wert in Klammern zu finden
        match = re.search(r'\(([^,\)]+)[,\)]', param_string)
        if match:
            return match.group(1).strip()
            
        return None

    def get_api_quote(self, quantity: int = 10, layers: int = 2, 
                   thickness_mm: float = 1.6, copper_weight_oz: float = 1.0,
                   solder_mask_color: str = "Green", silkscreen_color: str = "White",
                   surface_finish: str = "HASL", 
                   include_assembly: bool = False) -> Dict:
        """
        Holt ein automatisches Preisangebot vom PCB-Hersteller über dessen API.
        
        Args:
            quantity: Die gewünschte Stückzahl der Platinen.
            layers: Die Anzahl der Kupferlagen.
            thickness_mm: Die Dicke der Leiterplatte in mm.
            copper_weight_oz: Das Kupfergewicht in oz.
            solder_mask_color: Die Farbe der Lötstoppmaske.
            silkscreen_color: Die Farbe des Bestückungsdrucks.
            surface_finish: Die Oberflächenbehandlung.
            include_assembly: Ob Bestückung in die Anfrage einbezogen werden soll.
            
        Returns:
            Ein Dictionary mit den Angebotsinformationen oder Fehlermeldungen.
        """
        if not api_client.available:
            return {
                "success": False,
                "error": "API-Client-Modul nicht verfügbar",
                "details": "Das pcb_api_clients-Modul konnte nicht importiert werden."
            }
        
        # Überprüfe, ob der Hersteller API-Unterstützung bietet
        if not self.manufacturer_info.get("api_available", False):
            return {
                "success": False,
                "error": f"{self.manufacturer_info['name']} bietet keine API für Preisangebote",
                "details": f"Für manuelle Preisabschätzungen nutzen Sie bitte den Online-Kalkulator: {self.manufacturer_info.get('quote_calculator_url', self.manufacturer_info['url'])}"
            }
        
        # Holt zuerst BOM-Daten für die Assembly-Informationen
        if include_assembly:
            bom_info = self._get_bom_summary()
        else:
            bom_info = {"unique_parts": 0, "total_parts": 0, "tht_parts": 0}
        
        # Stelle sicher, dass wir die Platinengröße haben
        if not self.board_dimensions["width"] or not self.board_dimensions["height"]:
            logger.info("Keine Platinenabmessungen verfügbar, versuche SVG zu parsen...")
            if not self.parse_svg():
                return {
                    "success": False,
                    "error": "Platinenabmessungen konnten nicht bestimmt werden",
                    "details": "Die SVG-Datei konnte nicht korrekt analysiert werden, um die Platinenabmessungen zu ermitteln."
                }
        
        # Bereite die Parameter für die Preisanfrage vor
        pcb_params = {
            "width_mm": self.board_dimensions["width"],
            "height_mm": self.board_dimensions["height"],
            "layers": layers,
            "quantity": quantity,
            "pcb_thickness": thickness_mm,
            "copper_weight": copper_weight_oz,
            "solder_mask_color": solder_mask_color,
            "silkscreen_color": silkscreen_color,
            "surface_finish": surface_finish
        }
        
        # Füge Assembly-Parameter hinzu, wenn gewünscht
        if include_assembly:
            pcb_params["assembly"] = True
            pcb_params["assembly_side"] = "top"  # Standard: nur Oberseite
            pcb_params["unique_components"] = bom_info["unique_parts"]
            pcb_params["total_components"] = bom_info["total_parts"]
            pcb_params["tht_components"] = bom_info.get("tht_parts", 0)
        
        # Hole den API-Client für den Hersteller
        client = api_client.get_pcb_client(self.manufacturer)
        if not client:
            return {
                "success": False,
                "error": f"Kein API-Client für {self.manufacturer_info['name']} verfügbar",
                "details": "Trotz API-Unterstützung konnte kein passender API-Client initialisiert werden."
            }
        
        # Versuche die API-Authentifizierung
        logger.info(f"Authentifiziere mit {self.manufacturer_info['name']} API...")
        if not client.is_authenticated():
            # Biete interaktive Authentifizierung an, falls noch nicht authentifiziert
            auth_result = self._authenticate_api_client(client)
            if not auth_result["success"]:
                return auth_result
        
        # Hole das Preisangebot
        logger.info(f"Hole Preisangebot von {self.manufacturer_info['name']}...")
        quote_result = client.get_quote(pcb_params)
        
        # Füge zusätzliche Metadaten hinzu
        if quote_result.get("success", False):
            quote_result["query_params"] = pcb_params
            quote_result["timestamp"] = datetime.now().isoformat()
            
            # Speichere das Angebot in einer JSON-Datei zur späteren Referenz
            quote_file = self.mfg_dir / f"quote_{self.manufacturer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                with open(quote_file, 'w') as f:
                    json.dump(quote_result, f, indent=2)
                logger.info(f"Preisangebot gespeichert: {quote_file}")
            except Exception as e:
                logger.warning(f"Fehler beim Speichern des Preisangebots: {e}")
        
        return quote_result

    def _authenticate_api_client(self, client) -> Dict:
        """
        Authentifiziert einen API-Client mit Benutzerinteraktion.
        
        Args:
            client: Der zu authentifizierende API-Client
            
        Returns:
            Ein Dictionary mit dem Authentifizierungsergebnis
        """
        print(f"\n=== Authentifizierung für {client.name} API ===")
        print("Um ein Preisangebot zu erhalten, müssen Sie sich bei der API authentifizieren.")
        
        if client.name == "JLCPCB":
            print("Hinweis: Sie benötigen einen JLCPCB API-Key und API-Secret.")
            print("Diese können Sie im JLCPCB-Entwicklerportal erhalten: https://jlcpcb.com/api/")
            
            api_key = input("JLCPCB API-Key: ").strip()
            api_secret = input("JLCPCB API-Secret: ").strip()
            
            if not api_key or not api_secret:
                return {
                    "success": False,
                    "error": "Fehlende Authentifizierungsdaten",
                    "details": "API-Key und API-Secret sind erforderlich."
                }
            
            result = client.authenticate(api_key=api_key, api_secret=api_secret)
            
        elif client.name == "PCBWay":
            print("Hinweis: Sie benötigen einen PCBWay API-Key.")
            print("Diesen können Sie im PCBWay-Account erhalten: https://www.pcbway.com/api.html")
            
            api_key = input("PCBWay API-Key: ").strip()
            
            if not api_key:
                return {
                    "success": False,
                    "error": "Fehlende Authentifizierungsdaten",
                    "details": "API-Key ist erforderlich."
                }
            
            result = client.authenticate(api_key=api_key)
            
        elif client.name == "Eurocircuits":
            print("Hinweis: Sie benötigen Ihre Eurocircuits-Anmeldedaten.")
            print("Diese entsprechen Ihren normalen Zugangsdaten für den Eurocircuits-Shop.")
            
            username = input("Eurocircuits Benutzername: ").strip()
            password = input("Eurocircuits Passwort: ").strip()
            
            if not username or not password:
                return {
                    "success": False,
                    "error": "Fehlende Authentifizierungsdaten",
                    "details": "Benutzername und Passwort sind erforderlich."
                }
            
            result = client.authenticate(username=username, password=password)
            
        else:
            return {
                "success": False,
                "error": "Unbekannter API-Client",
                "details": f"Authentifizierungsmethode für {client.name} nicht implementiert."
            }
        
        if result:
            return {
                "success": True,
                "details": f"Authentifizierung bei {client.name} erfolgreich."
            }
        else:
            return {
                "success": False,
                "error": "Authentifizierung fehlgeschlagen",
                "details": f"Die Anmeldung bei {client.name} war nicht erfolgreich. Bitte überprüfen Sie Ihre Eingaben."
            }

    def get_and_display_quotes(self, quantity: int = 10, layers: int = 2, 
                              thickness_mm: float = 1.6, include_assembly: bool = False,
                              manufacturers: Optional[List[str]] = None) -> Dict:
        """
        Holt Preisangebote von mehreren Herstellern und zeigt einen Vergleich an.
        
        Args:
            quantity: Die gewünschte Stückzahl.
            layers: Die Anzahl der Kupferlagen.
            thickness_mm: Die Dicke der Leiterplatte in mm.
            include_assembly: Ob Bestückung in die Anfrage einbezogen werden soll.
            manufacturers: Liste der Hersteller (optional, Standard: alle mit API-Unterstützung)
            
        Returns:
            Ein Dictionary mit allen Angeboten und Vergleichsinformationen
        """
        if not api_client.available:
            print("\n❌ API-Client-Modul nicht verfügbar.")
            print("Automatische Preisangebote können nicht abgerufen werden.")
            print("Verwenden Sie stattdessen die manuelle Preisabschätzung mit --get-quotes.")
            return {"success": False, "quotes": []}
        
        # Wenn keine Hersteller angegeben wurden, verwende alle mit API-Unterstützung
        if not manufacturers:
            manufacturers = [mfg_id for mfg_id, mfg_info in PCB_MANUFACTURERS.items() 
                           if mfg_info.get("api_available", False)]
        
        # Überprüfe auf gültige Hersteller-IDs
        valid_manufacturers = []
        for mfg_id in manufacturers:
            if mfg_id in PCB_MANUFACTURERS and PCB_MANUFACTURERS[mfg_id].get("api_available", False):
                valid_manufacturers.append(mfg_id)
            else:
                logger.warning(f"Hersteller '{mfg_id}' hat keine API-Unterstützung und wird übersprungen.")
        
        if not valid_manufacturers:
            print("\n❌ Keine Hersteller mit API-Unterstützung ausgewählt.")
            return {"success": False, "quotes": []}
        
        # Hole die Angebote von allen ausgewählten Herstellern
        quotes = []
        current_manufacturer = self.manufacturer
        
        for mfg_id in valid_manufacturers:
            # Temporär zum angefragten Hersteller wechseln
            self.manufacturer = mfg_id
            self.manufacturer_info = PCB_MANUFACTURERS[mfg_id]
            
            print(f"\nAnfrage für {self.manufacturer_info['name']} wird vorbereitet...")
            
            # Hole das Angebot
            quote_result = self.get_api_quote(
                quantity=quantity,
                layers=layers,
                thickness_mm=thickness_mm,
                include_assembly=include_assembly
            )
            
            if quote_result.get("success", False):
                quotes.append(quote_result)
                print(f"✅ Preisangebot von {self.manufacturer_info['name']} erfolgreich erhalten!")
            else:
                print(f"❌ Fehler beim Abrufen des Angebots von {self.manufacturer_info['name']}: {quote_result.get('error', 'Unbekannter Fehler')}")
                print(f"Details: {quote_result.get('details', '')}")
        
        # Zurück zum ursprünglichen Hersteller wechseln
        self.manufacturer = current_manufacturer
        self.manufacturer_info = PCB_MANUFACTURERS[current_manufacturer]
        
        # Vergleiche die Angebote
        if quotes:
            comparison = api_client.compare_quotes(quotes)
            
            # Zeige die Ergebnisse an
            self._display_quote_comparison(comparison, include_assembly)
            
            return {
                "success": True,
                "quotes": quotes,
                "comparison": comparison
            }
        else:
            print("\n❌ Keine Preisangebote konnten abgerufen werden.")
            return {
                "success": False,
                "quotes": []
            }

    def _display_quote_comparison(self, comparison: Dict, include_assembly: bool = False) -> None:
        """
        Zeigt einen übersichtlichen Vergleich der Preisangebote an.
        
        Args:
            comparison: Das Vergleichs-Dictionary von compare_quotes()
            include_assembly: Ob Bestückung in den Angeboten enthalten ist
        """
        if not comparison.get("quotes"):
            print("\nKeine Angebote zum Vergleichen verfügbar.")
            return
        
        print("\n" + "="*80)
        print("=== VERGLEICH DER PREISANGEBOTE ===")
        print("="*80)
        
        # Zeige alle Angebote in einer Tabelle an
        print(f"\n{'Hersteller':<15} {'Preis':<15} {'Preis (EUR)':<15} {'Lieferzeit':<15} {'Angebotslink':<20}")
        print("-"*80)
        
        for quote in comparison["quotes"]:
            price_original = quote["total_price_original"]["formatted"]
            price_eur = quote["total_price_eur_formatted"]
            days = f"{quote['estimated_days']} Tage" if quote.get('estimated_days') else "Unbekannt"
            link = "[Verfügbar]" if quote.get("quote_url") else "N/A"
            
            print(f"{quote['manufacturer']:<15} {price_original:<15} {price_eur:<15} {days:<15} {link:<20}")
        
        print("-"*80)
        
        # Zeige das beste Angebot basierend auf Preis
        if comparison.get("cheapest"):
            cheapest = comparison["cheapest"]
            print(f"\n🏆 Günstigstes Angebot: {cheapest['manufacturer']} - {cheapest['total_price_eur_formatted']}")
        
        # Zeige das schnellste Angebot
        if comparison.get("fastest"):
            fastest = comparison["fastest"]
            days = f"{fastest['estimated_days']} Tage" if fastest.get('estimated_days') else "Unbekannt"
            print(f"🚀 Schnellstes Angebot: {fastest['manufacturer']} - {days}")
        
        print("\nHinweise:")
        if include_assembly:
            print("- Die Preise beinhalten Leiterplattenfertigung UND Bestückung.")
        else:
            print("- Die Preise beziehen sich nur auf die Leiterplattenfertigung ohne Bestückung.")
        print("- Alle Preise wurden zur besseren Vergleichbarkeit in EUR umgerechnet.")
        print("- Versandkosten können je nach Lieferziel und -option variieren.")
        print("- Für verbindliche Angebote besuchen Sie bitte die Websites der Hersteller.")
        
        # Zeige die Links zu den Angeboten
        print("\nDirecte Angebotslinks:")
        for quote in comparison["quotes"]:
            if quote.get("quote_url"):
                print(f"- {quote['manufacturer']}: {quote['quote_url']}")

def show_banner():
    """Zeigt einen Willkommens-Banner an."""
    print("""
╔════════════════════════════════════════════════════╗
║                                                    ║
║  PCB Export Tool - Flexible PCB-Fertigungslösung   ║
║                                                    ║
║  Unterstützt verschiedene PCB-Hersteller:          ║
║  - JLCPCB                                          ║
║  - PCBWay                                          ║
║  - OSH Park                                        ║
║  - Eurocircuits                                    ║
║                                                    ║
║  NEU: Automatische Angebote direkt von Herstellern ║
║  Verwenden Sie --api-quote oder --compare-quotes   ║
║                                                    ║
║  Version 2.0 - Mai 2025                            ║
║                                                    ║
╚════════════════════════════════════════════════════╝
""")

def main():
    """Hauptfunktion für die direkte Ausführung des Skripts."""
    show_banner()
    
    parser = argparse.ArgumentParser(description="PCB Export Tool für verschiedene Fertigungsanbieter")
    parser.add_argument("-s", "--svg", type=str, help="Pfad zur SVG-Datei (optional)")
    parser.add_argument("-o", "--output", type=str, help="Ausgabeverzeichnis (optional)")
    parser.add_argument("-m", "--manufacturer", type=str, choices=PCB_MANUFACTURERS.keys(), 
                        default="jlcpcb", help="PCB-Hersteller (Standard: jlcpcb)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Ausführliche Ausgabe")
    parser.add_argument("--get-quotes", action="store_true", help="Zeigt einen Leitfaden für Preisabschätzungen an.")
    parser.add_argument("--quantity", type=int, default=10, help="Stückzahl für die Preisabschätzung (Standard: 10).")
    parser.add_argument("--layers", type=int, default=2, help="Anzahl Kupferlagen für Preisabschätzung (Standard: 2).")
    parser.add_argument("--api-quote", action="store_true", help="Holt automatische Preisangebote über die API.")
    parser.add_argument("--compare-quotes", action="store_true", help="Vergleicht Preisangebote von mehreren Herstellern.")
    parser.add_argument("--include-assembly", action="store_true", help="Schließt Bestückung in die Preisanfragen ein.")
    parser.add_argument("--thickness", type=float, default=1.6, help="Leiterplattendicke in mm (Standard: 1.6).")
    
    args = parser.parse_args()
    
    # Setze Logging-Level basierend auf Verbose-Flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Bestimme SVG-Pfad
    svg_path = Path(args.svg) if args.svg else SVG_PATH
    if not svg_path.exists():
        logger.error(f"SVG-Datei nicht gefunden: {svg_path}")
        print("\nBitte geben Sie den korrekten Pfad zur SVG-Datei an:")
        print("  python pcb_export.py --svg /pfad/zu/design.svg")
        return 1
    
    # Bestimme Ausgabeverzeichnis
    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Erstelle Export-Tool und führe Export durch
    exporter = PCBExportTool(svg_path, output_dir, args.manufacturer)
    
    # API-Preisanfrage
    if args.api_quote:
        # Stelle sicher, dass SVG geparsed wurde, um Dimensionen zu haben
        if not exporter.components and not exporter.board_dimensions["width"]:
            exporter.parse_svg()  # Parse SVG explizit, wenn noch nicht geschehen
        
        quote_result = exporter.get_api_quote(
            quantity=args.quantity,
            layers=args.layers,
            thickness_mm=args.thickness,
            include_assembly=args.include_assembly
        )
        
        if quote_result.get("success", False):
            # Zeige das Ergebnis an
            print("\n✅ Preisangebot erfolgreich erhalten!")
            print(f"\nHersteller: {quote_result['manufacturer']}")
            print(f"Leiterplattenpreis: {api_client.format_price(quote_result['pcb_price']['amount'], quote_result['pcb_price']['currency'])}")
            
            if args.include_assembly and 'assembly_price' in quote_result:
                print(f"Bestückungspreis: {api_client.format_price(quote_result['assembly_price']['amount'], quote_result['assembly_price']['currency'])}")
            
            print(f"Versandkosten: {api_client.format_price(quote_result['shipping_price']['amount'], quote_result['shipping_price']['currency'])}")
            print(f"Gesamtpreis: {api_client.format_price(quote_result['total_price']['amount'], quote_result['total_price']['currency'])}")
            
            if quote_result.get("estimated_days"):
                print(f"Geschätzte Lieferzeit: {quote_result['estimated_days']} Tage")
            
            if quote_result.get("quote_url"):
                print(f"\nAngebotslink: {quote_result['quote_url']}")
                response = input("\nMöchten Sie den Angebotslink im Browser öffnen? (j/n): ")
                if response.lower() in ['j', 'ja', 'y', 'yes']:
                    webbrowser.open(quote_result['quote_url'])
        else:
            print(f"\n❌ Fehler beim Abrufen des Preisangebots: {quote_result.get('error', 'Unbekannter Fehler')}")
            print(f"Details: {quote_result.get('details', '')}")
            
            # Biete manuelle Preisabschätzung als Fallback an
            response = input("\nMöchten Sie stattdessen die manuelle Preisabschätzung verwenden? (j/n): ")
            if response.lower() in ['j', 'ja', 'y', 'yes']:
                exporter.provide_price_estimation_guidance(quantity=args.quantity, layers=args.layers)
        
        return 0
    
    # Vergleich von Preisangeboten mehrerer Hersteller
    elif args.compare_quotes:
        # Stelle sicher, dass SVG geparsed wurde, um Dimensionen zu haben
        if not exporter.components and not exporter.board_dimensions["width"]:
            exporter.parse_svg()  # Parse SVG explizit, wenn noch nicht geschehen
        
        # Hole und vergleiche Angebote
        exporter.get_and_display_quotes(
            quantity=args.quantity,
            layers=args.layers,
            thickness_mm=args.thickness,
            include_assembly=args.include_assembly
        )
        
        return 0
    
    # Manuelle Preisabschätzung
    elif args.get_quotes:
        # Stelle sicher, dass SVG geparsed wurde, um Dimensionen zu haben
        if not exporter.components and not exporter.board_dimensions["width"]:
            exporter.parse_svg()  # Parse SVG explizit, wenn noch nicht geschehen
        
        exporter.provide_price_estimation_guidance(quantity=args.quantity, layers=args.layers)
        return 0  # Beende nach der Preisinfo

    # Standard-Export-Prozess
    print(f"\nStarte Export für {PCB_MANUFACTURERS[args.manufacturer]['name']}...\n")
    
    # Erkenne verfügbare Tools und informiere den Benutzer
    available_tools = []
    for tool_id, available in exporter.available_tools.items():
        if available:
            available_tools.append(EDA_TOOLS[tool_id]['name'])
    
    if available_tools:
        print(f"Verfügbare EDA-Tools: {', '.join(available_tools)}")
    else:
        print("Warnung: Keine kompatiblen EDA-Tools gefunden. Funktionalität eingeschränkt.")
        print("Für beste Ergebnisse installieren Sie KiCad von https://www.kicad.org/\n")
    
    # Führe Export durch
    success = exporter.export_all()
    
    if success:
        print("\n✅ Export erfolgreich abgeschlossen!")
        print(f"Ergebnisse wurden in {exporter.mfg_dir} gespeichert.")
        
        # Zeige Fertigungspaket-Details an
        zip_path = exporter.mfg_dir / f"manufacturing_package_{args.manufacturer}.zip"
        if zip_path.exists():
            print(f"\nFertigungspaket: {zip_path}")
        
        # Biete an, die Hersteller-Website zu öffnen
        response = input(f"\nMöchten Sie die Website von {PCB_MANUFACTURERS[args.manufacturer]['name']} öffnen? (j/n): ")
        if response.lower() in ['j', 'ja', 'y', 'yes']:
            exporter.launch_manufacturer_website()
        
        # Biete an, Preisangebote einzuholen
        if PCB_MANUFACTURERS[args.manufacturer].get("api_available", False) and api_client.available:
            response = input(f"\nMöchten Sie ein automatisches Preisangebot von {PCB_MANUFACTURERS[args.manufacturer]['name']} einholen? (j/n): ")
            if response.lower() in ['j', 'ja', 'y', 'yes']:
                quote_result = exporter.get_api_quote(
                    quantity=args.quantity,
                    layers=args.layers,
                    include_assembly=args.include_assembly
                )
                
                if quote_result.get("success", False):
                    print("\n✅ Preisangebot erfolgreich erhalten!")
                    print(f"Gesamtpreis: {api_client.format_price(quote_result['total_price']['amount'], quote_result['total_price']['currency'])}")
                    
                    if quote_result.get("quote_url"):
                        response = input("\nMöchten Sie den Angebotslink im Browser öffnen? (j/n): ")
                        if response.lower() in ['j', 'ja', 'y', 'yes']:
                            webbrowser.open(quote_result['quote_url'])
                else:
                    print(f"\n❌ Fehler beim Abrufen des Preisangebots: {quote_result.get('error', 'Unbekannter Fehler')}")
                    print("Verwenden Sie stattdessen die manuelle Preisabschätzung mit --get-quotes.")
    else:
        print("\n❌ Export fehlgeschlagen.")
        print("Bitte prüfen Sie die Fehlermeldungen oben.")
        
        # Zeige alternative Ansätze an
        print("\nHier sind einige alternative Ansätze zur PCB-Fertigung:")
        for i, alt in enumerate(exporter.suggest_alternative_approaches(), 1):
            print(f"\n{i}. {alt['title']}")
            print(f"   {alt['description']}")
            print("   Schritte:")
            for step in alt['steps']:
                print(f"   {step}")
        
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())