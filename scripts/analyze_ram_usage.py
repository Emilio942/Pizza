#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAM-Nutzungsanalyse für Pizza-Detektions-System

Dieses Skript analysiert detailliert den RAM-Verbrauch aller relevanten Komponenten
des Pizza-Detektions-Systems auf dem RP2040 für den kritischsten Betriebsfall
(eine Inferenz inklusive Vorverarbeitung).

Komponenten:
- Tensor Arena (Inferenzpuffer)
- Framebuffer (Kamera-Bildspeicher)
- Zwischenpuffer für Vorverarbeitung/Ausgabe
- Stack
- Heap
- Globale Variablen
- Statische Puffer
"""

import os
import sys
import json
import time
from pathlib import Path
import logging
from datetime import datetime

# Projektverzeichnis zum Pythonpfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Konfiguration des Loggers
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Konstanten
OUTPUT_DIR = project_root / "output" / "ram_analysis"

def ensure_output_directory():
    """Stellt sicher, dass das Ausgabeverzeichnis existiert."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_tensor_arena_report():
    """Liest den vorhandenen Tensor-Arena-Bericht."""
    report_path = OUTPUT_DIR / "tensor_arena_report.json"
    if not report_path.exists():
        return None
    
    with open(report_path, 'r') as f:
        return json.load(f)

def read_emulator_logs():
    """
    Liest die neuesten Emulator-Logs aus dem output/emulator_logs Verzeichnis
    und extrahiert Informationen zur RAM-Nutzung.
    """
    logs_dir = project_root / "output" / "emulator_logs"
    performance_logs = sorted(logs_dir.glob("performance_log_*.csv"), key=os.path.getmtime, reverse=True)
    system_logs = sorted(logs_dir.glob("system_log_*.log"), key=os.path.getmtime, reverse=True)
    
    if not performance_logs or not system_logs:
        logger.error("Keine Emulator-Logs gefunden")
        return None
    
    latest_perf_log = performance_logs[0]
    latest_sys_log = system_logs[0]
    
    logger.info(f"Analysiere neueste Logs: {latest_perf_log.name} und {latest_sys_log.name}")
    
    # Performance-Log analysieren (enthält RAM-Nutzung)
    ram_usage_data = {}
    with open(latest_perf_log, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:  # Überprüfe, ob Daten vorhanden sind (Header + mind. 1 Zeile)
            logger.warning(f"Performance-Log {latest_perf_log.name} enthält zu wenig Daten")
            return None
        
        # Header extrahieren
        header = lines[0].strip().split(',')
        
        # RAM-Nutzungsdaten extrahieren
        for line in lines[1:]:
            if not line.strip():
                continue
            
            values = line.strip().split(',')
            data_point = {header[i]: values[i] for i in range(min(len(header), len(values)))}
            
            if 'ram_used_kb' in data_point:
                ram_usage_data['total_ram_used_kb'] = float(data_point['ram_used_kb'])
                break  # Wir benötigen nur einen Datenpunkt
    
    # System-Log analysieren (enthält detailliertere RAM-Informationen)
    with open(latest_sys_log, 'r') as f:
        content = f.read()
        
        # Suche nach RAM-Aufteilungsinformationen
        import re
        
        # Extrahiere den System-RAM-Overhead
        system_ram_match = re.search(r'System-RAM-Overhead:\s+(\d+\.?\d*)\s*KB', content)
        if system_ram_match:
            ram_usage_data['system_ram_kb'] = float(system_ram_match.group(1))
        
        # Extrahiere die Framebuffer-Größe
        framebuffer_match = re.search(r'Framebuffer-Größe:\s+(\d+\.?\d*)\s*KB', content)
        if framebuffer_match:
            ram_usage_data['framebuffer_kb'] = float(framebuffer_match.group(1))
        
        # Extrahiere die Modell-RAM-Nutzung
        model_ram_match = re.search(r'Modell-RAM:\s+(\d+\.?\d*)\s*KB', content)
        if model_ram_match:
            ram_usage_data['model_ram_kb'] = float(model_ram_match.group(1))
    
    return ram_usage_data

def estimate_preprocessing_buffer_size():
    """
    Schätzt die Größe des Zwischenpuffers für die Bildvorverarbeitung.
    Basierend auf den Berechnungen aus der Projektdokumentation und dem Code.
    """
    # Angenommen, wir haben ein 96x96 RGB-Bild für die Vorverarbeitung
    preprocessing_buffer_kb = 96 * 96 * 3 / 1024  # RGB-Bild (Byte pro Pixel)
    return preprocessing_buffer_kb

def estimate_stack_size():
    """
    Schätzt die Stack-Größe basierend auf typischen Werten für den RP2040.
    """
    # Typische Stack-Größe für eingebettete Anwendungen auf dem RP2040
    # Kann je nach Anwendung und Compiler-Einstellungen variieren
    return 8.0  # 8 KB

def estimate_heap_size():
    """
    Schätzt die maximal verwendete Heap-Größe während der Inferenz.
    """
    # Schätzung basierend auf typischen Werten für die Anwendung
    return 5.0  # 5 KB

def estimate_static_buffers():
    """
    Schätzt die Größe statischer Puffer, die nicht zum Modell oder zur Vorverarbeitung gehören.
    """
    # Typische Werte für diverse statische Puffer (Logging, Kommunikation, etc.)
    return 3.0  # 3 KB

def generate_ram_usage_report():
    """
    Generiert einen detaillierten Bericht zur RAM-Nutzung.
    """
    tensor_arena_report = read_tensor_arena_report()
    emulator_log_data = read_emulator_logs()
    
    if not tensor_arena_report:
        logger.warning("Tensor-Arena-Bericht nicht gefunden, verwende Standardwerte")
        tensor_arena_kb = 43.2  # Basierend auf vorherigen Ergebnissen
    else:
        tensor_arena_kb = tensor_arena_report["improved_estimation"]["estimated_tensor_arena_kb"]
    
    if not emulator_log_data:
        logger.warning("Emulator-Log-Daten nicht gefunden, verwende Standardwerte")
        system_ram_kb = 40.0
        framebuffer_kb = 76.8  # QVGA (320x240) Graustufenbild
        model_ram_kb = 0.0  # Wird durch Tensor-Arena abgedeckt
    else:
        system_ram_kb = emulator_log_data.get('system_ram_kb', 40.0)
        framebuffer_kb = emulator_log_data.get('framebuffer_kb', 76.8)
        model_ram_kb = emulator_log_data.get('model_ram_kb', 0.0)
    
    # Schätze andere Komponenten
    preprocessing_buffer_kb = estimate_preprocessing_buffer_size()
    stack_kb = estimate_stack_size()
    heap_kb = estimate_heap_size()
    static_buffers_kb = estimate_static_buffers()
    
    # Gesamtsumme berechnen
    total_ram_kb = (
        tensor_arena_kb +
        system_ram_kb +
        framebuffer_kb +
        preprocessing_buffer_kb +
        stack_kb +
        heap_kb +
        static_buffers_kb
    )
    
    # Bericht erstellen
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ra2040_specs": {
            "total_ram_kb": 264.0,  # RP2040 hat 264 KB RAM
            "max_usable_ram_kb": 204.0  # Ziel-Limit laut Aufgabe
        },
        "ram_usage_components": {
            "tensor_arena_kb": tensor_arena_kb,
            "system_overhead_kb": system_ram_kb,
            "framebuffer_kb": framebuffer_kb,
            "preprocessing_buffer_kb": preprocessing_buffer_kb,
            "stack_kb": stack_kb,
            "heap_kb": heap_kb, 
            "static_buffers_kb": static_buffers_kb
        },
        "summary": {
            "total_ram_usage_kb": total_ram_kb,
            "available_ram_kb": 264.0 - total_ram_kb,
            "percentage_used": (total_ram_kb / 264.0) * 100,
            "meets_requirements": total_ram_kb <= 204.0
        },
        "analysis": {
            "critical_components": [
                {"name": "Tensor Arena", "size_kb": tensor_arena_kb, "percentage": (tensor_arena_kb / total_ram_kb) * 100},
                {"name": "Framebuffer", "size_kb": framebuffer_kb, "percentage": (framebuffer_kb / total_ram_kb) * 100},
                {"name": "System Overhead", "size_kb": system_ram_kb, "percentage": (system_ram_kb / total_ram_kb) * 100}
            ],
            "recommendations": []
        }
    }
    
    # Empfehlungen basierend auf der Analyse
    if report["summary"]["total_ram_usage_kb"] > 204.0:
        report["analysis"]["recommendations"].append({
            "component": "Overall",
            "suggestion": "Die Gesamtnutzung überschreitet das Ziel von 204 KB. Optimierung notwendig."
        })
    
    if tensor_arena_kb > 30.0:
        report["analysis"]["recommendations"].append({
            "component": "Tensor Arena",
            "suggestion": "Erwägen Sie Pruning oder Int4-Quantisierung, um die Tensor-Arena-Größe zu reduzieren."
        })
    
    if framebuffer_kb > 40.0:
        report["analysis"]["recommendations"].append({
            "component": "Framebuffer",
            "suggestion": "Verwenden Sie eine niedrigere Auflösung oder ein einfacheres Pixelformat (z.B. Graustufen statt RGB)."
        })
    
    # Bericht speichern
    ensure_output_directory()
    report_path = OUTPUT_DIR / "ram_usage_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"RAM-Nutzungsbericht gespeichert unter: {report_path}")
    
    # Erstelle auch eine Markdown-Version für bessere Lesbarkeit
    md_report = f"""# RAM-Nutzungsanalyse für Pizza-Detektions-System

## Übersicht

**Datum:** {report['timestamp']}

**Gesamtnutzung:** {report['summary']['total_ram_usage_kb']:.1f} KB / 264.0 KB ({report['summary']['percentage_used']:.1f}%)

**Verfügbarer RAM:** {report['summary']['available_ram_kb']:.1f} KB

**Erfüllt Anforderungen:** {'Ja' if report['summary']['meets_requirements'] else 'Nein'} (Ziel: max 204 KB)

## Komponentenübersicht

| Komponente | Größe (KB) | Anteil (%) |
|------------|------------|------------|
| Tensor Arena | {tensor_arena_kb:.1f} | {(tensor_arena_kb / total_ram_kb) * 100:.1f} |
| Framebuffer | {framebuffer_kb:.1f} | {(framebuffer_kb / total_ram_kb) * 100:.1f} |
| System Overhead | {system_ram_kb:.1f} | {(system_ram_kb / total_ram_kb) * 100:.1f} |
| Vorverarbeitungspuffer | {preprocessing_buffer_kb:.1f} | {(preprocessing_buffer_kb / total_ram_kb) * 100:.1f} |
| Stack | {stack_kb:.1f} | {(stack_kb / total_ram_kb) * 100:.1f} |
| Heap | {heap_kb:.1f} | {(heap_kb / total_ram_kb) * 100:.1f} |
| Statische Puffer | {static_buffers_kb:.1f} | {(static_buffers_kb / total_ram_kb) * 100:.1f} |

## Empfehlungen

"""
    
    # Füge Empfehlungen zum Markdown-Bericht hinzu
    for recommendation in report["analysis"]["recommendations"]:
        md_report += f"- **{recommendation['component']}**: {recommendation['suggestion']}\n"
    
    # Speichere Markdown-Bericht
    md_path = OUTPUT_DIR / "ram_usage_report.md"
    with open(md_path, 'w') as f:
        f.write(md_report)
    
    logger.info(f"Markdown-Bericht gespeichert unter: {md_path}")
    
    return report

if __name__ == "__main__":
    try:
        ensure_output_directory()
        report = generate_ram_usage_report()
        
        # Zeige eine kurze Zusammenfassung in der Konsole
        print("\nRAM-Nutzungsanalyse abgeschlossen:")
        print(f"Gesamtnutzung: {report['summary']['total_ram_usage_kb']:.1f} KB / 264.0 KB ({report['summary']['percentage_used']:.1f}%)")
        print(f"Verfügbarer RAM: {report['summary']['available_ram_kb']:.1f} KB")
        print(f"Erfüllt Anforderungen: {'Ja' if report['summary']['meets_requirements'] else 'Nein'} (Ziel: max 204 KB)")
        print(f"\nDetaillierter Bericht gespeichert unter: {OUTPUT_DIR}/ram_usage_report.json")
        print(f"Markdown-Bericht gespeichert unter: {OUTPUT_DIR}/ram_usage_report.md")
        
    except Exception as e:
        logger.exception(f"Fehler bei der RAM-Nutzungsanalyse: {e}")
        sys.exit(1)
