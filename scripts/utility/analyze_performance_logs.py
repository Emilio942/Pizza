#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance-Log-Analyzer für das Pizza-Erkennungssystem

Dieses Skript analysiert die Performance-Logs, die vom RP2040 
über UART oder SD-Karte ausgegeben werden, und erstellt 
informative Visualisierungen und Berichte.

Nutzung:
    python analyze_performance_logs.py --input <Pfad-zur-Log-Datei> --output <Ausgabe-Verzeichnis>

Autor: Pizza Detection Team
Datum: 2025-05-05
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import serial
import serial.tools.list_ports
import json
from typing import Dict, List, Tuple, Optional, Union
import csv

# Konfiguration des Loggers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_class_names(config_path: str = None) -> List[str]:
    """
    Lädt Klassennamen aus einer Konfigurationsdatei
    
    Args:
        config_path: Pfad zur Konfigurationsdatei im JSON-Format
        
    Returns:
        Liste der Klassennamen
    
    Notes:
        Die Konfigurationsdatei sollte ein JSON-Objekt sein, dessen Schlüssel
        die Klassennamen darstellen.
    """
    # Wenn kein Pfad angegeben ist, verwende den Standardpfad
    if config_path is None:
        # Bestimme den Projektroot anhand des Skriptpfads
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        config_path = project_root / "data" / "class_definitions.json"
    
    try:
        logger.info(f"Lade Klassennamen aus: {config_path}")
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            
        # Die Schlüssel im JSON-Objekt sind die Klassennamen
        class_names = list(config_data.keys())
        
        if not class_names:
            logger.error(f"Keine Klassennamen in der Konfigurationsdatei gefunden: {config_path}")
            raise ValueError("Die Konfigurationsdatei enthält keine Klassennamen")
        
        logger.info(f"Klassennamen aus Konfiguration geladen: {class_names}")
        return class_names
    
    except FileNotFoundError:
        logger.error(f"Konfigurationsdatei nicht gefunden: {config_path}")
        raise FileNotFoundError(f"Die Konfigurationsdatei {config_path} wurde nicht gefunden. "
                               "Bitte stellen Sie sicher, dass die Datei existiert.")
    
    except json.JSONDecodeError:
        logger.error(f"Fehler beim Parsen der Konfigurationsdatei: {config_path}")
        raise json.JSONDecodeError(f"Die Konfigurationsdatei {config_path} enthält kein gültiges JSON-Format.", "", 0)
    
    except Exception as e:
        logger.error(f"Unerwarteter Fehler beim Laden der Klassennamen: {e}")
        raise

class PerformanceLogAnalyzer:
    """Klasse zur Analyse von Performance-Logs des Pizza-Erkennungssystems"""
    
    def __init__(self, input_path: str, output_dir: str, class_config_path: str = None):
        """
        Initialisiert den Analyzer
        
        Args:
            input_path: Pfad zur Log-Datei oder COM-Port (z.B. "COM3", "/dev/ttyACM0")
            output_dir: Verzeichnis für die Ausgabe der Analysen
            class_config_path: Pfad zur Klassendefinitionsdatei (optional)
        """
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.data = None  # DataFrame mit den Log-Daten
        
        # Lade Klassennamen aus der Konfigurationsdatei
        try:
            self.class_names = load_class_names(class_config_path)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Fehler beim Laden der Klassennamen: {e}")
            # Fallback zu Standardklassennamen, nur wenn ein kritischer Fehler auftritt
            logger.warning("Verwende Standardklassennamen als Fallback")
            self.class_names = ["basic", "burnt", "combined", "mixed", "progression", "segment"]
        
        # Erstelle Ausgabeverzeichnis, falls es nicht existiert
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Bestimme den Input-Typ
        if input_path.startswith("COM") or input_path.startswith("/dev/tty"):
            self.input_type = "serial"
        else:
            self.input_type = "file"
    
    def load_data(self) -> pd.DataFrame:
        """
        Lädt die Log-Daten aus der Datei oder dem seriellen Port
        
        Returns:
            DataFrame mit den Log-Daten
        """
        if self.input_type == "serial":
            data = self._load_from_serial()
        else:
            data = self._load_from_file()
        
        # Daten in DataFrame konvertieren
        if data:
            logger.info(f"Daten geladen: {len(data)} Einträge")
            
            # Spalten definieren
            columns = [
                "Timestamp", "InferenceTime", "PeakRamUsage", 
                "CpuLoad", "Temperature", "VddVoltage", "BatteryVoltage", "Prediction", "Confidence"
            ]
            
            # DataFrame erzeugen
            self.data = pd.DataFrame(data, columns=columns)
            
            # Datentypen konvertieren
            self.data["Timestamp"] = pd.to_numeric(self.data["Timestamp"])
            self.data["InferenceTime"] = pd.to_numeric(self.data["InferenceTime"])
            self.data["PeakRamUsage"] = pd.to_numeric(self.data["PeakRamUsage"])
            self.data["CpuLoad"] = pd.to_numeric(self.data["CpuLoad"]) / 100.0  # Prozent
            self.data["Temperature"] = pd.to_numeric(self.data["Temperature"]) / 100.0  # Grad Celsius
            self.data["VddVoltage"] = pd.to_numeric(self.data["VddVoltage"])  # Millivolt
            self.data["BatteryVoltage"] = pd.to_numeric(self.data["BatteryVoltage"])  # Millivolt
            self.data["Prediction"] = pd.to_numeric(self.data["Prediction"]).astype(int)
            self.data["Confidence"] = pd.to_numeric(self.data["Confidence"]) / 100.0  # Prozent
            
            # Berechne relative Zeit seit Beginn in Sekunden
            self.data["RelativeTime"] = (self.data["Timestamp"] - self.data["Timestamp"].min()) / 1000.0
            
            # Füge Klassennamen für Vorhersagen hinzu
            self.data["PredictionClass"] = self.data["Prediction"].apply(
                lambda x: self.class_names[x] if 0 <= x < len(self.class_names) else "unknown")
            
            return self.data
        else:
            logger.error("Keine Daten geladen")
            return pd.DataFrame()
    
    def _load_from_file(self) -> List[List[str]]:
        """
        Lädt die Log-Daten aus einer CSV-Datei
        
        Returns:
            Liste mit den Log-Einträgen
        """
        try:
            data = []
            with open(self.input_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)  # Header überspringen
                
                # Bestimme das Format anhand der Header-Zeile
                if header and len(header) > 0:
                    logger.info(f"CSV-Header erkannt: {header}")
                    
                    # Prüfe das Format der Datei
                    if 'timestamp' in header[0].lower() and 'temperature_c' in [col.lower() for col in header]:
                        # Neues Format (aus output/emulator_logs)
                        # timestamp,cpu_usage_percent,ram_used_kb,flash_used_kb,inference_time_ms,temperature_c
                        col_indices = {
                            'timestamp': header.index([col for col in header if 'timestamp' in col.lower()][0]),
                            'inference_time': header.index([col for col in header if 'inference_time' in col.lower()][0]),
                            'ram_used': header.index([col for col in header if 'ram_used' in col.lower()][0]),
                            'cpu_usage': header.index([col for col in header if 'cpu_usage' in col.lower()][0]),
                            'temperature': header.index([col for col in header if 'temperature' in col.lower()][0]),
                        }
                        
                        # Versuche die Spannungsspalten zu finden, falls sie existieren
                        try:
                            col_indices['vdd_voltage'] = header.index([col for col in header if 'vdd_voltage' in col.lower()][0])
                            col_indices['battery_voltage'] = header.index([col for col in header if 'battery_voltage' in col.lower()][0])
                        except (ValueError, IndexError):
                            # Falls nicht vorhanden, verwende Standardwerte
                            logger.warning("Spannungswerte nicht im Header gefunden, verwende Standardwerte")
                        
                        # Standardwerte für fehlende Spalten
                        prediction = 0  # Default: basic
                        confidence = 90  # Default: 90%
                        vdd_voltage = 3300  # Default: 3.3V (3300mV)
                        battery_voltage = 3700  # Default: 3.7V (3700mV)
                        
                        for row in reader:
                            if len(row) >= len(header):
                                # Konvertiere Zeitstempel in Epochenmillisekunden
                                try:
                                    timestamp_str = row[col_indices['timestamp']]
                                    if 'T' in timestamp_str:  # ISO-Format
                                        from datetime import datetime
                                        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                        timestamp = int(dt.timestamp() * 1000)
                                    else:
                                        timestamp = int(float(timestamp_str))
                                except Exception as e:
                                    logger.warning(f"Fehler beim Parsen des Zeitstempels: {e}")
                                    timestamp = 0
                                
                                # Konvertiere die restlichen Werte
                                try:
                                    inference_time = int(float(row[col_indices['inference_time']]) * 1000)  # ms zu us
                                    ram_used = int(float(row[col_indices['ram_used']]) * 1024)  # KB zu Bytes
                                    cpu_usage = int(float(row[col_indices['cpu_usage']]))
                                    temperature = int(float(row[col_indices['temperature']]) * 100)  # C zu 1/100 C
                                    
                                    # Spannungswerte, falls vorhanden
                                    vdd_voltage = int(float(row[col_indices.get('vdd_voltage', -1)])) if 'vdd_voltage' in col_indices else 3300
                                    battery_voltage = int(float(row[col_indices.get('battery_voltage', -1)])) if 'battery_voltage' in col_indices else 3700
                                except Exception as e:
                                    logger.warning(f"Fehler beim Konvertieren der Werte: {e}")
                                    continue
                                
                                # Füge die Daten im erwarteten Format hinzu
                                data.append([
                                    str(timestamp),
                                    str(inference_time),
                                    str(ram_used),
                                    str(cpu_usage),
                                    str(temperature),
                                    str(vdd_voltage),
                                    str(battery_voltage),
                                    str(prediction),
                                    str(confidence)
                                ])
                    else:
                        # Altes Format (original)
                        # Timestamp,InferenceTime,PeakRamUsage,CpuLoad,Temperature,VddVoltage,BatteryVoltage,Prediction,Confidence
                        for row in reader:
                            if len(row) == 9:  # Erwartetes Format mit 9 Spalten (enthält Spannungen)
                                data.append(row)
                            elif len(row) == 7:  # Altes Format mit 7 Spalten (ohne Spannungen)
                                # Füge Standardspannungswerte ein (3300mV für VDD, 3700mV für Batterie)
                                modified_row = row[:5] + ["3300", "3700"] + row[5:]
                                data.append(modified_row)
                else:
                    logger.warning("Kein Header in der CSV-Datei gefunden")
            
            if not data:
                logger.error(f"Keine gültigen Daten in der Datei {self.input_path} gefunden")
                
            return data
        except Exception as e:
            logger.error(f"Fehler beim Laden der Datei {self.input_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _load_from_serial(self) -> List[List[str]]:
        """
        Lädt die Log-Daten von einem seriellen Port
        
        Returns:
            Liste mit den Log-Einträgen
        """
        try:
            data = []
            
            logger.info(f"Verbinde mit seriellem Port {self.input_path}...")
            with serial.Serial(self.input_path, 115200, timeout=10) as ser:
                logger.info("Verbindung hergestellt. Empfange Daten (Strg+C zum Beenden)...")
                
                # Header überspringen (falls vorhanden)
                line = ser.readline().decode('utf-8').strip()
                if "Timestamp" in line:
                    logger.info("CSV-Header erkannt und übersprungen")
                
                # Maximal 100 Einträge oder bis Benutzer abbricht
                try:
                    for _ in range(100):
                        line = ser.readline().decode('utf-8').strip()
                        if not line:
                            continue
                        
                        parts = line.split(',')
                        if len(parts) == 7:
                            data.append(parts)
                            logger.info(f"Eintrag empfangen: {line}")
                except KeyboardInterrupt:
                    logger.info("Datenerfassung vom Benutzer beendet")
            
            return data
        except Exception as e:
            logger.error(f"Fehler bei der seriellen Verbindung zu {self.input_path}: {e}")
            return []
    
    def analyze(self) -> Dict:
        """
        Führt eine umfassende Analyse der Log-Daten durch
        
        Returns:
            Dictionary mit den Analyseergebnissen
        """
        if self.data is None or self.data.empty:
            logger.error("Keine Daten zum Analysieren verfügbar")
            return {
                "total_entries": 0,
                "error": "Keine Daten zum Analysieren verfügbar",
                "inference_time": {},
                "ram_usage": {},
                "cpu_load": {},
                "temperature": {},
                "predictions": {
                    "class_distribution": {}
                }
            }
        
        # Grundlegende Statistiken
        stats = {
            "total_entries": len(self.data),
            "total_duration_ms": self.data["Timestamp"].max() - self.data["Timestamp"].min(),
            "inference_time": {
                "min_us": self.data["InferenceTime"].min(),
                "max_us": self.data["InferenceTime"].max(),
                "avg_us": self.data["InferenceTime"].mean(),
                "median_us": self.data["InferenceTime"].median(),
                "std_us": self.data["InferenceTime"].std()
            },
            "ram_usage": {
                "min_bytes": self.data["PeakRamUsage"].min(),
                "max_bytes": self.data["PeakRamUsage"].max(),
                "avg_bytes": self.data["PeakRamUsage"].mean(),
                "median_bytes": self.data["PeakRamUsage"].median()
            },
            "cpu_load": {
                "min_percent": self.data["CpuLoad"].min() * 100,
                "max_percent": self.data["CpuLoad"].max() * 100,
                "avg_percent": self.data["CpuLoad"].mean() * 100
            },
            "temperature": {
                "min_c": self.data["Temperature"].min(),
                "max_c": self.data["Temperature"].max(),
                "avg_c": self.data["Temperature"].mean()
            },
            "vdd_voltage": {
                "min_mv": self.data["VddVoltage"].min(),
                "max_mv": self.data["VddVoltage"].max(),
                "avg_mv": self.data["VddVoltage"].mean()
            },
            "battery_voltage": {
                "min_mv": self.data["BatteryVoltage"].min(),
                "max_mv": self.data["BatteryVoltage"].max(),
                "avg_mv": self.data["BatteryVoltage"].mean()
            },
            "predictions": {
                "distribution": self.data["PredictionClass"].value_counts().to_dict(),
                "avg_confidence": self.data["Confidence"].mean() * 100
            }
        }
        
        # Berechne Inferenz pro Sekunde (wenn sinnvoll)
        if stats["total_duration_ms"] > 1000:
            stats["inferences_per_second"] = len(self.data) / (stats["total_duration_ms"] / 1000)
        
        return stats
    
    def generate_visualizations(self):
        """Erzeugt verschiedene Visualisierungen der Performance-Daten"""
        if self.data is None or self.data.empty:
            logger.error("Keine Daten zum Visualisieren verfügbar")
            return
        
        logger.info("Erzeuge Visualisierungen...")
        
        # Einheitlicher Stil
        plt.style.use('seaborn-v0_8-whitegrid')
        colors = sns.color_palette("viridis", 4)
        
        # 1. Zeitreihenanalyse der Inferenzzeit
        self._plot_inference_time_series()
        
        # 2. Zeitreihenanalyse des RAM-Verbrauchs
        self._plot_ram_usage_series()
        
        # 3. Verteilung der Inferenzzeiten
        self._plot_inference_time_histogram()
        
        # 4. Verteilung der Vorhersageklassen
        self._plot_prediction_distribution()
        
        # 5. Korrelation zwischen Metriken
        self._plot_metrics_correlation()
        
        # 6. Dashboard mit allen wichtigen Metriken
        self._create_dashboard()
        
        # 7. Zeitreihenanalyse der Spannungen
        self._plot_voltage_series()
        
        # 8. Korrelation zwischen Spannung und Temperatur
        self._plot_voltage_temperature_correlation()
        
        logger.info(f"Visualisierungen wurden im Verzeichnis {self.output_dir} gespeichert")
    
    def _plot_inference_time_series(self):
        """Erzeugt ein Liniendiagramm der Inferenzzeit über die Zeit"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(self.data["RelativeTime"], self.data["InferenceTime"] / 1000, 
                marker='o', linestyle='-', alpha=0.7, color="#1E88E5")
        
        ax.set_title("Inferenzzeit im Zeitverlauf", fontsize=16)
        ax.set_xlabel("Zeit seit Start (Sekunden)", fontsize=12)
        ax.set_ylabel("Inferenzzeit (ms)", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Durchschnitt als horizontale Linie
        mean_time = self.data["InferenceTime"].mean() / 1000
        ax.axhline(y=mean_time, color='r', linestyle='--', alpha=0.7, 
                   label=f'Durchschnitt: {mean_time:.2f} ms')
        
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "inference_time_series.png", dpi=150)
        plt.close()
    
    def _plot_ram_usage_series(self):
        """Erzeugt ein Liniendiagramm des RAM-Verbrauchs über die Zeit"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(self.data["RelativeTime"], self.data["PeakRamUsage"] / 1024, 
                marker='o', linestyle='-', alpha=0.7, color="#43A047")
        
        ax.set_title("Spitzen-RAM-Verbrauch im Zeitverlauf", fontsize=16)
        ax.set_xlabel("Zeit seit Start (Sekunden)", fontsize=12)
        ax.set_ylabel("RAM-Verbrauch (KB)", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Durchschnitt als horizontale Linie
        mean_ram = self.data["PeakRamUsage"].mean() / 1024
        ax.axhline(y=mean_ram, color='r', linestyle='--', alpha=0.7, 
                   label=f'Durchschnitt: {mean_ram:.2f} KB')
        
        # Maximales RAM als Referenz (204KB laut Projektdokumentation)
        max_available_ram = 204  # KB
        ax.axhline(y=max_available_ram, color='black', linestyle='-.', alpha=0.5,
                  label=f'Verfügbarer RAM: {max_available_ram} KB')
        
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "ram_usage_series.png", dpi=150)
        plt.close()
    
    def _plot_inference_time_histogram(self):
        """Erzeugt ein Histogramm der Inferenzzeiten"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.histplot(self.data["InferenceTime"] / 1000, bins=20, kde=True, ax=ax, color="#7E57C2")
        
        ax.set_title("Verteilung der Inferenzzeiten", fontsize=16)
        ax.set_xlabel("Inferenzzeit (ms)", fontsize=12)
        ax.set_ylabel("Häufigkeit", fontsize=12)
        
        # Statistik-Annotation
        stats_text = (
            f"Min: {self.data['InferenceTime'].min() / 1000:.2f} ms\n"
            f"Max: {self.data['InferenceTime'].max() / 1000:.2f} ms\n"
            f"Mittelwert: {self.data['InferenceTime'].mean() / 1000:.2f} ms\n"
            f"Median: {self.data['InferenceTime'].median() / 1000:.2f} ms\n"
            f"Std.abw.: {self.data['InferenceTime'].std() / 1000:.2f} ms"
        )
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "inference_time_histogram.png", dpi=150)
        plt.close()
    
    def _plot_metrics_correlation(self):
        """Erstellt eine Korrelationsmatrix der wichtigsten Metriken"""
        # Wähle relevante Metriken aus
        metrics = self.data[["InferenceTime", "PeakRamUsage", "CpuLoad", 
                            "Temperature", "VddVoltage", "BatteryVoltage", "Confidence"]]
        
        # Berechne die Korrelationsmatrix
        corr = metrics.corr()
        
        # Normalisiere die Inferenzzeit und RAM-Nutzung für bessere Lesbarkeit
        metrics_scaled = metrics.copy()
        metrics_scaled["InferenceTime"] = metrics_scaled["InferenceTime"] / 1000  # us zu ms
        metrics_scaled["PeakRamUsage"] = metrics_scaled["PeakRamUsage"] / 1024  # bytes zu kb
        metrics_scaled["VddVoltage"] = metrics_scaled["VddVoltage"] / 1000  # mV zu V
        metrics_scaled["BatteryVoltage"] = metrics_scaled["BatteryVoltage"] / 1000  # mV zu V
        
        # Angepasste Spaltennamen für die Visualisierung
        metrics_scaled.columns = ["Inferenzzeit (ms)", "RAM (KB)", "CPU-Last", 
                                 "Temperatur (°C)", "VDD (V)", "Batterie (V)", "Konfidenz"]
        
        # Plotte die Korrelationsmatrix
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        heatmap = sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                             mask=mask, vmin=-1, vmax=1, center=0,
                             square=True, linewidths=.5, cbar_kws={"shrink": .5})
        
        plt.title("Korrelation zwischen Performance-Metriken", fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / "metrics_correlation.png", dpi=150)
        plt.close()
        
        # Plotte zusätzlich ein Pairplot für detailliertere Einblicke
        g = sns.pairplot(metrics_scaled, diag_kind="kde", corner=True, 
                         plot_kws={"alpha": 0.5})
        g.fig.suptitle("Detaillierte Zusammenhänge zwischen Metriken", y=1.02, fontsize=16)
        plt.savefig(self.output_dir / "metrics_pairplot.png", dpi=150)
        plt.close()
    
    def _plot_voltage_series(self):
        """Erzeugt ein Liniendiagramm der Spannungen über die Zeit"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # VDD-Spannung
        ax1.plot(self.data["RelativeTime"], self.data["VddVoltage"] / 1000, 
                marker='o', linestyle='-', alpha=0.7, color="#E53935")
        
        ax1.set_title("VDD-Spannung im Zeitverlauf", fontsize=16)
        ax1.set_ylabel("Spannung (V)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Durchschnitt als horizontale Linie
        mean_vdd = self.data["VddVoltage"].mean() / 1000
        ax1.axhline(y=mean_vdd, color='blue', linestyle='--', alpha=0.7, 
                   label=f'Durchschnitt: {mean_vdd:.3f} V')
        
        ax1.legend()
        
        # Batteriespannung
        ax2.plot(self.data["RelativeTime"], self.data["BatteryVoltage"] / 1000, 
                marker='o', linestyle='-', alpha=0.7, color="#FB8C00")
        
        ax2.set_title("Batteriespannung im Zeitverlauf", fontsize=16)
        ax2.set_xlabel("Zeit seit Start (Sekunden)", fontsize=12)
        ax2.set_ylabel("Spannung (V)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Durchschnitt als horizontale Linie
        mean_battery = self.data["BatteryVoltage"].mean() / 1000
        ax2.axhline(y=mean_battery, color='blue', linestyle='--', alpha=0.7, 
                   label=f'Durchschnitt: {mean_battery:.3f} V')
        
        # Kritische Spannung als Referenz (3.2V für LiPo)
        ax2.axhline(y=3.2, color='red', linestyle='-.', alpha=0.5,
                  label=f'Kritische Spannung: 3.2 V')
        
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "voltage_series.png", dpi=150)
        plt.close()
    
    def _plot_voltage_temperature_correlation(self):
        """Erzeugt ein Streudiagramm zur Analyse der Korrelation zwischen Spannung und Temperatur"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # VDD vs Temperatur
        sns.scatterplot(x="Temperature", y="VddVoltage", data=self.data, ax=ax1, alpha=0.7, 
                       hue="CpuLoad", palette="viridis", size="InferenceTime", sizes=(30, 200))
        
        ax1.set_title("VDD-Spannung vs. Temperatur", fontsize=16)
        ax1.set_xlabel("Temperatur (°C)", fontsize=12)
        ax1.set_ylabel("VDD-Spannung (mV)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Batterie vs Temperatur
        sns.scatterplot(x="Temperature", y="BatteryVoltage", data=self.data, ax=ax2, alpha=0.7,
                       hue="CpuLoad", palette="viridis", size="InferenceTime", sizes=(30, 200))
        
        ax2.set_title("Batteriespannung vs. Temperatur", fontsize=16)
        ax2.set_xlabel("Temperatur (°C)", fontsize=12)
        ax2.set_ylabel("Batteriespannung (mV)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "voltage_temperature_correlation.png", dpi=150)
        plt.close()
    
    def _plot_voltage_series(self):
        """Erzeugt ein Liniendiagramm der Spannungen über die Zeit"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # VDD-Spannung
        ax1.plot(self.data["RelativeTime"], self.data["VddVoltage"] / 1000, 
                marker='o', linestyle='-', alpha=0.7, color="#E53935")
        
        ax1.set_title("VDD-Spannung im Zeitverlauf", fontsize=16)
        ax1.set_ylabel("Spannung (V)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Durchschnitt als horizontale Linie
        mean_vdd = self.data["VddVoltage"].mean() / 1000
        ax1.axhline(y=mean_vdd, color='blue', linestyle='--', alpha=0.7, 
                   label=f'Durchschnitt: {mean_vdd:.3f} V')
        
        ax1.legend()
        
        # Batteriespannung
        ax2.plot(self.data["RelativeTime"], self.data["BatteryVoltage"] / 1000, 
                marker='o', linestyle='-', alpha=0.7, color="#FB8C00")
        
        ax2.set_title("Batteriespannung im Zeitverlauf", fontsize=16)
        ax2.set_xlabel("Zeit seit Start (Sekunden)", fontsize=12)
        ax2.set_ylabel("Spannung (V)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Durchschnitt als horizontale Linie
        mean_battery = self.data["BatteryVoltage"].mean() / 1000
        ax2.axhline(y=mean_battery, color='blue', linestyle='--', alpha=0.7, 
                   label=f'Durchschnitt: {mean_battery:.3f} V')
        
        # Kritische Spannung als Referenz (3.2V für LiPo)
        ax2.axhline(y=3.2, color='red', linestyle='-.', alpha=0.5,
                  label=f'Kritische Spannung: 3.2 V')
        
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "voltage_series.png", dpi=150)
        plt.close()
    
    def _plot_voltage_temperature_correlation(self):
        """Erzeugt ein Streudiagramm zur Analyse der Korrelation zwischen Spannung und Temperatur"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # VDD vs Temperatur
        sns.scatterplot(x="Temperature", y="VddVoltage", data=self.data, ax=ax1, alpha=0.7, 
                       hue="CpuLoad", palette="viridis", size="InferenceTime", sizes=(30, 200))
        
        ax1.set_title("VDD-Spannung vs. Temperatur", fontsize=16)
        ax1.set_xlabel("Temperatur (°C)", fontsize=12)
        ax1.set_ylabel("VDD-Spannung (mV)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Batterie vs Temperatur
        sns.scatterplot(x="Temperature", y="BatteryVoltage", data=self.data, ax=ax2, alpha=0.7,
                       hue="CpuLoad", palette="viridis", size="InferenceTime", sizes=(30, 200))
        
        ax2.set_title("Batteriespannung vs. Temperatur", fontsize=16)
        ax2.set_xlabel("Temperatur (°C)", fontsize=12)
        ax2.set_ylabel("Batteriespannung (mV)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "voltage_temperature_correlation.png", dpi=150)
        plt.close()
    
    def _create_dashboard(self):
        """Erstellt ein Dashboard mit den wichtigsten Metriken"""
        fig = plt.figure(figsize=(15, 12))  # Größer für mehr Plots
        gs = fig.add_gridspec(4, 3)  # 4 Zeilen statt 3
        
        # 1. Inferenzzeit-Verlauf
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.data["RelativeTime"], self.data["InferenceTime"] / 1000, 
                color="#1E88E5", marker='o', markersize=3, linestyle='-', alpha=0.7)
        ax1.set_title("Inferenzzeit (ms)", fontsize=12)
        ax1.set_xlabel("Zeit (s)", fontsize=10)
        ax1.tick_params(labelsize=8)
        
        # 2. RAM-Nutzung
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(self.data["RelativeTime"], self.data["PeakRamUsage"] / 1024, 
                color="#43A047", marker='o', markersize=3, linestyle='-', alpha=0.7)
        ax2.set_title("RAM-Nutzung (KB)", fontsize=12)
        ax2.set_xlabel("Zeit (s)", fontsize=10)
        ax2.tick_params(labelsize=8)
        
        # 3. CPU-Auslastung
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(self.data["RelativeTime"], self.data["CpuLoad"] * 100, 
                color="#FFA000", marker='o', markersize=3, linestyle='-', alpha=0.7)
        ax3.set_title("CPU-Auslastung (%)", fontsize=12)
        ax3.set_xlabel("Zeit (s)", fontsize=10)
        ax3.tick_params(labelsize=8)
        
        # 4. Temperatur
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(self.data["RelativeTime"], self.data["Temperature"], 
                color="#E53935", marker='o', markersize=3, linestyle='-', alpha=0.7)
        ax4.set_title("Temperatur (°C)", fontsize=12)
        ax4.set_xlabel("Zeit (s)", fontsize=10)
        ax4.tick_params(labelsize=8)
        
        # 5. VDD-Spannung
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(self.data["RelativeTime"], self.data["VddVoltage"] / 1000, 
                color="#673AB7", marker='o', markersize=3, linestyle='-', alpha=0.7)
        ax5.set_title("VDD-Spannung (V)", fontsize=12)
        ax5.set_xlabel("Zeit (s)", fontsize=10)
        ax5.tick_params(labelsize=8)
        ax5.axhline(y=3.3, color='r', linestyle='--', alpha=0.5)  # Referenzlinie bei 3.3V
        
        # 6. Batteriespannung
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(self.data["RelativeTime"], self.data["BatteryVoltage"] / 1000, 
                color="#FF9800", marker='o', markersize=3, linestyle='-', alpha=0.7)
        ax6.set_title("Batteriespannung (V)", fontsize=12)
        ax6.set_xlabel("Zeit (s)", fontsize=10)
        ax6.tick_params(labelsize=8)
        ax6.axhline(y=3.7, color='g', linestyle='--', alpha=0.5)  # Referenzlinie bei 3.7V
        ax6.axhline(y=3.2, color='r', linestyle='--', alpha=0.5)  # Kritische Linie bei 3.2V
        
        # 7. Vorhersageverteilung
        ax7 = fig.add_subplot(gs[2, 2])
        class_counts = self.data["PredictionClass"].value_counts()
        class_counts = class_counts.reindex(self.class_names)
        class_counts = class_counts.fillna(0)
        ax7.bar(class_counts.index, class_counts.values, color=sns.color_palette("viridis", len(class_counts)))
        ax7.set_title("Vorhersageverteilung", fontsize=12)
        ax7.set_xticklabels(class_counts.index, rotation=45, ha='right', fontsize=8)
        ax7.tick_params(labelsize=8)
        
        # 8. Konfidenz-Histogramm
        ax8 = fig.add_subplot(gs[3, :])
        sns.histplot(self.data["Confidence"], bins=15, kde=True, ax=ax8, color="#7E57C2")
        ax8.set_title("Konfidenzverteilung", fontsize=12)
        ax8.set_xlabel("Konfidenz", fontsize=10)
        ax8.tick_params(labelsize=8)
        
        # Titel und Layout
        fig.suptitle("Pizza-Erkennungssystem: Performance-Dashboard", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.output_dir / "performance_dashboard.png", dpi=200)
        plt.close()
    
    def generate_report(self, stats: Dict) -> str:
        """
        Erzeugt einen HTML-Bericht mit den Analyseergebnissen
        
        Args:
            stats: Dictionary mit den Analyseergebnissen
        
        Returns:
            Pfad zum generierten HTML-Bericht
        """
        if not stats:
            logger.error("Keine Statistiken zum Erstellen des Berichts verfügbar")
            return ""
        
        logger.info("Erstelle HTML-Bericht...")
        
        # Datum und Uhrzeit
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # HTML-Template
        html = f"""<!DOCTYPE html>
        <html>
        <head>
            <title>Performance-Analyse: Pizza-Erkennungssystem</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; color: #333; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                h1 {{ border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #e1f5fe; }}
                .highlight {{ font-weight: bold; color: #e74c3c; }}
                .good {{ color: #27ae60; }}
                .warning {{ color: #f39c12; }}
                .danger {{ color: #c0392b; }}
                .gallery {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin: 20px 0; }}
                .gallery img {{ max-width: 45%; border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                footer {{ text-align: center; margin-top: 50px; color: #7f8c8d; font-size: 0.8em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Performance-Analyse: Pizza-Erkennungssystem</h1>
                
                <div class="summary">
                    <h2>Zusammenfassung</h2>
                    <p>
                        Diese Analyse basiert auf {stats['total_entries']} Messwerten über eine Gesamtdauer von 
                        {stats['total_duration_ms']/1000:.2f} Sekunden.
                    </p>
                    <p>
                        <strong>Durchschnittliche Inferenzzeit:</strong> 
                        <span class="{self._get_inference_time_class(stats['inference_time']['avg_us'])}">{stats['inference_time']['avg_us']/1000:.2f} ms</span>
                    </p>
                    <p>
                        <strong>Maximaler RAM-Verbrauch:</strong> 
                        <span class="{self._get_ram_usage_class(stats['ram_usage']['max_bytes'])}">{stats['ram_usage']['max_bytes']/1024:.2f} KB</span>
                    </p>
                    <p>
                        <strong>Durchschnittliche CPU-Auslastung:</strong> 
                        <span class="{self._get_cpu_load_class(stats['cpu_load']['avg_percent'])}">{stats['cpu_load']['avg_percent']:.2f}%</span>
                    </p>
                    <p>
                        <strong>Durchschnittliche Temperatur:</strong> 
                        <span class="{self._get_temperature_class(stats['temperature']['avg_c'])}">{stats['temperature']['avg_c']:.2f}°C</span>
                    </p>
                    <p>
                        <strong>VDD-Spannung:</strong> 
                        <span class="{self._get_vdd_voltage_class(stats['vdd_voltage']['avg_mv'])}">{stats['vdd_voltage']['avg_mv']/1000:.3f}V</span>
                    </p>
                    <p>
                        <strong>Batteriespannung:</strong> 
                        <span class="{self._get_battery_voltage_class(stats['battery_voltage']['avg_mv'])}">{stats['battery_voltage']['avg_mv']/1000:.3f}V</span>
                    </p>
                </div>
                
                <h2>Inferenzzeit</h2>
                <table>
                    <tr>
                        <th>Metrik</th>
                        <th>Wert</th>
                        <th>Bewertung</th>
                    </tr>
                    <tr>
                        <td>Minimum</td>
                        <td>{stats['inference_time']['min_us']/1000:.2f} ms</td>
                        <td class="{self._get_inference_time_class(stats['inference_time']['min_us'])}">{self._get_inference_time_rating(stats['inference_time']['min_us'])}</td>
                    </tr>
                    <tr>
                        <td>Maximum</td>
                        <td>{stats['inference_time']['max_us']/1000:.2f} ms</td>
                        <td class="{self._get_inference_time_class(stats['inference_time']['max_us'])}">{self._get_inference_time_rating(stats['inference_time']['max_us'])}</td>
                    </tr>
                    <tr>
                        <td>Durchschnitt</td>
                        <td>{stats['inference_time']['avg_us']/1000:.2f} ms</td>
                        <td class="{self._get_inference_time_class(stats['inference_time']['avg_us'])}">{self._get_inference_time_rating(stats['inference_time']['avg_us'])}</td>
                    </tr>
                    <tr>
                        <td>Median</td>
                        <td>{stats['inference_time']['median_us']/1000:.2f} ms</td>
                        <td class="{self._get_inference_time_class(stats['inference_time']['median_us'])}">{self._get_inference_time_rating(stats['inference_time']['median_us'])}</td>
                    </tr>
                    <tr>
                        <td>Standardabweichung</td>
                        <td>{stats['inference_time']['std_us']/1000:.2f} ms</td>
                        <td>{self._get_std_rating(stats['inference_time']['std_us'], stats['inference_time']['avg_us'])}</td>
                    </tr>
                </table>
                
                <h2>RAM-Nutzung</h2>
                <table>
                    <tr>
                        <th>Metrik</th>
                        <th>Wert</th>
                        <th>Bewertung</th>
                    </tr>
                    <tr>
                        <td>Minimum</td>
                        <td>{stats['ram_usage']['min_bytes']/1024:.2f} KB</td>
                        <td class="{self._get_ram_usage_class(stats['ram_usage']['min_bytes'])}">{self._get_ram_usage_rating(stats['ram_usage']['min_bytes'])}</td>
                    </tr>
                    <tr>
                        <td>Maximum</td>
                        <td>{stats['ram_usage']['max_bytes']/1024:.2f} KB</td>
                        <td class="{self._get_ram_usage_class(stats['ram_usage']['max_bytes'])}">{self._get_ram_usage_rating(stats['ram_usage']['max_bytes'])}</td>
                    </tr>
                    <tr>
                        <td>Durchschnitt</td>
                        <td>{stats['ram_usage']['avg_bytes']/1024:.2f} KB</td>
                        <td class="{self._get_ram_usage_class(stats['ram_usage']['avg_bytes'])}">{self._get_ram_usage_rating(stats['ram_usage']['avg_bytes'])}</td>
                    </tr>
                    <tr>
                        <td>Median</td>
                        <td>{stats['ram_usage']['median_bytes']/1024:.2f} KB</td>
                        <td class="{self._get_ram_usage_class(stats['ram_usage']['median_bytes'])}">{self._get_ram_usage_rating(stats['ram_usage']['median_bytes'])}</td>
                    </tr>
                    <tr>
                        <td>Verfügbarer RAM (gesamt)</td>
                        <td>204 KB</td>
                        <td>Referenzwert</td>
                    </tr>
                    <tr>
                        <td>RAM-Auslastung</td>
                        <td>{stats['ram_usage']['max_bytes']/204/1024*100:.2f}%</td>
                        <td class="{self._get_ram_percentage_class(stats['ram_usage']['max_bytes']/204/1024*100)}">{self._get_ram_percentage_rating(stats['ram_usage']['max_bytes']/204/1024*100)}</td>
                    </tr>
                </table>
                
                <h2>CPU und Temperatur</h2>
                <table>
                    <tr>
                        <th>Metrik</th>
                        <th>Wert</th>
                        <th>Bewertung</th>
                    </tr>
                    <tr>
                        <td>Min. CPU-Auslastung</td>
                        <td>{stats['cpu_load']['min_percent']:.2f}%</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td>Max. CPU-Auslastung</td>
                        <td>{stats['cpu_load']['max_percent']:.2f}%</td>
                        <td class="{self._get_cpu_load_class(stats['cpu_load']['max_percent'])}">{self._get_cpu_load_rating(stats['cpu_load']['max_percent'])}</td>
                    </tr>
                    <tr>
                        <td>Durchschn. CPU-Auslastung</td>
                        <td>{stats['cpu_load']['avg_percent']:.2f}%</td>
                        <td class="{self._get_cpu_load_class(stats['cpu_load']['avg_percent'])}">{self._get_cpu_load_rating(stats['cpu_load']['avg_percent'])}</td>
                    </tr>
                    <tr>
                        <td>Min. Temperatur</td>
                        <td>{stats['temperature']['min_c']:.2f}°C</td>
                        <td class="{self._get_temperature_class(stats['temperature']['min_c'])}">{self._get_temperature_rating(stats['temperature']['min_c'])}</td>
                    </tr>
                    <tr>
                        <td>Max. Temperatur</td>
                        <td>{stats['temperature']['max_c']:.2f}°C</td>
                        <td class="{self._get_temperature_class(stats['temperature']['max_c'])}">{self._get_temperature_rating(stats['temperature']['max_c'])}</td>
                    </tr>
                    <tr>
                        <td>Durchschn. Temperatur</td>
                        <td>{stats['temperature']['avg_c']:.2f}°C</td>
                        <td class="{self._get_temperature_class(stats['temperature']['avg_c'])}">{self._get_temperature_rating(stats['temperature']['avg_c'])}</td>
                    </tr>
                </table>
                
                <h2>Spannungen</h2>
                <table>
                    <tr>
                        <th>Metrik</th>
                        <th>Wert</th>
                        <th>Bewertung</th>
                    </tr>
                    <tr>
                        <td>Min. VDD-Spannung</td>
                        <td>{stats['vdd_voltage']['min_mv']/1000:.3f} V</td>
                        <td class="{self._get_vdd_voltage_class(stats['vdd_voltage']['min_mv'])}">{self._get_vdd_voltage_rating(stats['vdd_voltage']['min_mv'])}</td>
                    </tr>
                    <tr>
                        <td>Max. VDD-Spannung</td>
                        <td>{stats['vdd_voltage']['max_mv']/1000:.3f} V</td>
                        <td class="{self._get_vdd_voltage_class(stats['vdd_voltage']['max_mv'])}">{self._get_vdd_voltage_rating(stats['vdd_voltage']['max_mv'])}</td>
                    </tr>
                    <tr>
                        <td>Durchschn. VDD-Spannung</td>
                        <td>{stats['vdd_voltage']['avg_mv']/1000:.3f} V</td>
                        <td class="{self._get_vdd_voltage_class(stats['vdd_voltage']['avg_mv'])}">{self._get_vdd_voltage_rating(stats['vdd_voltage']['avg_mv'])}</td>
                    </tr>
                    <tr>
                        <td>Min. Batteriespannung</td>
                        <td>{stats['battery_voltage']['min_mv']/1000:.3f} V</td>
                        <td class="{self._get_battery_voltage_class(stats['battery_voltage']['min_mv'])}">{self._get_battery_voltage_rating(stats['battery_voltage']['min_mv'])}</td>
                    </tr>
                    <tr>
                        <td>Max. Batteriespannung</td>
                        <td>{stats['battery_voltage']['max_mv']/1000:.3f} V</td>
                        <td class="{self._get_battery_voltage_class(stats['battery_voltage']['max_mv'])}">{self._get_battery_voltage_rating(stats['battery_voltage']['max_mv'])}</td>
                    </tr>
                    <tr>
                        <td>Durchschn. Batteriespannung</td>
                        <td>{stats['battery_voltage']['avg_mv']/1000:.3f} V</td>
                        <td class="{self._get_battery_voltage_class(stats['battery_voltage']['avg_mv'])}">{self._get_battery_voltage_rating(stats['battery_voltage']['avg_mv'])}</td>
                    </tr>
                </table>
                
                <h2>Vorhersagen</h2>
                <table>
                    <tr>
                        <th>Klasse</th>
                        <th>Anzahl</th>
                        <th>Prozent</th>
                    </tr>
        """
        
        # Vorhersage-Verteilung hinzufügen
        total_predictions = stats['total_entries']
        for class_name, count in stats['predictions']['distribution'].items():
            percentage = (count / total_predictions) * 100
            html += f"""
                    <tr>
                        <td>{class_name}</td>
                        <td>{count}</td>
                        <td>{percentage:.2f}%</td>
                    </tr>
            """
        
        html += f"""
                </table>
                <p>Durchschnittliche Konfidenz: <strong>{stats['predictions']['avg_confidence']:.2f}%</strong></p>
                
                <h2>Visualisierungen</h2>
                <div class="gallery">
                    <img src="inference_time_series.png" alt="Inferenzzeit im Zeitverlauf">
                    <img src="ram_usage_series.png" alt="RAM-Nutzung im Zeitverlauf">
                    <img src="inference_time_histogram.png" alt="Verteilung der Inferenzzeiten">
                    <img src="prediction_distribution.png" alt="Verteilung der Vorhersageklassen">
                </div>
                
                <div class="gallery">
                    <img src="metrics_correlation.png" alt="Korrelation zwischen Metriken">
                    <img src="performance_dashboard.png" alt="Performance-Dashboard">
                </div>
                
                <div class="gallery">
                    <img src="voltage_series.png" alt="Spannungen im Zeitverlauf">
                    <img src="voltage_temperature_correlation.png" alt="Korrelation zwischen Spannung und Temperatur">
                </div>
                
                <h2>Empfehlungen</h2>
                {self._generate_recommendations(stats)}
                
                <footer>
                    <p>Pizza-Erkennungssystem Performance-Analyse | Erstellt am {now}</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        # HTML-Datei speichern
        report_path = self.output_dir / "performance_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"HTML-Bericht erstellt: {report_path}")
        return str(report_path)
    
    def _generate_recommendations(self, stats: Dict) -> str:
        """Generiert Empfehlungen basierend auf den Analyseergebnissen"""
        recommendations = ["<ul>"]
        
        # Inferenzzeit
        avg_inference_ms = stats['inference_time']['avg_us'] / 1000
        if avg_inference_ms > 100:
            recommendations.append(f"<li class='danger'>Die durchschnittliche Inferenzzeit von {avg_inference_ms:.2f} ms ist sehr hoch. "
                               "Überprüfen Sie das Modell auf Komplexität und erwägen Sie weitere Optimierungen.</li>")
        elif avg_inference_ms > 50:
            recommendations.append(f"<li class='warning'>Die durchschnittliche Inferenzzeit von {avg_inference_ms:.2f} ms könnte verbessert werden. "
                               "Erwägen Sie weitere Quantisierung oder Hardware-Beschleunigung.</li>")
        
        # RAM-Nutzung
        max_ram_kb = stats['ram_usage']['max_bytes'] / 1024
        ram_percentage = max_ram_kb / 204 * 100  # 204 KB verfügbar laut Projektdokumentation
        
        if ram_percentage > 90:
            recommendations.append(f"<li class='danger'>Die maximale RAM-Nutzung von {max_ram_kb:.2f} KB ({ram_percentage:.2f}%) ist kritisch hoch. "
                               "Das System steht kurz vor einem Speicherüberlauf. Reduzieren Sie die Modellgröße oder Bildauflösung.</li>")
        elif ram_percentage > 80:
            recommendations.append(f"<li class='warning'>Die RAM-Nutzung von {max_ram_kb:.2f} KB ({ram_percentage:.2f}%) ist hoch. "
                               "Erwägen Sie Optimierungen, um mehr Puffer für unvorhergesehene Situationen zu haben.</li>")
        
        # Temperatur
        avg_temp = stats['temperature']['avg_c']
        max_temp = stats['temperature']['max_c']
        
        if max_temp > 60:
            recommendations.append(f"<li class='danger'>Die maximale Temperatur von {max_temp:.2f}°C ist kritisch hoch. "
                               "Überprüfen Sie die Kühlung und reduzieren Sie die Taktrate.</li>")
        elif avg_temp > 50:
            recommendations.append(f"<li class='warning'>Die durchschnittliche Temperatur von {avg_temp:.2f}°C ist erhöht. "
                               "Dies könnte bei längeren Betriebszeiten zu Problemen führen.</li>")
        
        # Spannungen
        min_battery_v = stats['battery_voltage']['min_mv'] / 1000
        if min_battery_v < 3.3:
            recommendations.append(f"<li class='danger'>Die minimale Batteriespannung von {min_battery_v:.2f}V ist kritisch niedrig. "
                               "Die Batterie sollte umgehend geladen oder ausgetauscht werden.</li>")
        elif min_battery_v < 3.5:
            recommendations.append(f"<li class='warning'>Die Batteriespannung von {min_battery_v:.2f}V ist niedrig. "
                               "Die Batterie sollte bald geladen werden.</li>")
        
        # VDD Spannung
        avg_vdd_v = stats['vdd_voltage']['avg_mv'] / 1000
        if avg_vdd_v < 3.0 or avg_vdd_v > 3.6:
            recommendations.append(f"<li class='warning'>Die VDD-Spannung von {avg_vdd_v:.2f}V liegt außerhalb des empfohlenen Bereichs (3.0-3.6V). "
                               "Dies könnte die Stabilität des Systems beeinträchtigen.</li>")
        
        # CPU-Auslastung
        avg_cpu = stats['cpu_load']['avg_percent']
        if avg_cpu > 90:
            recommendations.append(f"<li class='danger'>Die durchschnittliche CPU-Auslastung von {avg_cpu:.2f}% ist sehr hoch. "
                               "Das System hat kaum Reserven für weitere Aufgaben.</li>")
        elif avg_cpu > 70:
            recommendations.append(f"<li class='warning'>Die CPU-Auslastung von {avg_cpu:.2f}% ist erhöht. "
                               "Dies könnte zu erhöhtem Stromverbrauch und kürzerer Batterielebensdauer führen.</li>")
        
        # Standardabweichung der Inferenzzeit
        std_percentage = (stats['inference_time']['std_us'] / stats['inference_time']['avg_us']) * 100
        if std_percentage > 30:
            recommendations.append(f"<li class='warning'>Die hohe Variabilität der Inferenzzeit (Standardabweichung = {std_percentage:.2f}% des Mittelwerts) "
                               "deutet auf Inkonsistenzen in der Verarbeitung hin. Prüfen Sie auf Hintergrundprozesse oder dynamisches Frequenzskalieren.</li>")
        
        # Wenn keine spezifischen Probleme gefunden wurden
        if len(recommendations) == 1:
            recommendations.append("<li class='good'>Alle Performance-Metriken liegen im optimalen Bereich. "
                               "Das System ist gut ausbalanciert und funktioniert effizient.</li>")
        else:
            # Füge positives Feedback hinzu, falls vorhanden
            if ram_percentage < 60:
                recommendations.append(f"<li class='good'>Die RAM-Nutzung von {max_ram_kb:.2f} KB ({ram_percentage:.2f}%) "
                                   "bietet ausreichend Puffer für zukünftige Erweiterungen.</li>")
            
            if avg_inference_ms < 30:
                recommendations.append(f"<li class='good'>Die durchschnittliche Inferenzzeit von {avg_inference_ms:.2f} ms "
                                   "ist sehr gut für ein Embedded-System.</li>")
        
        recommendations.append("</ul>")
        return "\n".join(recommendations)
    
    def _get_inference_time_class(self, time_us: float) -> str:
        """Bestimmt die CSS-Klasse basierend auf der Inferenzzeit"""
        time_ms = time_us / 1000
        if time_ms < 30:
            return "good"
        elif time_ms < 80:
            return "warning"
        else:
            return "danger"
    
    def _get_inference_time_rating(self, time_us: float) -> str:
        """Bewertet die Inferenzzeit"""
        time_ms = time_us / 1000
        if time_ms < 30:
            return "Ausgezeichnet"
        elif time_ms < 50:
            return "Gut"
        elif time_ms < 80:
            return "Akzeptabel"
        elif time_ms < 100:
            return "Langsam"
        else:
            return "Kritisch langsam"
    
    def _get_ram_usage_class(self, bytes_used: float) -> str:
        """Bestimmt die CSS-Klasse basierend auf dem RAM-Verbrauch"""
        kb_used = bytes_used / 1024
        total_kb = 204  # Laut Projektdokumentation: 204KB verfügbar
        percentage = (kb_used / total_kb) * 100
        
        if percentage < 60:
            return "good"
        elif percentage < 80:
            return "warning"
        else:
            return "danger"
    
    def _get_ram_usage_rating(self, bytes_used: float) -> str:
        """Bewertet den RAM-Verbrauch"""
        kb_used = bytes_used / 1024
        total_kb = 204  # Laut Projektdokumentation
        percentage = (kb_used / total_kb) * 100
        
        if percentage < 50:
            return "Niedrig"
        elif percentage < 70:
            return "Moderat"
        elif percentage < 85:
            return "Hoch"
        elif percentage < 95:
            return "Sehr hoch"
        else:
            return "Kritisch"
    
    def _get_ram_percentage_class(self, percentage: float) -> str:
        """Bestimmt die CSS-Klasse basierend auf dem RAM-Prozentsatz"""
        if percentage < 60:
            return "good"
        elif percentage < 80:
            return "warning"
        else:
            return "danger"
    
    def _get_ram_percentage_rating(self, percentage: float) -> str:
        """Bewertet den RAM-Prozentsatz"""
        if percentage < 50:
            return "Viel Puffer"
        elif percentage < 70:
            return "Ausreichend Puffer"
        elif percentage < 85:
            return "Wenig Puffer"
        elif percentage < 95:
            return "Kaum Puffer"
        else:
            return "Kritisch - Überlauf möglich"
    
    def _get_cpu_load_class(self, percentage: float) -> str:
        """Bestimmt die CSS-Klasse basierend auf der CPU-Auslastung"""
        if percentage < 60:
            return "good"
        elif percentage < 85:
            return "warning"
        else:
            return "danger"
    
    def _get_cpu_load_rating(self, percentage: float) -> str:
        """Bewertet die CPU-Auslastung"""
        if percentage < 40:
            return "Niedrig"
        elif percentage < 60:
            return "Moderat"
        elif percentage < 80:
            return "Hoch"
        elif percentage < 95:
            return "Sehr hoch"
        else:
            return "Kritisch"
    
    def _get_temperature_class(self, temp_c: float) -> str:
        """Bestimmt die CSS-Klasse basierend auf der Temperatur"""
        if temp_c < 45:
            return "good"
        elif temp_c < 60:
            return "warning"
        else:
            return "danger"
    
    def _get_temperature_rating(self, temp_c: float) -> str:
        """Bewertet die Temperatur"""
        if temp_c < 35:
            return "Optimal"
        elif temp_c < 45:
            return "Normal"
        elif temp_c < 55:
            return "Erhöht"
        elif temp_c < 65:
            return "Hoch"
        else:
            return "Kritisch"
    
    def _get_vdd_voltage_class(self, voltage_mv: float) -> str:
        """Bestimmt die CSS-Klasse basierend auf der VDD-Spannung"""
        voltage_v = voltage_mv / 1000
        if 3.0 <= voltage_v <= 3.6:  # Typischer Bereich für 3.3V Systeme
            return "good"
        elif 2.8 <= voltage_v < 3.0 or 3.6 < voltage_v <= 3.8:
            return "warning"
        else:
            return "danger"
    
    def _get_vdd_voltage_rating(self, voltage_mv: float) -> str:
        """Bewertet die VDD-Spannung"""
        voltage_v = voltage_mv / 1000
        if 3.2 <= voltage_v <= 3.4:  # Optimaler Bereich um 3.3V
            return "Optimal"
        elif 3.0 <= voltage_v < 3.2 or 3.4 < voltage_v <= 3.6:
            return "Normal"
        elif 2.8 <= voltage_v < 3.0 or 3.6 < voltage_v <= 3.8:
            return "Grenzwertig"
        else:
            return "Kritisch"
    
    def _get_battery_voltage_class(self, voltage_mv: float) -> str:
        """Bestimmt die CSS-Klasse basierend auf der Batteriespannung (LiPo)"""
        voltage_v = voltage_mv / 1000
        if voltage_v >= 3.7:  # Gut geladen
            return "good"
        elif 3.3 <= voltage_v < 3.7:  # Teilweise entladen
            return "warning"
        else:  # Fast leer
            return "danger"
    
    def _get_battery_voltage_rating(self, voltage_mv: float) -> str:
        """Bewertet die Batteriespannung (LiPo)"""
        voltage_v = voltage_mv / 1000
        if voltage_v >= 4.0:
            return "Voll geladen"
        elif 3.7 <= voltage_v < 4.0:
            return "Gut geladen"
        elif 3.5 <= voltage_v < 3.7:
            return "Teilweise entladen"
        elif 3.3 <= voltage_v < 3.5:
            return "Niedrig"
        elif 3.2 <= voltage_v < 3.3:
            return "Sehr niedrig"
        else:
            return "Kritisch niedrig"
    
    def _get_std_rating(self, std_value: float, avg_value: float) -> str:
        """Bewertet die Standardabweichung relativ zum Durchschnitt"""
        percentage = (std_value / avg_value) * 100
        
        if percentage < 10:
            return "<span class='good'>Sehr konsistent</span>"
        elif percentage < 20:
            return "<span class='good'>Konsistent</span>"
        elif percentage < 30:
            return "<span class='warning'>Leicht variabel</span>"
        elif percentage < 50:
            return "<span class='warning'>Variabel</span>"
        else:
            return "<span class='danger'>Stark variabel</span>"
    
    def run(self) -> str:
        """
        Führt die vollständige Analyse aus
        
        Returns:
            Pfad zum generierten HTML-Bericht
        """
        try:
            # Daten laden
            self.load_data()
            if self.data is None or self.data.empty:
                logger.error("Keine Daten zum Analysieren verfügbar")
                return ""
            
            # Daten analysieren
            stats = self.analyze()
            
            # Visualisierungen erstellen
            self.generate_visualizations()
            
            # Bericht erstellen
            report_path = self.generate_report(stats)
            
            logger.info(f"Analyse abgeschlossen. Bericht: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Fehler bei der Analyse: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""


def list_serial_ports():
    """Listet alle verfügbaren seriellen Ports auf"""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("Keine seriellen Ports gefunden.")
        return
    
    print("Verfügbare serielle Ports:")
    for port in ports:
        print(f"- {port.device}: {port.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse von Performance-Logs des Pizza-Erkennungssystems")
    parser.add_argument("--input", type=str, required=True, help="Pfad zur Log-Datei oder COM-Port (z.B. 'COM3', '/dev/ttyACM0')")
    parser.add_argument("--output", type=str, default="output/performance_analysis", help="Verzeichnis für die Ausgabe der Analysen")
    parser.add_argument("--list-ports", action="store_true", help="Listet alle verfügbaren seriellen Ports auf")
    parser.add_argument("--class-config", type=str, help="Pfad zur Konfigurationsdatei mit Klassennamen (JSON-Format)")
    
    args = parser.parse_args()
    
    if args.list_ports:
        list_serial_ports()
        sys.exit(0)
    
    # Überprüfe, ob die Konfigurationsdatei existiert, wenn ein Pfad angegeben wurde
    if args.class_config and not os.path.exists(args.class_config):
        print(f"Fehler: Die angegebene Konfigurationsdatei {args.class_config} existiert nicht.")
        sys.exit(1)
    
    try:
        analyzer = PerformanceLogAnalyzer(args.input, args.output, args.class_config)
        report_path = analyzer.run()
        
        if report_path:
            print(f"\nAnalyse abgeschlossen. Bericht unter: {report_path}")
            print(f"Bitte öffnen Sie den Bericht in einem Webbrowser, um die Ergebnisse zu sehen.")
        else:
            print("\nAnalyse fehlgeschlagen. Überprüfen Sie die Log-Ausgabe für Details.")
            sys.exit(1)
    except Exception as e:
        print(f"\nFehler bei der Ausführung: {e}")
        logger.error(f"Fehler bei der Ausführung: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)