#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energiemessdaten-Analyzer für das RP2040 Pizza-Erkennungssystem

Dieses Skript analysiert Energiemessdaten von externen Messgeräten (Power Analyzer, Oszilloskop)
und identifiziert verschiedene Betriebsmodi basierend auf Stromprofilen. Es berechnet 
durchschnittliche Stromverbräuche für jeden Modus und über repräsentative Duty-Cycle-Zeiträume.

Unterstützte Datenformate:
- CSV-Export von Power Analyzern (Keysight N6705C, etc.)
- CSV-Export von Oszilloskopen mit Shunt-Messungen
- Benutzerdefinierte CSV-Formate

Nutzung:
    python analyze_energy_measurements.py --input <Messdatei.csv> --output <Ausgabe-Verzeichnis>

Autor: Pizza Detection Team
Datum: 2025-05-24
Version: 1.0
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
from dataclasses import dataclass, asdict
from scipy import signal
from sklearn.cluster import KMeans
import warnings

# Konfiguration des Loggers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Ignore FutureWarnings from pandas/sklearn
warnings.filterwarnings("ignore", category=FutureWarning)

@dataclass
class OperatingMode:
    """Datenstruktur für einen Betriebsmodus."""
    name: str
    current_range_ma: Tuple[float, float]  # (min, max) in mA
    expected_duration_ms: Tuple[float, float]  # (min, max) in ms
    color: str  # Farbe für Visualisierung
    description: str

@dataclass
class MeasurementPoint:
    """Datenstruktur für einen Messpunkt nach dem Energiemessplan."""
    name: str
    description: str
    expected_range_ma: Tuple[float, float]
    csv_column: str  # Name der Spalte in der CSV-Datei

@dataclass
class EnergyAnalysisResult:
    """Ergebnis der Energieanalyse."""
    measurement_point: str
    total_duration_s: float
    operating_modes: Dict[str, Dict[str, float]]  # mode_name -> {duration_s, avg_current_ma, energy_mah}
    duty_cycle_analysis: Dict[str, float]  # Anteil jedes Modus in %
    average_power_consumption_mw: float
    peak_power_consumption_mw: float
    estimated_battery_life_hours: Dict[str, float]  # battery_type -> hours

class EnergyMeasurementAnalyzer:
    """Hauptklasse zur Analyse von Energiemessdaten."""
    
    def __init__(self, input_path: str, output_dir: str):
        """
        Initialisiert den Analyzer.
        
        Args:
            input_path: Pfad zur CSV-Messdatei
            output_dir: Verzeichnis für die Ausgabe
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Definiere die erwarteten Betriebsmodi basierend auf dem Energiemessplan
        self.operating_modes = {
            'sleep': OperatingMode(
                name='Deep Sleep',
                current_range_ma=(0.1, 1.0),
                expected_duration_ms=(30000, 300000),  # 30s - 5min
                color='blue',
                description='Sleep-Modus, sehr niedriger Stromverbrauch'
            ),
            'wakeup': OperatingMode(
                name='Wake-Up',
                current_range_ma=(5.0, 15.0),
                expected_duration_ms=(10, 50),
                color='green',
                description='Aufwachen aus Sleep-Modus'
            ),
            'camera_init': OperatingMode(
                name='Camera Init',
                current_range_ma=(15.0, 40.0),
                expected_duration_ms=(100, 200),
                color='orange',
                description='Kamera-Initialisierung'
            ),
            'image_capture': OperatingMode(
                name='Image Capture',
                current_range_ma=(40.0, 60.0),
                expected_duration_ms=(50, 100),
                color='red',
                description='Bildaufnahme'
            ),
            'inference': OperatingMode(
                name='Inference',
                current_range_ma=(150.0, 180.0),
                expected_duration_ms=(20, 50),
                color='purple',
                description='ML-Inferenz'
            ),
            'data_logging': OperatingMode(
                name='Data Logging',
                current_range_ma=(10.0, 20.0),
                expected_duration_ms=(10, 100),
                color='brown',
                description='Datenlogger-Betrieb'
            )
        }
        
        # Definiere die Messpunkte nach dem Energiemessplan
        self.measurement_points = {
            'MP1': MeasurementPoint(
                name='MP1: Gesamtsystem',
                description='Gesamter Systemstromverbrauch',
                expected_range_ma=(0.5, 200.0),
                csv_column='total_current_ma'
            ),
            'MP2': MeasurementPoint(
                name='MP2: Buck-Boost Eingang',
                description='Eingangsstrom zum Buck-Boost Regler',
                expected_range_ma=(0.5, 200.0),
                csv_column='input_current_ma'
            ),
            'MP3': MeasurementPoint(
                name='MP3: Buck-Boost Ausgang (3.3V)',
                description='3.3V Rail Stromverbrauch',
                expected_range_ma=(0.3, 150.0),
                csv_column='rail_3v3_current_ma'
            ),
            'MP4': MeasurementPoint(
                name='MP4: RP2040 MCU',
                description='MCU-spezifischer Stromverbrauch',
                expected_range_ma=(0.2, 120.0),
                csv_column='mcu_current_ma'
            ),
            'MP5': MeasurementPoint(
                name='MP5: OV2640 Kamera',
                description='Kamera-spezifischer Stromverbrauch',
                expected_range_ma=(0.1, 50.0),
                csv_column='camera_current_ma'
            ),
            'MP6': MeasurementPoint(
                name='MP6: I/O und Peripherie',
                description='I/O-Stromverbrauch',
                expected_range_ma=(0.05, 20.0),
                csv_column='io_current_ma'
            )
        }
        
        # Batterietype-Definitionen für Lebensdauer-Berechnungen
        self.battery_types = {
            'CR123A': {'capacity_mah': 1500, 'voltage_v': 3.0},
            'AA_Alkaline': {'capacity_mah': 2500, 'voltage_v': 1.5},
            '18650_LiIon': {'capacity_mah': 3400, 'voltage_v': 3.7},
            'LiPo_500mAh': {'capacity_mah': 500, 'voltage_v': 3.7}
        }
        
        self.data = None
        self.analysis_results = {}
        
    def load_measurement_data(self) -> bool:
        """
        Lädt die Messdaten aus der CSV-Datei.
        
        Returns:
            True wenn erfolgreich, False bei Fehler
        """
        try:
            logger.info(f"Lade Messdaten aus: {self.input_path}")
            
            # Versuche verschiedene CSV-Formate
            # Format 1: Standard-CSV mit Komma-Trennung
            try:
                self.data = pd.read_csv(self.input_path, sep=',')
            except:
                # Format 2: Semikolon-getrennt (häufig bei deutschen Geräten)
                try:
                    self.data = pd.read_csv(self.input_path, sep=';')
                except:
                    # Format 3: Tab-getrennt
                    self.data = pd.read_csv(self.input_path, sep='\t')
            
            logger.info(f"Daten geladen: {len(self.data)} Zeilen, {len(self.data.columns)} Spalten")
            logger.info(f"Spalten: {list(self.data.columns)}")
            
            # Automatische Spalten-Erkennung falls Standard-Namen nicht vorhanden
            self._auto_detect_columns()
            
            # Zeit-Spalte verarbeiten
            self._process_time_column()
            
            # Datenqualität prüfen
            self._validate_data()
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der Daten: {e}")
            return False
    
    def _auto_detect_columns(self):
        """Automatische Erkennung der relevanten Spalten."""
        columns = self.data.columns.str.lower()
        
        # Mapping für häufige Spaltennamen
        column_mappings = {
            'time': ['time', 'timestamp', 'zeit', 't'],
            'total_current_ma': ['current', 'i_total', 'total_current', 'strom', 'current_ma', 'i[ma]'],
            'voltage_v': ['voltage', 'v_battery', 'battery_voltage', 'spannung', 'voltage_v', 'u[v]'],
            'power_mw': ['power', 'leistung', 'power_mw', 'p[mw]']
        }
        
        detected_columns = {}
        for standard_name, possible_names in column_mappings.items():
            for col in columns:
                for possible in possible_names:
                    if possible in col:
                        detected_columns[standard_name] = self.data.columns[columns.get_loc(col)]
                        break
                if standard_name in detected_columns:
                    break
        
        # Füge erkannte Spalten zu den Messdaten hinzu
        if 'total_current_ma' in detected_columns:
            self.data['detected_current'] = self.data[detected_columns['total_current_ma']]
        
        logger.info(f"Automatisch erkannte Spalten: {detected_columns}")
    
    def _process_time_column(self):
        """Verarbeitet die Zeit-Spalte."""
        time_columns = ['time', 'timestamp', 'zeit', 't']
        time_col = None
        
        for col in time_columns:
            if col in self.data.columns.str.lower():
                time_col = self.data.columns[self.data.columns.str.lower().get_loc(col)]
                break
        
        if time_col is not None:
            try:
                # Versuche verschiedene Zeitformate
                self.data['timestamp'] = pd.to_datetime(self.data[time_col])
            except:
                # Falls Zeitstempel numerisch sind (z.B. Sekunden seit Start)
                try:
                    self.data['timestamp'] = pd.to_timedelta(self.data[time_col], unit='s')
                    self.data['timestamp'] = datetime.now() - timedelta(seconds=self.data[time_col].max()) + self.data['timestamp']
                except:
                    # Falls alles fehlschlägt, erstelle sequenzielle Zeitstempel
                    logger.warning("Konnte Zeitstempel nicht interpretieren, erstelle sequenzielle Zeitstempel")
                    self.data['timestamp'] = pd.date_range(start=datetime.now(), periods=len(self.data), freq='10ms')
        else:
            # Erstelle sequenzielle Zeitstempel falls keine Zeit-Spalte vorhanden
            logger.warning("Keine Zeit-Spalte gefunden, erstelle sequenzielle Zeitstempel")
            self.data['timestamp'] = pd.date_range(start=datetime.now(), periods=len(self.data), freq='10ms')
    
    def _validate_data(self):
        """Überprüft die Datenqualität."""
        if 'detected_current' not in self.data.columns:
            raise ValueError("Keine Strom-Spalte gefunden oder erkannt")
        
        # Prüfe auf fehlende Werte
        if self.data['detected_current'].isna().sum() > 0:
            logger.warning(f"{self.data['detected_current'].isna().sum()} fehlende Stromwerte gefunden")
            self.data['detected_current'].fillna(method='forward', inplace=True)
        
        # Prüfe Wertebereich
        current_min = self.data['detected_current'].min()
        current_max = self.data['detected_current'].max()
        logger.info(f"Strombereich: {current_min:.3f} bis {current_max:.3f} mA")
        
        if current_max > 1000:  # Wahrscheinlich in µA statt mA
            logger.info("Konvertiere µA zu mA")
            self.data['detected_current'] = self.data['detected_current'] / 1000
    
    def detect_operating_modes(self, current_column: str = 'detected_current') -> pd.Series:
        """
        Erkennt Betriebsmodi basierend auf Stromprofilen.
        
        Args:
            current_column: Name der Strom-Spalte
            
        Returns:
            Series mit den erkannten Modi für jeden Zeitpunkt
        """
        logger.info("Erkenne Betriebsmodi...")
        
        current_data = self.data[current_column].values
        modes = np.full(len(current_data), 'unknown', dtype=object)
        
        # Glättung der Daten zur Rauschreduzierung
        window_size = min(10, len(current_data) // 100)
        if window_size > 1:
            current_smooth = signal.savgol_filter(current_data, window_size, 3)
        else:
            current_smooth = current_data
        
        # Erkenne Modi basierend auf Stromwerten
        for mode_key, mode in self.operating_modes.items():
            min_current, max_current = mode.current_range_ma
            mask = (current_smooth >= min_current) & (current_smooth <= max_current)
            modes[mask] = mode_key
        
        # Verbessere die Erkennung durch Clustering für unbekannte Bereiche
        unknown_mask = modes == 'unknown'
        if unknown_mask.sum() > 0:
            self._cluster_unknown_modes(current_smooth, modes, unknown_mask)
        
        # Post-Processing: Entferne zu kurze Modi-Segmente
        modes = self._filter_short_segments(modes, min_duration_samples=5)
        
        return pd.Series(modes, index=self.data.index)
    
    def _cluster_unknown_modes(self, current_data: np.ndarray, modes: np.ndarray, unknown_mask: np.ndarray):
        """Verwendet K-Means Clustering für unbekannte Modi."""
        unknown_currents = current_data[unknown_mask].reshape(-1, 1)
        
        if len(unknown_currents) < 10:
            return
        
        # K-Means mit 3 Clustern für unbekannte Bereiche
        kmeans = KMeans(n_clusters=min(3, len(unknown_currents)), random_state=42, n_init=10)
        clusters = kmeans.fit_predict(unknown_currents)
        
        # Weise Cluster basierend auf Stromwerten zu
        for cluster_id in range(kmeans.n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_current = unknown_currents[cluster_mask].mean()
            
            # Finde den nächstgelegenen definierten Modus
            best_mode = 'unknown'
            min_distance = float('inf')
            
            for mode_key, mode in self.operating_modes.items():
                mode_center = (mode.current_range_ma[0] + mode.current_range_ma[1]) / 2
                distance = abs(cluster_current - mode_center)
                if distance < min_distance:
                    min_distance = distance
                    best_mode = mode_key
            
            # Aktualisiere Modi für diesen Cluster
            unknown_indices = np.where(unknown_mask)[0]
            cluster_indices = unknown_indices[cluster_mask]
            modes[cluster_indices] = best_mode
    
    def _filter_short_segments(self, modes: np.ndarray, min_duration_samples: int) -> np.ndarray:
        """Entfernt zu kurze Modi-Segmente."""
        filtered_modes = modes.copy()
        
        i = 0
        while i < len(modes):
            current_mode = modes[i]
            segment_start = i
            
            # Finde Ende des aktuellen Segments
            while i < len(modes) and modes[i] == current_mode:
                i += 1
            
            segment_length = i - segment_start
            
            # Wenn Segment zu kurz ist, weise den Modus der Nachbarn zu
            if segment_length < min_duration_samples:
                # Bestimme den häufigsten Nachbar-Modus
                neighbor_modes = []
                if segment_start > 0:
                    neighbor_modes.append(filtered_modes[segment_start - 1])
                if i < len(modes):
                    neighbor_modes.append(modes[i])
                
                if neighbor_modes:
                    # Wähle den häufigsten Nachbar-Modus
                    replacement_mode = max(set(neighbor_modes), key=neighbor_modes.count)
                    filtered_modes[segment_start:i] = replacement_mode
        
        return filtered_modes
    
    def analyze_duty_cycle(self, modes: pd.Series) -> Dict[str, float]:
        """
        Analysiert den Duty-Cycle der verschiedenen Modi.
        
        Args:
            modes: Series mit den erkannten Modi
            
        Returns:
            Dictionary mit dem Anteil jedes Modus in Prozent
        """
        logger.info("Analysiere Duty-Cycle...")
        
        total_samples = len(modes)
        duty_cycle = {}
        
        for mode_key in self.operating_modes.keys():
            mode_samples = (modes == mode_key).sum()
            duty_cycle[mode_key] = (mode_samples / total_samples) * 100
        
        # Füge unbekannte Modi hinzu
        unknown_samples = (modes == 'unknown').sum()
        if unknown_samples > 0:
            duty_cycle['unknown'] = (unknown_samples / total_samples) * 100
        
        return duty_cycle
    
    def calculate_mode_statistics(self, modes: pd.Series, current_column: str = 'detected_current') -> Dict[str, Dict[str, float]]:
        """
        Berechnet Statistiken für jeden erkannten Modus.
        
        Args:
            modes: Series mit den erkannten Modi
            current_column: Name der Strom-Spalte
            
        Returns:
            Dictionary mit Statistiken für jeden Modus
        """
        logger.info("Berechne Modus-Statistiken...")
        
        statistics = {}
        current_data = self.data[current_column]
        
        # Bestimme die Abtastrate
        time_diff = (self.data['timestamp'].iloc[-1] - self.data['timestamp'].iloc[0]).total_seconds()
        sample_rate = len(self.data) / time_diff if time_diff > 0 else 1000  # Hz
        
        for mode_key in self.operating_modes.keys():
            mode_mask = modes == mode_key
            mode_samples = mode_mask.sum()
            
            if mode_samples == 0:
                statistics[mode_key] = {
                    'duration_s': 0.0,
                    'avg_current_ma': 0.0,
                    'peak_current_ma': 0.0,
                    'energy_mah': 0.0,
                    'sample_count': 0
                }
                continue
            
            mode_current = current_data[mode_mask]
            duration_s = mode_samples / sample_rate
            avg_current = mode_current.mean()
            peak_current = mode_current.max()
            energy_mah = avg_current * (duration_s / 3600)  # mAh
            
            statistics[mode_key] = {
                'duration_s': duration_s,
                'avg_current_ma': avg_current,
                'peak_current_ma': peak_current,
                'energy_mah': energy_mah,
                'sample_count': mode_samples
            }
        
        return statistics
    
    def estimate_battery_life(self, mode_statistics: Dict[str, Dict[str, float]], duty_cycle: Dict[str, float]) -> Dict[str, float]:
        """
        Schätzt die Batterielebensdauer für verschiedene Batterietypen.
        
        Args:
            mode_statistics: Statistiken der verschiedenen Modi
            duty_cycle: Duty-Cycle Analyse
            
        Returns:
            Dictionary mit geschätzter Lebensdauer für jeden Batterietyp
        """
        logger.info("Schätze Batterielebensdauer...")
        
        # Berechne den durchschnittlichen Stromverbrauch basierend auf Duty-Cycle
        avg_current_ma = 0.0
        for mode_key, stats in mode_statistics.items():
            if mode_key in duty_cycle:
                avg_current_ma += stats['avg_current_ma'] * (duty_cycle[mode_key] / 100)
        
        battery_life = {}
        for battery_name, battery_specs in self.battery_types.items():
            if avg_current_ma > 0:
                life_hours = battery_specs['capacity_mah'] / avg_current_ma
                battery_life[battery_name] = life_hours
            else:
                battery_life[battery_name] = float('inf')
        
        return battery_life
    
    def analyze_measurement_point(self, mp_key: str) -> Optional[EnergyAnalysisResult]:
        """
        Analysiert einen spezifischen Messpunkt.
        
        Args:
            mp_key: Schlüssel des Messpunkts (z.B. 'MP1')
            
        Returns:
            Analyse-Ergebnis oder None bei Fehler
        """
        if mp_key not in self.measurement_points:
            logger.error(f"Unbekannter Messpunkt: {mp_key}")
            return None
        
        mp = self.measurement_points[mp_key]
        logger.info(f"Analysiere {mp.name}...")
        
        # Prüfe ob die erwartete Spalte vorhanden ist
        current_column = 'detected_current'  # Fallback auf erkannte Spalte
        if mp.csv_column in self.data.columns:
            current_column = mp.csv_column
        
        # Erkenne Betriebsmodi
        modes = self.detect_operating_modes(current_column)
        
        # Berechne Statistiken
        mode_statistics = self.calculate_mode_statistics(modes, current_column)
        duty_cycle = self.analyze_duty_cycle(modes)
        battery_life = self.estimate_battery_life(mode_statistics, duty_cycle)
        
        # Berechne Gesamtstatistiken
        total_duration = (self.data['timestamp'].iloc[-1] - self.data['timestamp'].iloc[0]).total_seconds()
        avg_power_mw = self.data[current_column].mean() * 3.3  # Annahme: 3.3V
        peak_power_mw = self.data[current_column].max() * 3.3
        
        result = EnergyAnalysisResult(
            measurement_point=mp.name,
            total_duration_s=total_duration,
            operating_modes=mode_statistics,
            duty_cycle_analysis=duty_cycle,
            average_power_consumption_mw=avg_power_mw,
            peak_power_consumption_mw=peak_power_mw,
            estimated_battery_life_hours=battery_life
        )
        
        return result
    
    def run_analysis(self) -> bool:
        """
        Führt die komplette Energieanalyse durch.
        
        Returns:
            True wenn erfolgreich, False bei Fehler
        """
        logger.info("Starte Energieanalyse...")
        
        if self.data is None:
            logger.error("Keine Daten geladen")
            return False
        
        # Analysiere alle verfügbaren Messpunkte
        for mp_key in self.measurement_points.keys():
            result = self.analyze_measurement_point(mp_key)
            if result is not None:
                self.analysis_results[mp_key] = result
        
        # Falls keine spezifischen Messpunkte gefunden wurden, analysiere Gesamtstrom
        if not self.analysis_results:
            logger.info("Keine spezifischen Messpunkte gefunden, analysiere Gesamtstrom...")
            result = self.analyze_measurement_point('MP1')  # Fallback auf MP1
            if result is not None:
                self.analysis_results['MP1'] = result
        
        return len(self.analysis_results) > 0
    
    def create_visualizations(self):
        """Erstellt Visualisierungen der Analyseergebnisse."""
        logger.info("Erstelle Visualisierungen...")
        
        # Set style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                plt.style.use('default')
        
        for mp_key, result in self.analysis_results.items():
            self._create_measurement_point_plots(mp_key, result)
        
        # Erstelle Zusammenfassungsplots
        self._create_summary_plots()
        
        logger.info(f"Visualisierungen gespeichert in: {self.output_dir}")
    
    def _create_measurement_point_plots(self, mp_key: str, result: EnergyAnalysisResult):
        """Erstellt Plots für einen spezifischen Messpunkt."""
        # Stromverlauf mit Modi
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Energieanalyse: {result.measurement_point}', fontsize=16)
        
        # Plot 1: Stromverlauf über Zeit mit Modi
        ax1 = axes[0, 0]
        modes = self.detect_operating_modes()
        current_data = self.data['detected_current']
        
        # Plotte Stromverlauf
        ax1.plot(self.data['timestamp'], current_data, 'k-', alpha=0.7, linewidth=0.5)
        
        # Füge farbige Bereiche für Modi hinzu
        for mode_key, mode in self.operating_modes.items():
            mode_mask = modes == mode_key
            if mode_mask.any():
                ax1.fill_between(self.data['timestamp'], 0, current_data.max() * 1.1, 
                               where=mode_mask, alpha=0.3, color=mode.color, label=mode.name)
        
        ax1.set_xlabel('Zeit')
        ax1.set_ylabel('Strom (mA)')
        ax1.set_title('Stromverlauf mit Betriebsmodi')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Duty-Cycle
        ax2 = axes[0, 1]
        duty_cycle_data = [(mode, percentage) for mode, percentage in result.duty_cycle_analysis.items() if percentage > 0]
        if duty_cycle_data:
            modes_names = [self.operating_modes.get(mode, type('obj', (object,), {'name': mode})).name for mode, _ in duty_cycle_data]
            percentages = [percentage for _, percentage in duty_cycle_data]
            colors = [self.operating_modes.get(mode, type('obj', (object,), {'color': 'gray'})).color for mode, _ in duty_cycle_data]
            
            ax2.pie(percentages, labels=modes_names, autopct='%1.1f%%', colors=colors)
            ax2.set_title('Duty-Cycle der Betriebsmodi')
        
        # Plot 3: Durchschnittlicher Stromverbrauch pro Modus
        ax3 = axes[1, 0]
        mode_names = []
        avg_currents = []
        colors_list = []
        
        for mode_key, stats in result.operating_modes.items():
            if stats['avg_current_ma'] > 0:
                mode_names.append(self.operating_modes[mode_key].name)
                avg_currents.append(stats['avg_current_ma'])
                colors_list.append(self.operating_modes[mode_key].color)
        
        if mode_names:
            bars = ax3.bar(mode_names, avg_currents, color=colors_list)
            ax3.set_ylabel('Durchschnittlicher Strom (mA)')
            ax3.set_title('Stromverbrauch pro Betriebsmodus')
            ax3.tick_params(axis='x', rotation=45)
            
            # Füge Werte auf den Balken hinzu
            for bar, current in zip(bars, avg_currents):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                        f'{current:.1f}mA', ha='center', va='bottom')
        
        # Plot 4: Geschätzte Batterielebensdauer
        ax4 = axes[1, 1]
        battery_names = list(result.estimated_battery_life_hours.keys())
        battery_hours = [result.estimated_battery_life_hours[name] for name in battery_names]
        battery_days = [h/24 if h != float('inf') else 999 for h in battery_hours]
        
        bars = ax4.bar(battery_names, battery_days, color='skyblue')
        ax4.set_ylabel('Geschätzte Lebensdauer (Tage)')
        ax4.set_title('Batterielebensdauer-Schätzung')
        ax4.tick_params(axis='x', rotation=45)
        
        # Füge Werte auf den Balken hinzu
        for bar, days in zip(bars, battery_days):
            if days < 999:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                        f'{days:.1f}d', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'energy_analysis_{mp_key}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_summary_plots(self):
        """Erstellt Zusammenfassungsplots aller Messpunkte."""
        if not self.analysis_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Energieanalyse - Zusammenfassung aller Messpunkte', fontsize=16)
        
        # Plot 1: Durchschnittliche Leistung pro Messpunkt
        ax1 = axes[0, 0]
        mp_names = [result.measurement_point for result in self.analysis_results.values()]
        avg_powers = [result.average_power_consumption_mw for result in self.analysis_results.values()]
        
        bars = ax1.bar(mp_names, avg_powers, color='lightblue')
        ax1.set_ylabel('Durchschnittliche Leistung (mW)')
        ax1.set_title('Leistungsverbrauch pro Messpunkt')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Peak-Leistung pro Messpunkt
        ax2 = axes[0, 1]
        peak_powers = [result.peak_power_consumption_mw for result in self.analysis_results.values()]
        
        bars = ax2.bar(mp_names, peak_powers, color='lightcoral')
        ax2.set_ylabel('Peak-Leistung (mW)')
        ax2.set_title('Spitzenleistung pro Messpunkt')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Batterielebensdauer Vergleich (für CR123A)
        ax3 = axes[1, 0]
        battery_lives = []
        for result in self.analysis_results.values():
            if 'CR123A' in result.estimated_battery_life_hours:
                battery_lives.append(result.estimated_battery_life_hours['CR123A'] / 24)  # Tage
            else:
                battery_lives.append(0)
        
        bars = ax3.bar(mp_names, battery_lives, color='lightgreen')
        ax3.set_ylabel('Geschätzte Lebensdauer (Tage)')
        ax3.set_title('Batterielebensdauer CR123A pro Messpunkt')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Energieverteilung nach Modi (erste verfügbare Messung)
        ax4 = axes[1, 1]
        first_result = list(self.analysis_results.values())[0]
        mode_energies = []
        mode_names = []
        mode_colors = []
        
        for mode_key, stats in first_result.operating_modes.items():
            if stats['energy_mah'] > 0:
                mode_names.append(self.operating_modes[mode_key].name)
                mode_energies.append(stats['energy_mah'])
                mode_colors.append(self.operating_modes[mode_key].color)
        
        if mode_energies:
            ax4.pie(mode_energies, labels=mode_names, autopct='%1.1f%%', colors=mode_colors)
            ax4.set_title('Energieverteilung nach Modi')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'energy_analysis_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_results(self):
        """Exportiert die Analyseergebnisse in verschiedene Formate."""
        logger.info("Exportiere Ergebnisse...")
        
        # JSON-Export
        json_results = {}
        for mp_key, result in self.analysis_results.items():
            json_results[mp_key] = asdict(result)
        
        json_path = self.output_dir / 'energy_analysis_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False, default=str)
        
        # CSV-Export der wichtigsten Kennzahlen
        summary_data = []
        for mp_key, result in self.analysis_results.items():
            row = {
                'Messpunkt': result.measurement_point,
                'Messdauer_s': result.total_duration_s,
                'Durchschnittliche_Leistung_mW': result.average_power_consumption_mw,
                'Peak_Leistung_mW': result.peak_power_consumption_mw,
                'Batterielebensdauer_CR123A_Tage': result.estimated_battery_life_hours.get('CR123A', 0) / 24
            }
            
            # Füge Modi-spezifische Daten hinzu
            for mode_key, stats in result.operating_modes.items():
                mode_name = self.operating_modes[mode_key].name.replace(' ', '_')
                row[f'{mode_name}_Dauer_s'] = stats['duration_s']
                row[f'{mode_name}_Strom_mA'] = stats['avg_current_ma']
                row[f'{mode_name}_Energie_mAh'] = stats['energy_mah']
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        csv_path = self.output_dir / 'energy_analysis_summary.csv'
        summary_df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Markdown-Bericht
        self._create_markdown_report()
        
        logger.info(f"Ergebnisse exportiert:")
        logger.info(f"  JSON: {json_path}")
        logger.info(f"  CSV:  {csv_path}")
        logger.info(f"  Report: {self.output_dir / 'energy_analysis_report.md'}")
    
    def _create_markdown_report(self):
        """Erstellt einen detaillierten Markdown-Bericht."""
        report_path = self.output_dir / 'energy_analysis_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Energiemessdaten-Analyse\n\n")
            f.write(f"**Erstellt am:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Eingabedatei:** {self.input_path.name}\n\n")
            
            f.write("## Zusammenfassung\n\n")
            if self.analysis_results:
                first_result = list(self.analysis_results.values())[0]
                f.write(f"- **Messdauer:** {first_result.total_duration_s:.1f} Sekunden\n")
                f.write(f"- **Analysierte Messpunkte:** {len(self.analysis_results)}\n")
                f.write(f"- **Erkannte Betriebsmodi:** {len([m for m in first_result.operating_modes.values() if m['sample_count'] > 0])}\n\n")
            
            # Detailanalyse für jeden Messpunkt
            for mp_key, result in self.analysis_results.items():
                f.write(f"## {result.measurement_point}\n\n")
                
                f.write("### Allgemeine Kennzahlen\n\n")
                f.write(f"- **Durchschnittliche Leistung:** {result.average_power_consumption_mw:.2f} mW\n")
                f.write(f"- **Spitzenleistung:** {result.peak_power_consumption_mw:.2f} mW\n")
                f.write(f"- **Messdauer:** {result.total_duration_s:.1f} Sekunden\n\n")
                
                f.write("### Betriebsmodi-Analyse\n\n")
                f.write("| Modus | Dauer (s) | Anteil (%) | Ø Strom (mA) | Peak Strom (mA) | Energie (mAh) |\n")
                f.write("|-------|-----------|------------|-------------|----------------|---------------|\n")
                
                for mode_key, stats in result.operating_modes.items():
                    if stats['sample_count'] > 0:
                        mode_name = self.operating_modes[mode_key].name
                        duty_percentage = result.duty_cycle_analysis.get(mode_key, 0)
                        f.write(f"| {mode_name} | {stats['duration_s']:.1f} | {duty_percentage:.1f} | "
                               f"{stats['avg_current_ma']:.2f} | {stats['peak_current_ma']:.2f} | {stats['energy_mah']:.4f} |\n")
                
                f.write("\n### Geschätzte Batterielebensdauer\n\n")
                f.write("| Batterietyp | Kapazität (mAh) | Lebensdauer (Stunden) | Lebensdauer (Tage) |\n")
                f.write("|-------------|-----------------|----------------------|--------------------|\n")
                
                for battery_name, hours in result.estimated_battery_life_hours.items():
                    capacity = self.battery_types[battery_name]['capacity_mah']
                    days = hours / 24 if hours != float('inf') else float('inf')
                    f.write(f"| {battery_name} | {capacity} | {hours:.1f} | {days:.1f} |\n")
                
                f.write("\n---\n\n")
            
            f.write("## Empfohlene Optimierungen\n\n")
            f.write("Basierend auf der Analyse werden folgende Optimierungen empfohlen:\n\n")
            
            # Automatische Empfehlungen basierend auf den Ergebnissen
            if self.analysis_results:
                result = list(self.analysis_results.values())[0]
                
                # Prüfe Sleep-Modus Effizienz
                sleep_stats = result.operating_modes.get('sleep', {})
                if sleep_stats.get('avg_current_ma', 0) > 1.0:
                    f.write("- **Sleep-Modus optimieren:** Der Sleep-Modus verbraucht mehr als 1 mA. "
                           "Prüfen Sie die Peripherie-Abschaltung.\n")
                
                # Prüfe Inferenz-Effizienz
                inference_stats = result.operating_modes.get('inference', {})
                if inference_stats.get('avg_current_ma', 0) > 180:
                    f.write("- **Inferenz-Optimierung:** Der Stromverbrauch während der Inferenz ist höher als erwartet. "
                           "Erwägen Sie CPU-Taktrate-Anpassungen oder Modell-Optimierungen.\n")
                
                # Prüfe Duty-Cycle
                sleep_duty = result.duty_cycle_analysis.get('sleep', 0)
                if sleep_duty < 80:
                    f.write(f"- **Duty-Cycle optimieren:** Der Sleep-Anteil beträgt nur {sleep_duty:.1f}%. "
                           "Für bessere Batterielebensdauer sollte dieser über 80% liegen.\n")
            
            f.write("\n## Visualisierungen\n\n")
            f.write("Die folgenden Diagramme wurden erstellt:\n\n")
            
            for mp_key in self.analysis_results.keys():
                f.write(f"- `energy_analysis_{mp_key}.png` - Detailanalyse {mp_key}\n")
            
            f.write("- `energy_analysis_summary.png` - Gesamtübersicht aller Messpunkte\n")


def create_sample_data(output_path: str):
    """
    Erstellt Beispiel-Messdaten für Tests.
    
    Args:
        output_path: Pfad für die Beispieldatei
    """
    logger.info(f"Erstelle Beispiel-Messdaten: {output_path}")
    
    # Simuliere einen typischen Duty-Cycle
    duration_seconds = 120  # 2 Minuten
    sample_rate_hz = 1000   # 1 kHz
    total_samples = duration_seconds * sample_rate_hz
    
    # Zeitstempel
    timestamps = pd.date_range(start=datetime.now(), periods=total_samples, freq='1ms')
    
    # Simuliere Stromverbrauch
    current_ma = np.zeros(total_samples)
    
    # Definiere einen typischen Zyklus (30 Sekunden)
    cycle_samples = 30 * sample_rate_hz
    cycles = total_samples // cycle_samples
    
    for cycle in range(cycles):
        start_idx = cycle * cycle_samples
        
        # Sleep (25 Sekunden)
        sleep_end = start_idx + 25 * sample_rate_hz
        current_ma[start_idx:sleep_end] = np.random.normal(0.5, 0.05, sleep_end - start_idx)
        
        # Wake-up (100 ms)
        wakeup_end = sleep_end + 100
        current_ma[sleep_end:wakeup_end] = np.random.normal(10, 1, wakeup_end - sleep_end)
        
        # Camera Init (200 ms)
        camera_init_end = wakeup_end + 200
        current_ma[wakeup_end:camera_init_end] = np.random.normal(25, 2, camera_init_end - wakeup_end)
        
        # Image Capture (100 ms)
        capture_end = camera_init_end + 100
        current_ma[camera_init_end:capture_end] = np.random.normal(50, 3, capture_end - camera_init_end)
        
        # Inference (50 ms)
        inference_end = capture_end + 50
        current_ma[capture_end:inference_end] = np.random.normal(165, 10, inference_end - capture_end)
        
        # Data Logging (100 ms)
        logging_end = inference_end + 100
        if logging_end <= start_idx + cycle_samples:
            current_ma[inference_end:logging_end] = np.random.normal(15, 1, logging_end - inference_end)
        
        # Rest of cycle in sleep
        if logging_end < start_idx + cycle_samples:
            current_ma[logging_end:start_idx + cycle_samples] = np.random.normal(0.5, 0.05, 
                                                                               start_idx + cycle_samples - logging_end)
    
    # Erstelle DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'total_current_ma': np.maximum(0, current_ma),  # Keine negativen Werte
        'battery_voltage_v': np.random.normal(3.0, 0.1, total_samples),  # Simulierte Batteriespannung
        'temperature_c': np.random.normal(25, 2, total_samples)  # Simulierte Temperatur
    })
    
    # Speichere als CSV
    data.to_csv(output_path, index=False)
    logger.info(f"Beispieldaten erstellt: {total_samples} Samples über {duration_seconds} Sekunden")


def main():
    """Hauptfunktion des Skripts."""
    parser = argparse.ArgumentParser(
        description='Analysiert Energiemessdaten von Power Analyzern oder Oszilloskopen',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  %(prog)s --input messung.csv --output ./analysis
  %(prog)s --input messung.csv --output ./analysis --create-sample
  %(prog)s --create-sample-only ./sample_data.csv
        """
    )
    
    parser.add_argument('--input', '-i', 
                       help='Pfad zur CSV-Messdatei')
    parser.add_argument('--output', '-o', 
                       default='./energy_analysis_output',
                       help='Ausgabe-Verzeichnis (Standard: ./energy_analysis_output)')
    parser.add_argument('--create-sample', 
                       action='store_true',
                       help='Erstelle zusätzlich Beispiel-Messdaten')
    parser.add_argument('--create-sample-only',
                       help='Erstelle nur Beispiel-Messdaten und beende')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Ausführliche Ausgabe')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Nur Beispieldaten erstellen
    if args.create_sample_only:
        create_sample_data(args.create_sample_only)
        return 0
    
    # Eingabe-Validierung
    if not args.input:
        parser.error("--input ist erforderlich (außer bei --create-sample-only)")
    
    if not Path(args.input).exists():
        logger.error(f"Eingabedatei nicht gefunden: {args.input}")
        return 1
    
    try:
        # Erstelle Analyzer
        analyzer = EnergyMeasurementAnalyzer(args.input, args.output)
        
        # Lade Daten
        if not analyzer.load_measurement_data():
            logger.error("Fehler beim Laden der Messdaten")
            return 1
        
        # Führe Analyse durch
        if not analyzer.run_analysis():
            logger.error("Fehler bei der Energieanalyse")
            return 1
        
        # Erstelle Visualisierungen
        analyzer.create_visualizations()
        
        # Exportiere Ergebnisse
        analyzer.export_results()
        
        # Erstelle Beispieldaten falls gewünscht
        if args.create_sample:
            sample_path = Path(args.output) / 'sample_measurement_data.csv'
            create_sample_data(str(sample_path))
        
        logger.info("Energieanalyse erfolgreich abgeschlossen!")
        logger.info(f"Ergebnisse verfügbar in: {analyzer.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Unerwarteter Fehler: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
