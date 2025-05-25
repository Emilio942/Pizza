#!/usr/bin/env python3
"""
Visualisierungsskript für Temperaturdaten aus dem RP2040-Emulator.
Dieses Skript lädt CSV-Logdateien und erstellt Visualisierungen der Temperaturverläufe.
"""

import os
import argparse
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def find_latest_log(log_dir, log_type="temperature"):
    """
    Findet die neueste Logdatei eines bestimmten Typs.
    
    Args:
        log_dir: Verzeichnis mit den Logdateien
        log_type: Typ der Logdatei ("temperature" oder "performance")
    
    Returns:
        Pfad zur neuesten Logdatei oder None wenn keine gefunden wurde
    """
    pattern = f"{log_type}_log_*.csv"
    logs = list(Path(log_dir).glob(pattern))
    
    if not logs:
        return None
    
    # Sortiere nach Änderungsdatum, neueste zuerst
    return str(sorted(logs, key=lambda x: x.stat().st_mtime, reverse=True)[0])

def load_temperature_log(log_path):
    """
    Lädt ein Temperatur-Logfile und bereitet es für die Visualisierung vor.
    
    Args:
        log_path: Pfad zur CSV-Datei
    
    Returns:
        DataFrame mit den Daten oder None bei Fehler
    """
    try:
        df = pd.read_csv(log_path)
        
        # Prüfe ob die erwarteten Spalten vorhanden sind
        if 'timestamp' not in df.columns or 'temperature_c' not in df.columns:
            print(f"Fehler: CSV-Datei hat nicht das erwartete Format.")
            return None
        
        # Konvertiere Zeitstempel
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    except Exception as e:
        print(f"Fehler beim Laden der Logdatei: {e}")
        return None

def plot_temperature(df, output_path=None, show_plot=True):
    """
    Erstellt einen Plot des Temperaturverlaufs.
    
    Args:
        df: DataFrame mit den Temperaturdaten
        output_path: Pfad zum Speichern des Plots (optional)
        show_plot: Wenn True, wird der Plot angezeigt
    """
    plt.figure(figsize=(12, 6))
    
    # Erzeuge Hauptplot
    ax = plt.subplot(111)
    ax.plot(df['timestamp'], df['temperature_c'], marker='o', linestyle='-', linewidth=2, label='Temperatur (°C)')
    
    # Formatiere X-Achse für bessere Lesbarkeit
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gcf().autofmt_xdate()
    
    # Setze Titel und Labels
    plt.title('RP2040 Temperaturverlauf', fontsize=16)
    plt.xlabel('Zeit', fontsize=14)
    plt.ylabel('Temperatur (°C)', fontsize=14)
    
    # Zeige Statistiken im Plot
    min_temp = df['temperature_c'].min()
    max_temp = df['temperature_c'].max()
    avg_temp = df['temperature_c'].mean()
    plt.text(0.02, 0.95, f"Min: {min_temp:.1f}°C\nMax: {max_temp:.1f}°C\nDurchschnitt: {avg_temp:.1f}°C", 
             transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    # Setze Gitterlinien und Legende
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Speichere den Plot wenn gewünscht
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Plot gespeichert unter: {output_path}")
    
    # Zeige den Plot an
    if show_plot:
        plt.show()
    
    plt.close()

def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description="Visualisierung von Temperaturdaten aus dem RP2040-Emulator")
    parser.add_argument("-l", "--log", help="Pfad zur Temperature-Log CSV-Datei")
    parser.add_argument("-d", "--dir", default="output/emulator_logs", help="Verzeichnis mit Log-Dateien")
    parser.add_argument("-o", "--output", help="Pfad zum Speichern des Plots")
    parser.add_argument("--no-show", action="store_true", help="Plot nicht anzeigen")
    
    args = parser.parse_args()
    
    # Bestimme die zu verwendende Logdatei
    log_path = args.log
    if not log_path:
        log_path = find_latest_log(args.dir, "temperature")
        if not log_path:
            print(f"Keine Temperatur-Logdatei gefunden in {args.dir}")
            return
    
    print(f"Verwende Logdatei: {log_path}")
    
    # Lade Daten
    df = load_temperature_log(log_path)
    if df is None:
        return
    
    # Erzeuge Plot
    plot_temperature(df, args.output, not args.no_show)

if __name__ == "__main__":
    main()
