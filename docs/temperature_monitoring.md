# Temperaturmessung und Logging im RP2040-Emulator

Diese Implementierung ermöglicht die Simulation eines Temperatursensors im RP2040-Emulator sowie das Logging von Temperaturwerten und anderen Systemdaten.

## Funktionen

1. **Temperatursensor-Emulation**
   - Unterstützung für interne ADC-Temperatursensoren des RP2040
   - Unterstützung für externe I2C/SPI-Temperatursensoren
   - Realistische Simulation von Temperaturschwankungen und Sensorungenauigkeiten
   - Möglichkeit, Temperatur-Spikes zu injizieren für Testszenarien

2. **UART-Logging**
   - Emulation der UART-Schnittstelle für Diagnoseausgaben
   - Realistische Simulation von Übertragungszeiten basierend auf Baudrate
   - Ausgabe auf Konsole und/oder in Logdateien

3. **Strukturiertes Logging-System**
   - Unterschiedliche Log-Typen (Temperatur, Performance, System, Diagnose)
   - Mehrere Log-Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
   - CSV-Format für einfache Weiterverarbeitung und Analyse
   - Integration mit dem Emulator und PowerManager

## Verwendung

### Temperaturmessung

```python
# Temperatur auslesen
temperature = emulator.read_temperature()
print(f"Aktuelle Temperatur: {temperature:.2f}°C")

# Temperatur-Logging-Intervall konfigurieren
emulator.set_temperature_log_interval(60.0)  # Alle 60 Sekunden loggen

# Manuelles Temperatur-Logging auslösen
emulator.log_temperature()

# Künstlichen Temperaturanstieg für Tests injizieren
emulator.inject_temperature_spike(delta=5.0, duration=60.0)
```

### Systemstatistiken mit Temperaturinformationen

```python
stats = emulator.get_system_stats()
print(f"Aktuelle Temperatur: {stats['current_temperature_c']:.2f}°C")
print(f"Min Temperatur: {stats['temperature_min_c']:.2f}°C")
print(f"Max Temperatur: {stats['temperature_max_c']:.2f}°C")
print(f"Durchschnitt: {stats['temperature_avg_c']:.2f}°C")
```

### Zusammenhang zwischen Temperatur, Leistung und Energieverbrauch

Die Implementierung berücksichtigt die folgenden Wechselwirkungen:

1. **Temperaturentwicklung bei CPU-Last**
   - Bei höherer CPU-Last (z.B. Inferenz) steigt die Temperatur
   - Die Temperaturerhöhung ist proportional zur Ausführungszeit und Speichernutzung
   - Beispiel: `execute_operation(memory_usage_bytes, operation_time_ms)` erhöht die Temperatur

2. **Temperaturabhängiger Energieverbrauch**
   - Der PowerManager berücksichtigt die Temperatur bei der Berechnung des Energieverbrauchs
   - Bei höheren Temperaturen steigt der Energieverbrauch leicht an
   - Die Batterielebensdauer wird basierend auf Temperatur und Aktivitätsprofil geschätzt

3. **Temperaturverhalten im Sleep-Modus**
   - Im Sleep-Modus stabilisiert sich die Temperatur langsam in Richtung Raumtemperatur
   - Die Abkühlrate hängt von der Temperaturdifferenz zur Umgebung ab
   - Der Energieverbrauch sinkt im Sleep-Modus erheblich

4. **Anpassungen basierend auf Temperatur**
   - Bei hohen Temperaturen kann der Emulator automatisch die Taktrate reduzieren
   - Möglich ist auch ein temperaturbedingter Sleep-Modus zum Schutz der Komponenten
   - Die Sensorgenauigkeit variiert mit der Temperatur

## Testskript

Das Testskript `tests/test_temperature_logging.py` demonstriert die Verwendung aller implementierten Funktionen:

1. Normale Temperaturmessung
2. Temperaturentwicklung unter Last (Inferenz)
3. Simulation eines Temperatur-Spikes
4. Temperaturverhalten im Sleep-Modus
5. Anzeige von Systemstatistiken mit Temperaturinformationen

Ausführung:
```bash
python -m tests.test_temperature_logging
```

Die Logs werden im Verzeichnis `output/emulator_logs/` gespeichert.

## Log-Dateien

- `temperature_log_<timestamp>.csv`: Zeitreihen von Temperaturmessungen im CSV-Format
- `performance_log_<timestamp>.csv`: Performance-Metriken mit Temperaturinformationen
- `system_log_<timestamp>.log`: Allgemeine Systemlogs
- `uart_log_<timestamp>.txt`: Emulierte UART-Ausgaben

## Visualisierung der Temperaturdaten

Die generierten CSV-Dateien können mit verschiedenen Tools visualisiert werden:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Temperaturlog laden
df = pd.read_csv('output/emulator_logs/temperature_log_20250518_162526.csv')

# Zeitstempel in datetime konvertieren
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Temperaturverlauf plotten
plt.figure(figsize=(10, 6))
plt.plot(df['timestamp'], df['temperature_c'], marker='o')
plt.title('RP2040 Temperaturverlauf')
plt.xlabel('Zeit')
plt.ylabel('Temperatur (°C)')
plt.grid(True)
plt.tight_layout()
plt.savefig('temperature_plot.png')
plt.show()
```
