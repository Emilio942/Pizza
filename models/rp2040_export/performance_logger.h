/**
 * @file performance_logger.h
 * @brief Bibliothek für Performance-Logging auf dem RP2040
 * 
 * Diese Bibliothek implementiert Funktionen zur Messung und zum Logging von 
 * Performance-Metriken auf dem RP2040-Mikrocontroller, insbesondere für
 * die Pizza-Erkennungsanwendung.
 * 
 * Metriken: 
 * - Inferenzzeit pro Zyklus (in Millisekunden)
 * - RAM-Nutzung (Peak und Durchschnitt in KB)
 * - CPU-Auslastung (in Prozent)
 * - Temperatur (in Grad Celsius)
 * 
 * @author Pizza Detection Team
 * @date 2025-05-05
 */

#ifndef PERFORMANCE_LOGGER_H
#define PERFORMANCE_LOGGER_H

#include <stdint.h>
#include <stdbool.h>
#include "pico/stdlib.h"
#include "hardware/timer.h"
#include "hardware/watchdog.h"
#include "hardware/adc.h"
#include "hardware/gpio.h"
#include "pico/multicore.h"
#include "hardware/flash.h"

// Maximale Anzahl an Einträgen, die im Ringpuffer gespeichert werden können
#define MAX_LOG_ENTRIES 100

// Struktur für einen einzelnen Log-Eintrag
typedef struct {
    uint32_t timestamp;        // Zeitstempel in Millisekunden seit Start
    uint32_t inference_time;   // Dauer der Inferenz in Mikrosekunden
    uint32_t peak_ram_usage;   // Maximaler RAM-Verbrauch in Bytes
    uint16_t cpu_load;         // CPU-Auslastung in Prozent * 100 (z.B. 5432 = 54.32%)
    uint16_t temperature;      // Temperatur in Grad Celsius * 100 (z.B. 2345 = 23.45°C)
    uint8_t  prediction;       // Vorhergesagte Klasse (0-5)
    uint8_t  confidence;       // Vertrauen in Vorhersage (0-100%)
} performance_log_entry_t;

// Performance-Logger-Konfiguration
typedef struct {
    bool log_to_uart;          // Log-Ausgabe über UART
    bool log_to_sd;            // Log-Ausgabe auf SD-Karte
    bool log_to_ram;           // Log-Ausgabe im RAM-Puffer
    uint16_t log_interval;     // Intervall zwischen Logs in ms (0 = jede Inferenz)
    uint8_t uart_port;         // UART-Port (0 oder 1)
    const char* sd_filename;   // Dateiname auf SD-Karte
} performance_logger_config_t;

// Logger-Status
typedef enum {
    LOGGER_IDLE,               // Logger ist im Leerlauf
    LOGGER_RUNNING,            // Logger ist aktiv
    LOGGER_ERROR               // Fehler im Logger
} performance_logger_status_t;

// Globaler Logger-Status
extern performance_logger_status_t logger_status;

/**
 * @brief Initialisiert den Performance-Logger
 * 
 * @param config Konfiguration des Loggers
 * @return true wenn erfolgreich, false bei Fehler
 */
bool performance_logger_init(performance_logger_config_t config);

/**
 * @brief Startet eine neue Performance-Messung
 * 
 * Diese Funktion sollte zu Beginn einer Inferenz aufgerufen werden.
 * Sie startet die Timer und initialisiert die Speichermessung.
 * 
 * @return true wenn erfolgreich, false bei Fehler
 */
bool performance_logger_start_measurement();

/**
 * @brief Beendet eine Performance-Messung und zeichnet die Daten auf
 * 
 * Diese Funktion sollte nach einer Inferenz aufgerufen werden.
 * Sie stoppt die Timer, erfasst den Peak-RAM und protokolliert die Daten.
 * 
 * @param prediction Die vorhergesagte Klasse (0-5)
 * @param confidence Das Vertrauen in die Vorhersage (0-100%)
 * @return true wenn erfolgreich, false bei Fehler
 */
bool performance_logger_end_measurement(uint8_t prediction, uint8_t confidence);

/**
 * @brief Schreibt alle gepufferten Logs auf die ausgewählten Ausgabekanäle
 * 
 * Diese Funktion sollte regelmäßig aufgerufen werden, um gepufferte Logs
 * auf UART oder SD-Karte zu schreiben.
 * 
 * @return true wenn erfolgreich, false bei Fehler
 */
bool performance_logger_flush();

/**
 * @brief Gibt aktuelle Performance-Statistiken zurück
 * 
 * Berechnet Durchschnitte und Extremwerte aus den gesammelten Logs.
 * 
 * @param avg_inference_time Durchschnittliche Inferenzzeit in µs
 * @param max_inference_time Maximale Inferenzzeit in µs
 * @param avg_ram_usage Durchschnittlicher RAM-Verbrauch in Bytes
 * @param peak_ram_usage Maximaler RAM-Verbrauch in Bytes
 * @param avg_temperature Durchschnittliche Temperatur in °C * 100
 * @return true wenn erfolgreich, false bei Fehler
 */
bool performance_logger_get_stats(uint32_t* avg_inference_time, 
                                 uint32_t* max_inference_time,
                                 uint32_t* avg_ram_usage,
                                 uint32_t* peak_ram_usage,
                                 uint16_t* avg_temperature);

/**
 * @brief Gibt den gesamten Logpuffer zurück
 * 
 * Diese Funktion ist nützlich, um alle Logs auf einmal abzurufen.
 * 
 * @param buffer Puffer, in den die Logs kopiert werden
 * @param max_entries Maximale Anzahl an Einträgen, die kopiert werden sollen
 * @return Anzahl der tatsächlich kopierten Einträge
 */
uint16_t performance_logger_get_log_entries(performance_log_entry_t* buffer, 
                                           uint16_t max_entries);

/**
 * @brief Löscht alle Logs
 * 
 * @return true wenn erfolgreich, false bei Fehler
 */
bool performance_logger_clear();

/**
 * @brief Beendet den Logger
 * 
 * Schreibt alle verbleibenden Logs und gibt Ressourcen frei.
 * 
 * @return true wenn erfolgreich, false bei Fehler
 */
bool performance_logger_deinit();

#endif /* PERFORMANCE_LOGGER_H */