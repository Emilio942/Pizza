/**
 * @file pizza_clahe.h
 * @brief CLAHE (Contrast Limited Adaptive Histogram Equalization) für RP2040
 * 
 * Ressourcensparende Implementierung für automatische Beleuchtungskorrektur
 * auf dem RP2040-Mikrocontroller. Diese Implementierung ist speziell für
 * das Pizza-Erkennungssystem optimiert und wird vor jedem Netzaufruf ausgeführt.
 */

#ifndef PIZZA_CLAHE_H
#define PIZZA_CLAHE_H

#include <stdint.h>

/**
 * @brief Konfigurationsstruktur für CLAHE
 */
typedef struct {
    uint8_t clip_limit;       // Limitierung der Histogramm-Kontrastverstärkung (0-255)
    uint8_t grid_size;        // Größe der Gitter-Unterteilung (empfohlen: 8)
    uint16_t image_width;     // Bildbreite
    uint16_t image_height;    // Bildhöhe
} PizzaClaheConfig;

/**
 * @brief Initialisiert die CLAHE-Konfiguration mit Standardwerten
 * 
 * @param config Zeiger auf die Konfigurationsstruktur
 * @param width Bildbreite
 * @param height Bildhöhe
 */
void pizza_clahe_init(PizzaClaheConfig* config, uint16_t width, uint16_t height);

/**
 * @brief Wendet CLAHE auf ein Graustufenbild an
 * 
 * @param input Eingabebild (Graustufen, 8-bit pro Pixel)
 * @param output Ausgabebild (kann identisch mit input sein für In-Place-Verarbeitung)
 * @param config CLAHE-Konfiguration
 * @return int Fehlercode (0 bei Erfolg)
 */
int pizza_clahe_apply_grayscale(uint8_t* input, uint8_t* output, const PizzaClaheConfig* config);

/**
 * @brief Wendet CLAHE auf ein RGB-Bild an, indem es nur den Helligkeitskanal bearbeitet
 * 
 * @param input Eingabebild (RGB, 3 Bytes pro Pixel in der Reihenfolge R,G,B)
 * @param output Ausgabebild (kann identisch mit input sein für In-Place-Verarbeitung)
 * @param config CLAHE-Konfiguration
 * @return int Fehlercode (0 bei Erfolg)
 */
int pizza_clahe_apply_rgb(uint8_t* input, uint8_t* output, const PizzaClaheConfig* config);

/**
 * @brief Gibt Speichernutzung und Verarbeitungszeit für CLAHE zurück
 * 
 * @param config CLAHE-Konfiguration
 * @param peak_ram_bytes Ausgabe: Maximaler RAM-Verbrauch in Bytes
 * @param flash_usage_bytes Ausgabe: Flash-Nutzung in Bytes
 * @param estimated_cycles Ausgabe: Geschätzte CPU-Zyklen auf RP2040
 * @return int Fehlercode (0 bei Erfolg)
 */
int pizza_clahe_get_resource_usage(
    const PizzaClaheConfig* config,
    uint32_t* peak_ram_bytes,
    uint32_t* flash_usage_bytes,
    uint32_t* estimated_cycles
);

/**
 * @brief Berechnet den erforderlichen Arbeitsspeicher für die CLAHE-Verarbeitung
 * 
 * @param config CLAHE-Konfiguration
 * @return uint32_t Benötigter Arbeitsspeicher in Bytes
 */
uint32_t pizza_clahe_get_required_memory(const PizzaClaheConfig* config);

#endif /* PIZZA_CLAHE_H */