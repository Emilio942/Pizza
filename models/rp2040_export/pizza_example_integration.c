/**
 * @file pizza_example_integration.c
 * @brief Beispielintegration der CLAHE-Vorverarbeitung mit dem Pizza-Erkennungssystem
 * 
 * Diese Datei demonstriert, wie die CLAHE-Bildvorverarbeitung vor der
 * Inferenz des neuronalen Netzes integriert werden kann.
 */

#include <stdlib.h>
#include <stdint.h>
#include "pizza_inference.h"
#include "pizza_temporal.h"
#include "clahe/pizza_clahe.h"

// Bildgröße aus den Projektkonstanten 
#define PIZZA_IMG_WIDTH  48
#define PIZZA_IMG_HEIGHT 48
#define PIZZA_IMG_CHANNELS 3

// CLAHE-Konfiguration
static PizzaClaheConfig g_clahe_config;
static int g_clahe_initialized = 0;

// Temporärer Speicher für Zwischenbilder
static uint8_t* g_img_buffer = NULL;

/**
 * @brief Initialisiert die Bildvorverarbeitung
 * 
 * @return int 0 bei Erfolg, -1 bei Fehler
 */
int pizza_preprocessing_init(void) {
    // Initialisiere CLAHE-Konfiguration
    pizza_clahe_init(&g_clahe_config, PIZZA_IMG_WIDTH, PIZZA_IMG_HEIGHT);
    
    // Passe Parameter an (diese Werte wurden in Tests als optimal identifiziert)
    g_clahe_config.clip_limit = 3;  // Etwas konservativeres Clip-Limit für natürlicheres Aussehen
    g_clahe_config.grid_size = 6;   // Optimal für 48x48 Bilder
    
    // Prüfe Ressourcenanforderungen
    uint32_t ram_required, flash_required, cycles;
    if (pizza_clahe_get_resource_usage(&g_clahe_config, &ram_required, &flash_required, &cycles) != 0) {
        return -1;
    }
    
    // Prüfe, ob der RAM-Verbrauch in Ordnung ist
    // Hinweis: Dies sollte in den Speicherschätzungen berücksichtigt werden
    if (ram_required > 10000) {  // Maximal 10KB zusätzlicher RAM-Verbrauch erlaubt
        return -1;
    }
    
    // Alloziere Puffer für Bildverarbeitung (falls nötig)
    g_img_buffer = (uint8_t*)malloc(PIZZA_IMG_WIDTH * PIZZA_IMG_HEIGHT * PIZZA_IMG_CHANNELS);
    if (!g_img_buffer) {
        return -1;
    }
    
    g_clahe_initialized = 1;
    return 0;
}

/**
 * @brief Führt die Bildvorverarbeitung durch
 * 
 * @param image Eingabebild (RGB-Format, 3 Byte pro Pixel)
 * @return int 0 bei Erfolg, -1 bei Fehler
 */
int pizza_preprocess_image(uint8_t* image) {
    if (!g_clahe_initialized || !image) {
        return -1;
    }
    
    // Anwenden der CLAHE-Bildvorverarbeitung
    // Hinweis: Dies wird in-place auf dem Eingabebild durchgeführt
    return pizza_clahe_apply_rgb(image, image, &g_clahe_config);
}

/**
 * @brief Führt die vollständige Pizza-Erkennung durch (Vorverarbeitung + Inferenz)
 * 
 * @param image Eingabebild (RGB-Format, 3 Byte pro Pixel)
 * @param result Ergebnis der Erkennung (Pizza-Klasse und Konfidenz)
 * @return int 0 bei Erfolg, -1 bei Fehler
 */
int pizza_detect_with_preprocessing(uint8_t* image, PizzaResult* result) {
    if (!image || !result) {
        return -1;
    }
    
    // 1. Kopiere das Bild in den Zwischenpuffer
    memcpy(g_img_buffer, image, PIZZA_IMG_WIDTH * PIZZA_IMG_HEIGHT * PIZZA_IMG_CHANNELS);
    
    // 2. Wende Bildvorverarbeitung an
    if (pizza_preprocess_image(g_img_buffer) != 0) {
        return -1;
    }
    
    // 3. Führe neuronale Netzinferenz durch
    float confidences[PIZZA_NUM_CLASSES];
    if (pizza_inference(g_img_buffer, confidences) != 0) {
        return -1;
    }
    
    // 4. Füge das Ergebnis zum Temporal Smoothing hinzu (falls aktiviert)
    pizza_temporal_add_result(confidences);
    
    // 5. Hole das endgültige Ergebnis (mit Temporal Smoothing, falls aktiviert)
    pizza_temporal_get_smoothed_result(confidences);
    
    // 6. Bestimme die Klasse mit der höchsten Konfidenz
    uint8_t max_class = 0;
    float max_confidence = confidences[0];
    
    for (uint8_t i = 1; i < PIZZA_NUM_CLASSES; i++) {
        if (confidences[i] > max_confidence) {
            max_confidence = confidences[i];
            max_class = i;
        }
    }
    
    // 7. Setze Ergebnis
    result->pizza_class = max_class;
    result->confidence = max_confidence;
    
    return 0;
}

/**
 * @brief Cleanup der Bildvorverarbeitung
 */
void pizza_preprocessing_cleanup(void) {
    if (g_img_buffer) {
        free(g_img_buffer);
        g_img_buffer = NULL;
    }
    
    g_clahe_initialized = 0;
}