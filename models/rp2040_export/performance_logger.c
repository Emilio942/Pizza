/**
 * @file performance_logger.c
 * @brief Implementierung der Performance-Logging-Bibliothek für RP2040
 * 
 * @author Pizza Detection Team
 * @date 2025-05-05
 */

#include "performance_logger.h"
#include <string.h>
#include <stdio.h>

// Wenn SD-Karten-Unterstützung aktiviert ist, FatFS einbinden
#ifdef ENABLE_SD_CARD
#include "ff.h"
#include "pico/sd_card.h"
#endif

// Ringpuffer für Logs
static performance_log_entry_t log_buffer[MAX_LOG_ENTRIES];
static uint16_t log_buffer_head = 0;
static uint16_t log_buffer_tail = 0;
static uint16_t log_buffer_count = 0;

// Aktuelle Logger-Konfiguration
static performance_logger_config_t logger_config;

// Globaler Logger-Status
performance_logger_status_t logger_status = LOGGER_IDLE;

// Messvariablen
static absolute_time_t measurement_start_time;
static uint32_t heap_start;
static uint32_t peak_heap_usage;

// Lokale Hilfsfunktionen
static uint32_t get_current_heap_usage(void);
static uint16_t get_cpu_load(void);
static uint16_t get_temperature(void);
static bool write_log_to_uart(const performance_log_entry_t *entry);
static bool write_log_to_sd(const performance_log_entry_t *entry);
static bool add_log_to_buffer(const performance_log_entry_t *entry);

//------------------------------------------------------
// Öffentliche Funktionen
//------------------------------------------------------

bool performance_logger_init(performance_logger_config_t config) {
    // Konfiguration speichern
    memcpy(&logger_config, &config, sizeof(performance_logger_config_t));
    
    // Logpuffer zurücksetzen
    memset(log_buffer, 0, sizeof(log_buffer));
    log_buffer_head = 0;
    log_buffer_tail = 0;
    log_buffer_count = 0;
    
    // Hardware initialisieren
    
    // Timer initialisieren für präzise Zeitmessungen
    if (!timer_is_running(0)) {
        timer_hw->armed = 1;
    }
    
    // ADC für Temperaturmessung initialisieren
    adc_init();
    adc_set_temp_sensor_enabled(true);
    
    // UART-Setup bei Bedarf
    if (config.log_to_uart) {
        if (config.uart_port == 0) {
            uart_init(uart0, 115200);
            gpio_set_function(0, GPIO_FUNC_UART); // TX
            gpio_set_function(1, GPIO_FUNC_UART); // RX
        } else {
            uart_init(uart1, 115200);
            gpio_set_function(4, GPIO_FUNC_UART); // TX
            gpio_set_function(5, GPIO_FUNC_UART); // RX
        }
    }
    
    // SD-Karten-Setup bei Bedarf
#ifdef ENABLE_SD_CARD
    if (config.log_to_sd) {
        // SD-Karte initialisieren
        if (sd_init_driver() != SD_DRIVER_OK) {
            return false;
        }
        
        // FatFS initialisieren
        static FATFS fs;
        FRESULT fr = f_mount(&fs, "", 1);
        if (fr != FR_OK) {
            return false;
        }
        
        // Überprüfen, ob die Datei erstellt werden kann
        FIL file;
        fr = f_open(&file, config.sd_filename, FA_WRITE | FA_CREATE_ALWAYS);
        if (fr == FR_OK) {
            // CSV-Header schreiben
            f_printf(&file, "Timestamp,InferenceTime,PeakRamUsage,CpuLoad,Temperature,Prediction,Confidence\n");
            f_close(&file);
        } else {
            return false;
        }
    }
#endif
    
    // Alles erfolgreich initialisiert
    logger_status = LOGGER_RUNNING;
    return true;
}

bool performance_logger_start_measurement() {
    if (logger_status != LOGGER_RUNNING) {
        return false;
    }
    
    // Aktuelle Zeit als Startpunkt speichern
    measurement_start_time = get_absolute_time();
    
    // Aktuellen Heap-Verbrauch als Baseline speichern
    heap_start = get_current_heap_usage();
    peak_heap_usage = heap_start;
    
    return true;
}

bool performance_logger_end_measurement(uint8_t prediction, uint8_t confidence) {
    if (logger_status != LOGGER_RUNNING) {
        return false;
    }
    
    // Zeit seit Beginn der Messung berechnen
    uint32_t inference_time = absolute_time_diff_us(measurement_start_time, get_absolute_time());
    
    // Neuen Log-Eintrag erstellen
    performance_log_entry_t entry = {
        .timestamp = to_ms_since_boot(get_absolute_time()),
        .inference_time = inference_time,
        .peak_ram_usage = peak_heap_usage,
        .cpu_load = get_cpu_load(),
        .temperature = get_temperature(),
        .prediction = prediction,
        .confidence = confidence
    };
    
    // Log ausgeben, je nach Konfiguration
    if (logger_config.log_to_uart) {
        write_log_to_uart(&entry);
    }
    
    if (logger_config.log_to_sd) {
        write_log_to_sd(&entry);
    }
    
    if (logger_config.log_to_ram) {
        add_log_to_buffer(&entry);
    }
    
    return true;
}

bool performance_logger_flush() {
    if (logger_status != LOGGER_RUNNING || log_buffer_count == 0) {
        return false;
    }
    
    // Alle Einträge im Puffer ausgeben
    uint16_t processed = 0;
    
    while (processed < log_buffer_count) {
        performance_log_entry_t *entry = &log_buffer[log_buffer_tail];
        
        if (logger_config.log_to_uart) {
            write_log_to_uart(entry);
        }
        
        if (logger_config.log_to_sd) {
            write_log_to_sd(entry);
        }
        
        // Zum nächsten Eintrag wechseln
        log_buffer_tail = (log_buffer_tail + 1) % MAX_LOG_ENTRIES;
        processed++;
    }
    
    // Puffer zurücksetzen
    log_buffer_count = 0;
    log_buffer_head = 0;
    log_buffer_tail = 0;
    
    return true;
}

bool performance_logger_get_stats(uint32_t* avg_inference_time, 
                                 uint32_t* max_inference_time,
                                 uint32_t* avg_ram_usage,
                                 uint32_t* peak_ram_usage,
                                 uint16_t* avg_temperature) {
    if (logger_status != LOGGER_RUNNING || log_buffer_count == 0) {
        return false;
    }
    
    uint64_t total_inference_time = 0;
    uint64_t total_ram_usage = 0;
    uint32_t total_temperature = 0;
    uint32_t max_inf_time = 0;
    uint32_t max_ram = 0;
    
    uint16_t entry_index = log_buffer_tail;
    uint16_t processed = 0;
    
    // Durchlaufe alle Einträge im Puffer
    while (processed < log_buffer_count) {
        performance_log_entry_t *entry = &log_buffer[entry_index];
        
        total_inference_time += entry->inference_time;
        total_ram_usage += entry->peak_ram_usage;
        total_temperature += entry->temperature;
        
        if (entry->inference_time > max_inf_time) {
            max_inf_time = entry->inference_time;
        }
        
        if (entry->peak_ram_usage > max_ram) {
            max_ram = entry->peak_ram_usage;
        }
        
        entry_index = (entry_index + 1) % MAX_LOG_ENTRIES;
        processed++;
    }
    
    // Berechnete Werte zurückgeben
    *avg_inference_time = (uint32_t)(total_inference_time / log_buffer_count);
    *max_inference_time = max_inf_time;
    *avg_ram_usage = (uint32_t)(total_ram_usage / log_buffer_count);
    *peak_ram_usage = max_ram;
    *avg_temperature = (uint16_t)(total_temperature / log_buffer_count);
    
    return true;
}

uint16_t performance_logger_get_log_entries(performance_log_entry_t* buffer, 
                                           uint16_t max_entries) {
    if (logger_status != LOGGER_RUNNING || log_buffer_count == 0 || buffer == NULL) {
        return 0;
    }
    
    uint16_t entries_to_copy = (log_buffer_count < max_entries) ? log_buffer_count : max_entries;
    uint16_t entry_index = log_buffer_tail;
    
    for (uint16_t i = 0; i < entries_to_copy; i++) {
        memcpy(&buffer[i], &log_buffer[entry_index], sizeof(performance_log_entry_t));
        entry_index = (entry_index + 1) % MAX_LOG_ENTRIES;
    }
    
    return entries_to_copy;
}

bool performance_logger_clear() {
    if (logger_status != LOGGER_RUNNING) {
        return false;
    }
    
    // Puffer zurücksetzen
    log_buffer_count = 0;
    log_buffer_head = 0;
    log_buffer_tail = 0;
    
    return true;
}

bool performance_logger_deinit() {
    // Zuerst alle verbleibenden Logs ausgeben
    performance_logger_flush();
    
    // Hardware-Ressourcen freigeben
    
    // ADC deaktivieren
    adc_set_temp_sensor_enabled(false);
    
    // UART deaktivieren bei Bedarf
    if (logger_config.log_to_uart) {
        if (logger_config.uart_port == 0) {
            uart_deinit(uart0);
        } else {
            uart_deinit(uart1);
        }
    }
    
    // SD-Karte deaktivieren bei Bedarf
#ifdef ENABLE_SD_CARD
    if (logger_config.log_to_sd) {
        // SD-Karte unmounten
        f_unmount("");
    }
#endif
    
    // Status zurücksetzen
    logger_status = LOGGER_IDLE;
    
    return true;
}

//------------------------------------------------------
// Lokale Hilfsfunktionen
//------------------------------------------------------

static uint32_t get_current_heap_usage(void) {
    // Hier wäre die tatsächliche Heap-Nutzungsmessung
    // Diese Funktion ist stark von der spezifischen Implementierung abhängig
    // und könnte z.B. malloc_stats oder eine ähnliche Funktion verwenden
    
    // Einfache Implementierung für Demonstrationszwecke
    // Für tatsächliche Implementierung, siehe pico-sdk Dokumentation
    extern char __StackLimit, __bss_end__;
    char stackptr;
    uint32_t heap_used = &stackptr - &__bss_end__;
    
    // Aktualisiere den Peak-Wert, wenn nötig
    if (heap_used > peak_heap_usage) {
        peak_heap_usage = heap_used;
    }
    
    return heap_used;
}

static uint16_t get_cpu_load(void) {
    // Diese Funktion würde die CPU-Auslastung messen
    // Für den RP2040 könnte dies über Timer-Interrupts oder andere Metriken erfolgen
    
    // Einfache Implementierung für Demonstrationszwecke
    // In der Realität würde hier ein moving average oder ähnliches stehen
    return 5000; // 50.00%
}

static uint16_t get_temperature(void) {
    // Lese die Temperatur über den internen Temperatursensor des RP2040
    adc_select_input(4); // Temperatursensor auf ADC4
    uint16_t raw = adc_read();
    
    // Umrechnung in Grad Celsius * 100 laut RP2040-Datenblatt
    // T = 27 - (ADC_Voltage - 0.706)/0.001721
    float voltage = raw * 3.3f / 4096.0f;
    float temperature = 27.0f - (voltage - 0.706f) / 0.001721f;
    
    return (uint16_t)(temperature * 100.0f);
}

static bool write_log_to_uart(const performance_log_entry_t *entry) {
    if (entry == NULL) {
        return false;
    }
    
    // UART-Port auswählen
    uart_inst_t *uart_port = (logger_config.uart_port == 0) ? uart0 : uart1;
    
    // Log-String formatieren (CSV-Format)
    char log_str[128];
    sprintf(log_str, "%lu,%lu,%lu,%u,%u,%u,%u\n",
            entry->timestamp,
            entry->inference_time,
            entry->peak_ram_usage,
            entry->cpu_load,
            entry->temperature,
            entry->prediction,
            entry->confidence);
    
    // String senden
    uart_puts(uart_port, log_str);
    
    return true;
}

static bool write_log_to_sd(const performance_log_entry_t *entry) {
    if (entry == NULL) {
        return false;
    }
    
#ifdef ENABLE_SD_CARD
    // Datei öffnen
    FIL file;
    FRESULT fr = f_open(&file, logger_config.sd_filename, FA_WRITE | FA_OPEN_APPEND);
    if (fr != FR_OK) {
        return false;
    }
    
    // Log-Eintrag schreiben (CSV-Format)
    f_printf(&file, "%lu,%lu,%lu,%u,%u,%u,%u\n",
             entry->timestamp,
             entry->inference_time,
             entry->peak_ram_usage,
             entry->cpu_load,
             entry->temperature,
             entry->prediction,
             entry->confidence);
    
    // Datei schließen
    f_close(&file);
    
    return true;
#else
    return false;
#endif
}

static bool add_log_to_buffer(const performance_log_entry_t *entry) {
    if (entry == NULL) {
        return false;
    }
    
    // Wenn der Puffer voll ist, ältesten Eintrag überschreiben
    if (log_buffer_count == MAX_LOG_ENTRIES) {
        log_buffer_tail = (log_buffer_tail + 1) % MAX_LOG_ENTRIES;
        log_buffer_count--;
    }
    
    // Neuen Eintrag hinzufügen
    memcpy(&log_buffer[log_buffer_head], entry, sizeof(performance_log_entry_t));
    log_buffer_head = (log_buffer_head + 1) % MAX_LOG_ENTRIES;
    log_buffer_count++;
    
    return true;
}