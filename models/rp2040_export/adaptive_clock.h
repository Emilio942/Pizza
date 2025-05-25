/**
 * @file adaptive_clock.h
 * @brief Adaptive clock frequency management for RP2040 based on temperature monitoring
 * 
 * This header defines the interface for temperature-based adaptive clock frequency
 * adjustment for the RP2040 microcontroller.
 * 
 * @author Pizza Detection Team
 * @date 2025-01-20
 */

#ifndef ADAPTIVE_CLOCK_H
#define ADAPTIVE_CLOCK_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Clock adjustment direction for hysteresis tracking
 */
typedef enum {
    CLOCK_ADJUSTMENT_NONE = 0,
    CLOCK_ADJUSTMENT_UP,
    CLOCK_ADJUSTMENT_DOWN
} clock_adjustment_direction_t;

/**
 * Configuration for adaptive clock frequency management
 */
typedef struct {
    bool enabled;                   ///< Enable/disable adaptive clock management
    uint32_t update_interval_ms;    ///< Update interval in milliseconds
    bool verbose_logging;           ///< Enable detailed logging
} adaptive_clock_config_t;

/**
 * Current state of adaptive clock management
 */
typedef struct {
    uint32_t current_frequency_mhz;        ///< Current system clock frequency in MHz
    uint32_t target_frequency_mhz;         ///< Target frequency based on temperature
    float last_temperature;                ///< Last measured temperature in °C
    uint32_t last_adjustment_time;         ///< Time of last adjustment (ms)
    clock_adjustment_direction_t last_adjustment_direction; ///< Direction of last adjustment
    bool thermal_protection_active;       ///< True if thermal protection is active
    uint32_t total_adjustments;           ///< Total number of frequency adjustments
    uint32_t emergency_activations;       ///< Number of emergency mode activations
} adaptive_clock_state_t;

/**
 * Statistics for adaptive clock management
 */
typedef struct {
    uint32_t total_adjustments;           ///< Total frequency adjustments
    uint32_t emergency_activations;       ///< Emergency mode activations
    uint32_t current_frequency_mhz;       ///< Current frequency in MHz
    float current_temperature;            ///< Current temperature in °C
    bool thermal_protection_active;       ///< Thermal protection status
} adaptive_clock_stats_t;

/**
 * @brief Initialize adaptive clock frequency management
 * 
 * Initializes the adaptive clock frequency system with the specified configuration.
 * Must be called before using any other functions.
 * 
 * @param config Pointer to configuration structure
 * @return true if initialization successful, false otherwise
 */
bool adaptive_clock_init(const adaptive_clock_config_t* config);

/**
 * @brief Update adaptive clock frequency based on current temperature
 * 
 * Reads the current temperature and adjusts the system clock frequency
 * if necessary based on predefined thresholds. Should be called
 * periodically (e.g., in main loop or timer interrupt).
 * 
 * @return true if update successful, false on error
 */
bool adaptive_clock_update(void);

/**
 * @brief Get current adaptive clock state
 * 
 * Returns the current state of the adaptive clock management system
 * including current frequency, temperature, and adjustment history.
 * 
 * @return Current state structure
 */
adaptive_clock_state_t adaptive_clock_get_state(void);

/**
 * @brief Get adaptive clock statistics
 * 
 * Returns statistical information about the adaptive clock system
 * including total adjustments and emergency activations.
 * 
 * @return Statistics structure
 */
adaptive_clock_stats_t adaptive_clock_get_stats(void);

/**
 * @brief Force specific clock frequency (for testing)
 * 
 * Forces the system clock to a specific frequency, bypassing
 * temperature-based control. Useful for testing and debugging.
 * 
 * @param freq_mhz Target frequency in MHz
 * @return true if frequency change successful, false otherwise
 */
bool adaptive_clock_force_frequency(uint32_t freq_mhz);

/**
 * @brief Enable or disable adaptive clock management
 * 
 * Enables or disables the adaptive clock frequency management.
 * When disabled, the system will maintain its current frequency.
 * 
 * @param enabled true to enable, false to disable
 */
void adaptive_clock_set_enabled(bool enabled);

/**
 * @brief Check if thermal protection is currently active
 * 
 * Returns whether thermal protection mode is currently active
 * (emergency frequency mode due to high temperature).
 * 
 * @return true if thermal protection active, false otherwise
 */
bool adaptive_clock_is_thermal_protection_active(void);

// Predefined configurations for common use cases

/**
 * Default configuration for balanced performance and thermal management
 */
#define ADAPTIVE_CLOCK_CONFIG_DEFAULT { \
    .enabled = true, \
    .update_interval_ms = 1000, \
    .verbose_logging = false \
}

/**
 * Configuration for aggressive thermal protection  
 */
#define ADAPTIVE_CLOCK_CONFIG_AGGRESSIVE { \
    .enabled = true, \
    .update_interval_ms = 500, \
    .verbose_logging = true \
}

/**
 * Configuration for minimal intervention (conservative adjustment)
 */
#define ADAPTIVE_CLOCK_CONFIG_CONSERVATIVE { \
    .enabled = true, \
    .update_interval_ms = 2000, \
    .verbose_logging = false \
}

#ifdef __cplusplus
}
#endif

#endif // ADAPTIVE_CLOCK_H
