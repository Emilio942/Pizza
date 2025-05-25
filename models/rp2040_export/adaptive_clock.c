/**
 * @file adaptive_clock.c
 * @brief Adaptive clock frequency management for RP2040 based on temperature monitoring
 * 
 * This module implements temperature-based adaptive clock frequency adjustment
 * for the RP2040 microcontroller to prevent overheating and optimize power consumption.
 * 
 * Temperature thresholds and corresponding clock frequencies:
 * - Below 40°C: Maximum performance (133 MHz)
 * - 40-60°C: Balanced mode (100 MHz) 
 * - 60-75°C: Conservative mode (75 MHz)
 * - Above 75°C: Emergency mode (48 MHz) - thermal protection
 * 
 * @author Pizza Detection Team
 * @date 2025-01-20
 */

#include "adaptive_clock.h"
#include "pico/stdlib.h"
#include "hardware/clocks.h"
#include "hardware/pll.h"
#include "hardware/vreg.h"
#include "hardware/adc.h"
#include "hardware/gpio.h"
#include <stdio.h>
#include <math.h>

// Temperature thresholds in Celsius
#define TEMP_THRESHOLD_LOW      40.0f   // Below this: maximum performance
#define TEMP_THRESHOLD_MEDIUM   60.0f   // Above this: balanced mode
#define TEMP_THRESHOLD_HIGH     75.0f   // Above this: conservative mode
#define TEMP_THRESHOLD_CRITICAL 85.0f   // Above this: emergency mode

// Clock frequencies in MHz
#define CLOCK_FREQ_MAX          133     // Maximum performance
#define CLOCK_FREQ_BALANCED     100     // Balanced performance/thermal
#define CLOCK_FREQ_CONSERVATIVE 75      // Conservative mode
#define CLOCK_FREQ_EMERGENCY    48      // Emergency thermal protection

// Hysteresis to prevent oscillation (2°C deadband)
#define TEMP_HYSTERESIS         2.0f

// Global state
static adaptive_clock_config_t g_config;
static adaptive_clock_state_t g_state;
static bool g_initialized = false;

/**
 * Read temperature from internal RP2040 ADC sensor
 */
static float read_internal_temperature(void) {
    // Select ADC input 4 (temperature sensor)
    adc_select_input(4);
    
    // Read raw ADC value
    uint16_t raw = adc_read();
    
    // Convert to voltage
    const float conversion_factor = 3.3f / 4096.0f;
    float voltage = raw * conversion_factor;
    
    // Convert to temperature using RP2040 formula
    // T = 27 - (ADC_voltage - 0.706)/0.001721
    float temperature = 27.0f - (voltage - 0.706f) / 0.001721f;
    
    return temperature;
}

/**
 * Determine target clock frequency based on temperature
 */
static uint32_t determine_target_frequency(float temperature) {
    uint32_t target_freq;
    
    // Apply hysteresis to prevent oscillation
    float temp_adjusted = temperature;
    if (g_state.last_adjustment_direction == CLOCK_ADJUSTMENT_UP) {
        temp_adjusted -= TEMP_HYSTERESIS;
    } else if (g_state.last_adjustment_direction == CLOCK_ADJUSTMENT_DOWN) {
        temp_adjusted += TEMP_HYSTERESIS;
    }
    
    // Determine frequency based on temperature thresholds
    if (temp_adjusted >= TEMP_THRESHOLD_CRITICAL) {
        target_freq = CLOCK_FREQ_EMERGENCY;
        g_state.thermal_protection_active = true;
    } else if (temp_adjusted >= TEMP_THRESHOLD_HIGH) {
        target_freq = CLOCK_FREQ_CONSERVATIVE;
        g_state.thermal_protection_active = false;
    } else if (temp_adjusted >= TEMP_THRESHOLD_MEDIUM) {
        target_freq = CLOCK_FREQ_BALANCED;
        g_state.thermal_protection_active = false;
    } else if (temp_adjusted >= TEMP_THRESHOLD_LOW) {
        target_freq = CLOCK_FREQ_BALANCED;
        g_state.thermal_protection_active = false;
    } else {
        target_freq = CLOCK_FREQ_MAX;
        g_state.thermal_protection_active = false;
    }
    
    return target_freq;
}

/**
 * Actually change the system clock frequency
 */
static bool set_system_clock_frequency(uint32_t freq_mhz) {
    // Store current frequency for comparison
    uint32_t current_freq = clock_get_hz(clk_sys) / 1000000;
    
    if (current_freq == freq_mhz) {
        return true; // Already at target frequency
    }
    
    // Log frequency change
    printf("[ADAPTIVE_CLOCK] Changing system clock: %lu MHz -> %lu MHz\n", 
           current_freq, freq_mhz);
    
    // Configure voltage regulator for frequency
    enum vreg_voltage voltage = VREG_VOLTAGE_1_10;
    if (freq_mhz >= 120) {
        voltage = VREG_VOLTAGE_1_20; // Higher voltage for high frequencies
    } else if (freq_mhz >= 80) {
        voltage = VREG_VOLTAGE_1_15; // Medium voltage 
    } else {
        voltage = VREG_VOLTAGE_1_10; // Lower voltage for efficiency
    }
    
    // Set voltage regulator
    vreg_set_voltage(voltage);
    sleep_ms(10); // Allow voltage to stabilize
    
    // Configure PLL for target frequency
    uint32_t vco_freq = freq_mhz * 6; // VCO frequency (MHz)
    if (vco_freq < 400 || vco_freq > 1600) {
        printf("[ADAPTIVE_CLOCK] ERROR: Invalid VCO frequency %lu MHz\n", vco_freq);
        return false;
    }
    
    // Calculate PLL settings
    uint8_t post_div1 = 6;  // Post divider 1
    uint8_t post_div2 = 1;  // Post divider 2
    
    // Stop system clock temporarily
    clock_stop(clk_sys);
    
    // Configure PLL SYS
    pll_init(pll_sys, 1, vco_freq * 1000000, post_div1, post_div2);
    
    // Configure system clock to use PLL SYS
    clock_configure(clk_sys,
                   CLOCKS_CLK_SYS_CTRL_SRC_VALUE_CLKSRC_CLK_SYS_AUX,
                   CLOCKS_CLK_SYS_CTRL_AUXSRC_VALUE_CLKSRC_PLL_SYS,
                   freq_mhz * MHZ,
                   freq_mhz * MHZ);
    
    // Update peripheral clocks if needed
    clock_configure(clk_peri,
                   0,
                   CLOCKS_CLK_PERI_CTRL_AUXSRC_VALUE_CLK_SYS,
                   freq_mhz * MHZ,
                   freq_mhz * MHZ);
    
    // Allow system to stabilize
    sleep_ms(5);
    
    // Verify frequency change
    uint32_t actual_freq = clock_get_hz(clk_sys) / 1000000;
    if (abs((int)(actual_freq - freq_mhz)) > 2) {
        printf("[ADAPTIVE_CLOCK] WARNING: Frequency mismatch: target %lu MHz, actual %lu MHz\n", 
               freq_mhz, actual_freq);
        return false;
    }
    
    printf("[ADAPTIVE_CLOCK] System clock successfully changed to %lu MHz\n", actual_freq);
    return true;
}

/**
 * Initialize adaptive clock frequency management
 */
bool adaptive_clock_init(const adaptive_clock_config_t* config) {
    if (!config) {
        return false;
    }
    
    // Copy configuration
    g_config = *config;
    
    // Initialize state
    g_state.current_frequency_mhz = CLOCK_FREQ_MAX;
    g_state.target_frequency_mhz = CLOCK_FREQ_MAX;
    g_state.last_temperature = 25.0f;
    g_state.last_adjustment_time = 0;
    g_state.last_adjustment_direction = CLOCK_ADJUSTMENT_NONE;
    g_state.thermal_protection_active = false;
    g_state.total_adjustments = 0;
    g_state.emergency_activations = 0;
    
    // Initialize ADC for temperature reading
    adc_init();
    adc_set_temp_sensor_enabled(true);
    
    printf("[ADAPTIVE_CLOCK] Initialized with update interval: %lu ms\n", 
           g_config.update_interval_ms);
    printf("[ADAPTIVE_CLOCK] Temperature thresholds: %.1f/%.1f/%.1f/%.1f°C\n",
           TEMP_THRESHOLD_LOW, TEMP_THRESHOLD_MEDIUM, 
           TEMP_THRESHOLD_HIGH, TEMP_THRESHOLD_CRITICAL);
    printf("[ADAPTIVE_CLOCK] Clock frequencies: %d/%d/%d/%d MHz\n",
           CLOCK_FREQ_MAX, CLOCK_FREQ_BALANCED, 
           CLOCK_FREQ_CONSERVATIVE, CLOCK_FREQ_EMERGENCY);
    
    g_initialized = true;
    return true;
}

/**
 * Update adaptive clock frequency based on current temperature
 */
bool adaptive_clock_update(void) {
    if (!g_initialized) {
        return false;
    }
    
    uint32_t current_time = time_us_32() / 1000; // Convert to ms
    
    // Check if enough time has passed since last update
    if (current_time - g_state.last_adjustment_time < g_config.update_interval_ms) {
        return true; // Not time to update yet
    }
    
    // Read current temperature
    float temperature = read_internal_temperature();
    g_state.last_temperature = temperature;
    
    // Determine target frequency
    uint32_t target_freq = determine_target_frequency(temperature);
    g_state.target_frequency_mhz = target_freq;
    
    // Check if frequency adjustment is needed
    if (target_freq != g_state.current_frequency_mhz) {
        // Determine adjustment direction
        if (target_freq > g_state.current_frequency_mhz) {
            g_state.last_adjustment_direction = CLOCK_ADJUSTMENT_UP;
        } else {
            g_state.last_adjustment_direction = CLOCK_ADJUSTMENT_DOWN;
        }
        
        // Apply frequency change
        if (set_system_clock_frequency(target_freq)) {
            g_state.current_frequency_mhz = target_freq;
            g_state.total_adjustments++;
            
            if (target_freq == CLOCK_FREQ_EMERGENCY) {
                g_state.emergency_activations++;
                printf("[ADAPTIVE_CLOCK] EMERGENCY: Thermal protection activated at %.1f°C\n", 
                       temperature);
            }
            
            // Log the adjustment
            if (g_config.verbose_logging) {
                printf("[ADAPTIVE_CLOCK] Frequency adjusted: %lu MHz (temp: %.1f°C)\n", 
                       target_freq, temperature);
            }
            
        } else {
            printf("[ADAPTIVE_CLOCK] ERROR: Failed to set frequency to %lu MHz\n", target_freq);
            return false;
        }
    } else if (g_config.verbose_logging && (current_time % 10000) == 0) {
        // Log status every 10 seconds when verbose
        printf("[ADAPTIVE_CLOCK] Status: %lu MHz, %.1f°C\n", 
               g_state.current_frequency_mhz, temperature);
    }
    
    g_state.last_adjustment_time = current_time;
    return true;
}

/**
 * Get current adaptive clock state
 */
adaptive_clock_state_t adaptive_clock_get_state(void) {
    return g_state;
}

/**
 * Get adaptive clock statistics  
 */
adaptive_clock_stats_t adaptive_clock_get_stats(void) {
    adaptive_clock_stats_t stats;
    
    stats.total_adjustments = g_state.total_adjustments;
    stats.emergency_activations = g_state.emergency_activations;
    stats.current_frequency_mhz = g_state.current_frequency_mhz;
    stats.current_temperature = g_state.last_temperature;
    stats.thermal_protection_active = g_state.thermal_protection_active;
    
    return stats;
}

/**
 * Force specific clock frequency (for testing)
 */
bool adaptive_clock_force_frequency(uint32_t freq_mhz) {
    if (!g_initialized) {
        return false;
    }
    
    printf("[ADAPTIVE_CLOCK] Forcing frequency to %lu MHz\n", freq_mhz);
    
    if (set_system_clock_frequency(freq_mhz)) {
        g_state.current_frequency_mhz = freq_mhz;
        g_state.target_frequency_mhz = freq_mhz;
        return true;
    }
    
    return false;
}

/**
 * Enable or disable adaptive clock management
 */
void adaptive_clock_set_enabled(bool enabled) {
    g_config.enabled = enabled;
    printf("[ADAPTIVE_CLOCK] Adaptive clock management %s\n", 
           enabled ? "enabled" : "disabled");
}

/**
 * Check if thermal protection is currently active
 */
bool adaptive_clock_is_thermal_protection_active(void) {
    return g_state.thermal_protection_active;
}
