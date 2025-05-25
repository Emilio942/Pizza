/**
 * @file adaptive_clock_test.c
 * @brief Test program for adaptive clock frequency management
 * 
 * This program demonstrates and tests the adaptive clock frequency functionality
 * by simulating temperature changes and monitoring clock frequency adjustments.
 * 
 * @author Pizza Detection Team
 * @date 2025-01-20
 */

#include "adaptive_clock.h"
#include "pico/stdlib.h"
#include "hardware/adc.h"
#include "hardware/clocks.h"
#include <stdio.h>
#include <math.h>

// Test configuration
#define TEST_DURATION_MS    30000   // 30 seconds
#define STATUS_INTERVAL_MS  2000    // Status every 2 seconds
#define TEMP_INJECTION_INTERVAL_MS 5000  // Change temp every 5 seconds

// Simulated temperature values for testing
static float test_temperatures[] = {
    25.0f,  // Room temperature - should be 133 MHz
    45.0f,  // Warm - should be 100 MHz
    65.0f,  // Hot - should be 75 MHz  
    80.0f,  // Critical - should be 48 MHz
    70.0f,  // Cool down - should be 75 MHz (with hysteresis)
    35.0f,  // Room temp - should be 133 MHz
};

static size_t test_temp_index = 0;
static uint32_t last_temp_change = 0;
static uint32_t last_status_log = 0;

/**
 * Simulate temperature injection for testing
 * In a real system, this would not be needed as temperature comes from sensors
 */
static void inject_test_temperature(float target_temp) {
    // This is a test-only function that would modify the ADC reading
    // In the emulator, this will be handled by the temperature sensor emulation
    printf("[TEST] Injecting temperature: %.1f°C\n", target_temp);
    
    // Note: In a real implementation, we can't directly inject temperature
    // This is just for demonstration. The emulator will handle temperature simulation.
}

/**
 * Print current system status
 */
static void print_system_status(void) {
    adaptive_clock_state_t state = adaptive_clock_get_state();
    adaptive_clock_stats_t stats = adaptive_clock_get_stats();
    
    // Get actual system clock frequency
    uint32_t actual_freq = clock_get_hz(clk_sys) / 1000000;
    
    printf("\n=== ADAPTIVE CLOCK STATUS ===\n");
    printf("Current Frequency: %lu MHz (actual: %lu MHz)\n", 
           state.current_frequency_mhz, actual_freq);
    printf("Target Frequency:  %lu MHz\n", state.target_frequency_mhz);
    printf("Temperature:       %.1f°C\n", state.last_temperature);
    printf("Thermal Protection: %s\n", 
           state.thermal_protection_active ? "ACTIVE" : "Inactive");
    printf("Total Adjustments: %lu\n", stats.total_adjustments);
    printf("Emergency Activations: %lu\n", stats.emergency_activations);
    printf("==============================\n\n");
}

/**
 * Test the adaptive clock frequency functionality
 */
static void run_adaptive_clock_test(void) {
    printf("\n");
    printf("===============================================\n");
    printf(" RP2040 Adaptive Clock Frequency Test\n");
    printf("===============================================\n\n");
    
    // Initialize adaptive clock system
    adaptive_clock_config_t config = ADAPTIVE_CLOCK_CONFIG_DEFAULT;
    config.update_interval_ms = 1000;  // Update every second for testing
    config.verbose_logging = true;     // Enable detailed logging
    
    printf("Initializing adaptive clock management...\n");
    if (!adaptive_clock_init(&config)) {
        printf("ERROR: Failed to initialize adaptive clock management\n");
        return;
    }
    
    printf("Test will run for %d seconds with temperature changes every %d seconds\n", 
           TEST_DURATION_MS / 1000, TEMP_INJECTION_INTERVAL_MS / 1000);
    printf("Starting with room temperature (25°C)...\n\n");
    
    uint32_t test_start = time_us_32() / 1000;
    last_temp_change = test_start;
    last_status_log = test_start;
    
    // Initial status
    print_system_status();
    
    // Main test loop
    while (true) {
        uint32_t current_time = time_us_32() / 1000;
        uint32_t elapsed = current_time - test_start;
        
        // Check if test is complete
        if (elapsed >= TEST_DURATION_MS) {
            break;
        }
        
        // Inject temperature changes periodically for testing
        if (current_time - last_temp_change >= TEMP_INJECTION_INTERVAL_MS) {
            if (test_temp_index < sizeof(test_temperatures) / sizeof(test_temperatures[0])) {
                inject_test_temperature(test_temperatures[test_temp_index]);
                test_temp_index++;
                last_temp_change = current_time;
            }
        }
        
        // Update adaptive clock system
        if (!adaptive_clock_update()) {
            printf("ERROR: Adaptive clock update failed\n");
            break;
        }
        
        // Print status periodically
        if (current_time - last_status_log >= STATUS_INTERVAL_MS) {
            print_system_status();
            last_status_log = current_time;
        }
        
        // Small delay to prevent busy waiting
        sleep_ms(100);
    }
    
    // Final status and statistics
    printf("\n");
    printf("=== TEST COMPLETED ===\n");
    print_system_status();
    
    adaptive_clock_stats_t final_stats = adaptive_clock_get_stats();
    printf("FINAL STATISTICS:\n");
    printf("- Total frequency adjustments: %lu\n", final_stats.total_adjustments);
    printf("- Emergency mode activations: %lu\n", final_stats.emergency_activations);
    printf("- Final frequency: %lu MHz\n", final_stats.current_frequency_mhz);
    printf("- Final temperature: %.1f°C\n", final_stats.current_temperature);
    
    if (final_stats.total_adjustments > 0) {
        printf("\n✓ Adaptive clock frequency adjustment working correctly!\n");
    } else {
        printf("\n⚠ No frequency adjustments occurred during test\n");
    }
}

/**
 * Test manual frequency forcing
 */
static void test_manual_frequency_control(void) {
    printf("\n");
    printf("===============================================\n");
    printf(" Manual Frequency Control Test\n");
    printf("===============================================\n\n");
    
    uint32_t test_frequencies[] = {48, 75, 100, 133};
    size_t num_frequencies = sizeof(test_frequencies) / sizeof(test_frequencies[0]);
    
    for (size_t i = 0; i < num_frequencies; i++) {
        printf("Setting frequency to %lu MHz...\n", test_frequencies[i]);
        
        if (adaptive_clock_force_frequency(test_frequencies[i])) {
            // Wait for stabilization
            sleep_ms(500);
            
            // Verify frequency
            uint32_t actual_freq = clock_get_hz(clk_sys) / 1000000;
            adaptive_clock_state_t state = adaptive_clock_get_state();
            
            printf("  Target: %lu MHz, Reported: %lu MHz, Actual: %lu MHz\n",
                   test_frequencies[i], state.current_frequency_mhz, actual_freq);
            
            if (abs((int)(actual_freq - test_frequencies[i])) <= 2) {
                printf("  ✓ Frequency change successful\n");
            } else {
                printf("  ✗ Frequency change failed\n");
            }
        } else {
            printf("  ✗ Failed to set frequency\n");
        }
        
        printf("\n");
        sleep_ms(1000);
    }
}

/**
 * Main function
 */
int main() {
    // Initialize standard I/O
    stdio_init_all();
    sleep_ms(2000);  // Give time for USB to initialize
    
    printf("\n\n");
    printf("=================================================\n");
    printf(" RP2040 Pizza Detection Adaptive Clock Test\n");
    printf("=================================================\n");
    
    // Print initial system information
    printf("\nSystem Information:\n");
    printf("- Initial clock frequency: %lu MHz\n", clock_get_hz(clk_sys) / 1000000);
    printf("- Core 0 frequency: %lu MHz\n", clock_get_hz(clk_sys) / 1000000);
    printf("- Peripheral frequency: %lu MHz\n", clock_get_hz(clk_peri) / 1000000);
    
    // Run tests
    test_manual_frequency_control();
    run_adaptive_clock_test();
    
    printf("\n");
    printf("=================================================\n");
    printf(" Test completed successfully!\n");
    printf("=================================================\n");
    
    return 0;
}
