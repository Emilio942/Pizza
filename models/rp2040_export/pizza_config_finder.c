/**
 * Pizza Model Optimization Configuration Finder
 * Helps determine the optimal configuration for CMSIS-NN acceleration
 */

#include "pizza_model.h"
#include "pizza_benchmark.h"
#include "performance_logger.h"
#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "hardware/timer.h"

// Optimization level names for display
static const char* OPTIMIZATION_LEVEL_NAMES[] = {
    "None",
    "Basic (Conv only)",
    "Standard (Conv+Depthwise)",
    "Maximum (All layers)"
};

// Test image data
extern uint8_t test_image[CAMERA_WIDTH * CAMERA_HEIGHT * 3];

// Preprocessed tensors and output buffers
static float tensor_buffer[MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3];
static float probability_buffer[MODEL_NUM_CLASSES];

// Number of iterations per configuration test
#define CONFIG_TEST_ITERATIONS 20

// A mini report on a specific optimization configuration
typedef struct {
    int optimization_level;
    uint32_t avg_inference_time_us;
    uint32_t max_ram_usage;
    float speedup_vs_baseline;
    float power_usage_factor;  // Relative to baseline
} optimization_config_report_t;

/**
 * Finds the optimal hardware optimization configuration for this specific model and device
 * by testing different combinations of optimization levels and measuring performance
 */
void pizza_find_optimal_config(void) {
    optimization_config_report_t reports[4];  // One for each optimization level
    uint32_t baseline_time = 0;
    
    printf("Testing different optimization configurations...\n");
    
    // Initialize model
    pizza_model_init();
    
    // Preprocess test image once (this is common to all configurations)
    pizza_model_preprocess(test_image, tensor_buffer);
    
    // First run with no optimizations (baseline)
    pizza_model_set_hardware_optimization(false);
    
    // Measure baseline performance
    uint32_t total_time = 0;
    for (int i = 0; i < CONFIG_TEST_ITERATIONS; i++) {
        uint32_t start_time = time_us_32();
        pizza_model_infer(tensor_buffer, probability_buffer);
        uint32_t end_time = time_us_32();
        
        total_time += (end_time - start_time);
    }
    
    baseline_time = total_time / CONFIG_TEST_ITERATIONS;
    
    // Store baseline metrics
    uint32_t baseline_ram_usage;
    performance_logger_get_stats(NULL, NULL, NULL, &baseline_ram_usage, NULL);
    performance_logger_clear();
    
    reports[0].optimization_level = 0;
    reports[0].avg_inference_time_us = baseline_time;
    reports[0].max_ram_usage = baseline_ram_usage;
    reports[0].speedup_vs_baseline = 1.0f;  // By definition
    reports[0].power_usage_factor = 1.0f;   // By definition
    
    printf("Baseline established: %lu µs average inference time\n", baseline_time);
    
    // Enable hardware optimization for further tests
    pizza_model_set_hardware_optimization(true);
    
    // Test each optimization level
    for (int level = 1; level <= 3; level++) {
        printf("Testing optimization level %d (%s)...\n", 
               level, OPTIMIZATION_LEVEL_NAMES[level]);
        
        // Temporarily override the optimization level
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wuninitialized"
        #undef PIZZA_CONV_OPTIMIZATION_LEVEL
        #define PIZZA_CONV_OPTIMIZATION_LEVEL level
        #pragma GCC diagnostic pop
        
        // Measure performance at this level
        total_time = 0;
        for (int i = 0; i < CONFIG_TEST_ITERATIONS; i++) {
            uint32_t start_time = time_us_32();
            pizza_model_infer(tensor_buffer, probability_buffer);
            uint32_t end_time = time_us_32();
            
            total_time += (end_time - start_time);
        }
        
        uint32_t avg_time = total_time / CONFIG_TEST_ITERATIONS;
        
        // Get RAM usage for this configuration
        uint32_t ram_usage;
        performance_logger_get_stats(NULL, NULL, NULL, &ram_usage, NULL);
        performance_logger_clear();
        
        // Store metrics
        reports[level].optimization_level = level;
        reports[level].avg_inference_time_us = avg_time;
        reports[level].max_ram_usage = ram_usage;
        reports[level].speedup_vs_baseline = (float)baseline_time / (float)avg_time;
        
        // Estimate power usage (simplistic model based on processing time)
        reports[level].power_usage_factor = 0.8f + (0.2f * avg_time / baseline_time);
        
        printf("Level %d complete: %lu µs average inference time (%.2fx speedup)\n", 
               level, avg_time, reports[level].speedup_vs_baseline);
    }
    
    // Print summary table
    printf("\n=== Optimization Configuration Report ===\n");
    printf("%-20s | %-15s | %-15s | %-15s | %-15s\n", 
           "Level", "Inference Time", "RAM Usage", "Speedup", "Power Factor");
    printf("-------------------+----------------+----------------+----------------+----------------\n");
    
    for (int i = 0; i < 4; i++) {
        printf("%-20s | %-15lu | %-15lu | %-15.2f | %-15.2f\n",
               OPTIMIZATION_LEVEL_NAMES[i],
               reports[i].avg_inference_time_us,
               reports[i].max_ram_usage,
               reports[i].speedup_vs_baseline,
               reports[i].power_usage_factor);
    }
    
    // Find and recommend optimal configuration
    int optimal_level = 0;
    float best_score = 0.0f;
    
    for (int i = 0; i < 4; i++) {
        // Scoring formula: balance speed, memory and power
        // Weight speed more heavily as it's the primary goal
        float score = (reports[i].speedup_vs_baseline * 0.7f) + 
                      (1.0f / reports[i].power_usage_factor * 0.2f) +
                      (baseline_ram_usage / (float)reports[i].max_ram_usage * 0.1f);
        
        if (score > best_score) {
            best_score = score;
            optimal_level = i;
        }
    }
    
    printf("\nRecommended configuration: Level %d (%s)\n", 
           optimal_level, OPTIMIZATION_LEVEL_NAMES[optimal_level]);
    printf("Set PIZZA_CONV_OPTIMIZATION_LEVEL=%d in your build\n", optimal_level);
}