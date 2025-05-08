/**
 * Pizza Model CMSIS-NN Benchmarking Utility
 * Measures performance improvements from hardware optimization
 */

#include "pizza_model.h"
#include "pizza_model_cmsis.h"
#include "performance_logger.h"
#include "hardware/timer.h"
#include <stdio.h>
#include <string.h>

// Test image data (could be loaded from flash or camera)
static uint8_t test_image[CAMERA_WIDTH * CAMERA_HEIGHT * 3];

// Preprocessed tensors and output buffers
static float input_tensor[MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3];
static float output_probs[MODEL_NUM_CLASSES];

// Number of iterations for benchmark
#define BENCHMARK_ITERATIONS 100

/**
 * Pizza CMSIS-NN benchmark results
 */
typedef struct {
    uint32_t standard_avg_time_us;   // Average inference time without optimization (µs)
    uint32_t cmsis_avg_time_us;      // Average inference time with CMSIS-NN (µs)
    uint32_t standard_max_time_us;   // Maximum inference time without optimization (µs)
    uint32_t cmsis_max_time_us;      // Maximum inference time with CMSIS-NN (µs)
    float speedup_factor;            // Speed improvement factor
    uint32_t standard_ram_usage;     // RAM usage without optimization (bytes)
    uint32_t cmsis_ram_usage;        // RAM usage with CMSIS-NN (bytes)
} pizza_benchmark_results_t;

/**
 * Run inference benchmark with and without CMSIS-NN acceleration
 * and measure performance differences
 * 
 * @param results Pointer to store benchmark results
 * @return true if benchmark succeeded, false on error
 */
bool pizza_benchmark_run(pizza_benchmark_results_t* results) {
    uint32_t total_standard_time = 0;
    uint32_t total_cmsis_time = 0;
    uint32_t max_standard_time = 0;
    uint32_t max_cmsis_time = 0;
    
    // Initialize model and logger
    pizza_model_init();
    
    // Configure performance logger
    performance_logger_config_t logger_config = {
        .log_to_uart = true,
        .log_to_sd = false,
        .log_to_ram = true,
        .log_interval = 0,
        .uart_port = 0,
        .sd_filename = "benchmark.csv"
    };
    
    if (!performance_logger_init(logger_config)) {
        printf("Failed to initialize performance logger\n");
        return false;
    }
    
    // Preprocess test image once (this is common to both implementations)
    pizza_model_preprocess(test_image, input_tensor);
    
    // First run standard implementation
    pizza_model_set_hardware_optimization(false);
    
    printf("Running benchmark with standard implementation...\n");
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        uint32_t start_time = time_us_32();
        
        // Run inference
        pizza_model_infer(input_tensor, output_probs);
        
        uint32_t end_time = time_us_32();
        uint32_t duration = end_time - start_time;
        
        total_standard_time += duration;
        if (duration > max_standard_time) {
            max_standard_time = duration;
        }
    }
    
    // Get standard implementation RAM usage
    uint32_t standard_ram_usage;
    uint32_t standard_max_inference_time;
    performance_logger_get_stats(NULL, &standard_max_inference_time, NULL, &standard_ram_usage, NULL);
    
    // Clear logs for next test
    performance_logger_clear();
    
    // Now run CMSIS-NN implementation
    pizza_model_set_hardware_optimization(true);
    
    printf("Running benchmark with CMSIS-NN optimization...\n");
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        uint32_t start_time = time_us_32();
        
        // Run inference
        pizza_model_infer(input_tensor, output_probs);
        
        uint32_t end_time = time_us_32();
        uint32_t duration = end_time - start_time;
        
        total_cmsis_time += duration;
        if (duration > max_cmsis_time) {
            max_cmsis_time = duration;
        }
    }
    
    // Get CMSIS-NN implementation RAM usage
    uint32_t cmsis_ram_usage;
    uint32_t cmsis_max_inference_time;
    performance_logger_get_stats(NULL, &cmsis_max_inference_time, NULL, &cmsis_ram_usage, NULL);
    
    // Calculate average times
    uint32_t avg_standard_time = total_standard_time / BENCHMARK_ITERATIONS;
    uint32_t avg_cmsis_time = total_cmsis_time / BENCHMARK_ITERATIONS;
    
    // Calculate speedup factor
    float speedup = (float)avg_standard_time / (float)avg_cmsis_time;
    
    // Store results
    if (results) {
        results->standard_avg_time_us = avg_standard_time;
        results->cmsis_avg_time_us = avg_cmsis_time;
        results->standard_max_time_us = max_standard_time;
        results->cmsis_max_time_us = max_cmsis_time;
        results->speedup_factor = speedup;
        results->standard_ram_usage = standard_ram_usage;
        results->cmsis_ram_usage = cmsis_ram_usage;
    }
    
    // Print results
    printf("=== CMSIS-NN Benchmark Results ===\n");
    printf("Standard Implementation:\n");
    printf("  Average Inference Time: %lu µs\n", avg_standard_time);
    printf("  Maximum Inference Time: %lu µs\n", max_standard_time);
    printf("  Peak RAM Usage: %lu bytes\n", standard_ram_usage);
    printf("\n");
    printf("CMSIS-NN Optimized Implementation:\n");
    printf("  Average Inference Time: %lu µs\n", avg_cmsis_time);
    printf("  Maximum Inference Time: %lu µs\n", max_cmsis_time);
    printf("  Peak RAM Usage: %lu bytes\n", cmsis_ram_usage);
    printf("\n");
    printf("Performance Improvement:\n");
    printf("  Speedup Factor: %.2fx\n", speedup);
    printf("  Time Reduction: %.1f%%\n", (1.0f - 1.0f/speedup) * 100.0f);
    printf("  RAM Usage Difference: %ld bytes\n", (int32_t)cmsis_ram_usage - (int32_t)standard_ram_usage);
    
    // Clean up
    performance_logger_deinit();
    
    return true;
}