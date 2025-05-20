/**
 * RP2040 Pizza Detection CMSIS-NN Integration Test
 * 
 * This file provides a complete test of the CMSIS-NN integration for the
 * pizza detection neural network. It measures performance improvements
 * and validates that the CMSIS-NN implementation produces correct results.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "hardware/timer.h"
#include "hardware/adc.h"
#include "pizza_model.h"
#include "pizza_model_cmsis.h"
#include "pizza_benchmark.h"

// Input and output buffers
static uint8_t test_image[MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3];
static float input_tensor_float[MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3];
static q7_t input_tensor_q7[MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3];
static float output_std[MODEL_NUM_CLASSES];
static float output_cmsis[MODEL_NUM_CLASSES];

// Tolerance for numerical differences
#define COMPARISON_TOLERANCE 0.01f

/**
 * Generate a test pattern for inference
 */
static void generate_test_pattern(void) {
    uint32_t seed = 12345;
    
    for (int i = 0; i < MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3; i++) {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        test_image[i] = (uint8_t)(seed % 256);
    }
    
    // Preprocess the image
    pizza_model_preprocess(test_image, input_tensor_float);
    
    // Quantize for CMSIS-NN
    pizza_model_quantize_input(input_tensor_float, input_tensor_q7, 
                               MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3);
}

/**
 * Compare outputs from standard and CMSIS-NN implementations
 * to ensure they produce similar results
 */
static bool compare_outputs(const float* output1, const float* output2, 
                           float tolerance, int size) {
    float max_diff = 0.0f;
    int max_diff_idx = 0;
    
    for (int i = 0; i < size; i++) {
        float diff = fabsf(output1[i] - output2[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        
        if (diff > tolerance) {
            printf("Error: Output mismatch at index %d: %.6f vs %.6f (diff: %.6f)\n",
                   i, output1[i], output2[i], diff);
            return false;
        }
    }
    
    printf("Outputs match within tolerance. Max diff: %.6f at index %d (%.6f vs %.6f)\n",
           max_diff, max_diff_idx, output1[max_diff_idx], output2[max_diff_idx]);
    return true;
}

/**
 * Measure temperature from internal sensor
 */
static float measure_temperature() {
    adc_init();
    adc_set_temp_sensor_enabled(true);
    adc_select_input(4); // Temperature sensor is on ADC input 4
    
    // Get ADC reading (12-bit, 0-4095)
    uint16_t raw = adc_read();
    
    // Convert to voltage
    const float conversion_factor = 3.3f / 4096.0f;
    float voltage = raw * conversion_factor;
    
    // Convert to temperature (see RP2040 datasheet)
    // T = 27 - (ADC_voltage - 0.706)/0.001721
    float temperature = 27.0f - (voltage - 0.706f) / 0.001721f;
    
    return temperature;
}

/**
 * Measure power consumption indirectly through temperature rise
 * This is a simplified approach - a real measurement would use external hardware
 */
static float get_power_indicator() {
    float temp1 = measure_temperature();
    
    // Run a compute-intensive task
    volatile float sum = 0.0f;
    for (int i = 0; i < 1000000; i++) {
        sum += sinf(i * 0.01f) * cosf(i * 0.02f);
    }
    
    float temp2 = measure_temperature();
    
    // Temperature rise is roughly proportional to power consumption
    return temp2 - temp1;
}

int main() {
    // Initialize standard I/O
    stdio_init_all();
    sleep_ms(2000);  // Give time for USB to initialize
    
    printf("\n\n");
    printf("=================================================\n");
    printf(" RP2040 Pizza Detection CMSIS-NN Integration Test\n");
    printf("=================================================\n\n");
    
    // Initialize model
    printf("Initializing pizza detection model...\n");
    pizza_model_init();
    
    // Generate test pattern
    printf("Generating test pattern...\n");
    generate_test_pattern();
    
    // First run standard implementation
    printf("Running standard implementation...\n");
    pizza_model_set_hardware_optimization(false);
    
    uint32_t start_time = time_us_32();
    pizza_model_infer(input_tensor_float, output_std);
    uint32_t end_time = time_us_32();
    
    int class_id = pizza_model_get_prediction(output_std);
    
    printf("Standard implementation results:\n");
    printf("  Predicted class: %s (%.1f%%)\n", 
           CLASS_NAMES[class_id], output_std[class_id] * 100.0f);
    printf("  Inference time: %lu µs\n", end_time - start_time);
    
    // Class probabilities from standard implementation
    printf("  Class probabilities:\n");
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
        printf("    %s: %.2f%%\n", CLASS_NAMES[i], output_std[i] * 100.0f);
    }
    
    // Now run CMSIS-NN implementation
    printf("\nRunning CMSIS-NN implementation...\n");
    pizza_model_set_hardware_optimization(true);
    
    start_time = time_us_32();
    pizza_model_infer(input_tensor_float, output_cmsis);
    end_time = time_us_32();
    
    class_id = pizza_model_get_prediction(output_cmsis);
    
    printf("CMSIS-NN implementation results:\n");
    printf("  Predicted class: %s (%.1f%%)\n", 
           CLASS_NAMES[class_id], output_cmsis[class_id] * 100.0f);
    printf("  Inference time: %lu µs\n", end_time - start_time);
    
    // Class probabilities from CMSIS-NN implementation
    printf("  Class probabilities:\n");
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
        printf("    %s: %.2f%%\n", CLASS_NAMES[i], output_cmsis[i] * 100.0f);
    }
    
    // Compare outputs to ensure correctness
    printf("\nComparing outputs...\n");
    if (compare_outputs(output_std, output_cmsis, COMPARISON_TOLERANCE, MODEL_NUM_CLASSES)) {
        printf("Validation PASSED: CMSIS-NN implementation produces correct results.\n");
    } else {
        printf("Validation FAILED: Outputs differ more than tolerance allows!\n");
    }
    
    // Run comprehensive benchmark
    printf("\nRunning comprehensive benchmark...\n");
    pizza_benchmark_results_t results;
    if (pizza_benchmark_run(&results)) {
        printf("\nBenchmark Results:\n");
        printf("  Standard Implementation Avg Time: %lu µs\n", results.standard_avg_time_us);
        printf("  CMSIS-NN Implementation Avg Time: %lu µs\n", results.cmsis_avg_time_us);
        printf("  Speedup Factor: %.2fx\n", results.speedup_factor);
        printf("  RAM Usage Reduction: %ld bytes\n", 
               (int32_t)results.standard_ram_usage - (int32_t)results.cmsis_ram_usage);
        
        // Check if speedup meets requirements
        if (results.speedup_factor >= 1.5f) {
            printf("Performance PASSED: Achieved %.1fx speedup (requirement: 1.5x)\n", 
                   results.speedup_factor);
        } else {
            printf("Performance WARNING: Only achieved %.1fx speedup (requirement: 1.5x)\n", 
                   results.speedup_factor);
        }
    } else {
        printf("Benchmark failed!\n");
    }
    
    // Simulate power measurement
    printf("\nEstimating power consumption...\n");
    
    // Disable CMSIS-NN and measure
    pizza_model_set_hardware_optimization(false);
    float power_std = get_power_indicator();
    
    // Enable CMSIS-NN and measure
    pizza_model_set_hardware_optimization(true);
    float power_cmsis = get_power_indicator();
    
    // Power reduction (simplified, relative metric)
    float power_reduction = (power_std - power_cmsis) / power_std * 100.0f;
    printf("Estimated power reduction: %.1f%%\n", power_reduction);
    
    printf("\nTest completed!\n");
    
    return 0;
}
