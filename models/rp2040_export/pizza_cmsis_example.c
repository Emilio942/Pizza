/**
 * CMSIS-NN Optimization Example for RP2040 Pizza Detection
 * This example demonstrates the performance improvements
 * from using CMSIS-NN for neural network operations
 */

#include <stdio.h>
#include <stdlib.h>
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "hardware/timer.h"
#include "pizza_model.h"
#include "pizza_benchmark.h"
#include "camera_utils.h"

// RGB buffer for camera image
static uint8_t camera_buffer[CAMERA_WIDTH * CAMERA_HEIGHT * 3];

// Tensor and probability buffers
static float tensor_buffer[MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3];
static float probability_buffer[MODEL_NUM_CLASSES];

int main() {
    // Initialize standard I/O
    stdio_init_all();
    sleep_ms(2000);  // Give time for USB to initialize
    
    printf("\n\n");
    printf("===================================\n");
    printf("RP2040 Pizza Detection with CMSIS-NN\n");
    printf("===================================\n\n");
    
    // Initialize camera
    camera_init();
    
    // Initialize neural network model
    pizza_model_init();
    
    printf("Running standard inference...\n");
    
    // Capture image
    if (!camera_capture_image(camera_buffer)) {
        printf("Error capturing image!\n");
        return -1;
    }
    
    // Preprocess image
    pizza_model_preprocess(camera_buffer, tensor_buffer);
    
    // Run standard inference (without CMSIS-NN)
    pizza_model_set_hardware_optimization(false);
    uint32_t start_time = time_us_32();
    pizza_model_infer(tensor_buffer, probability_buffer);
    uint32_t end_time = time_us_32();
    
    // Print results
    int class_id = pizza_model_get_prediction(probability_buffer);
    printf("Detected: %s (%.1f%% confidence)\n", 
           CLASS_NAMES[class_id], 
           probability_buffer[class_id] * 100.0f);
    printf("Inference time: %lu µs\n\n", end_time - start_time);
    
    // Run again with CMSIS-NN optimization
    printf("Running CMSIS-NN optimized inference...\n");
    pizza_model_set_hardware_optimization(true);
    start_time = time_us_32();
    pizza_model_infer(tensor_buffer, probability_buffer);
    end_time = time_us_32();
    
    // Print results
    class_id = pizza_model_get_prediction(probability_buffer);
    printf("Detected: %s (%.1f%% confidence)\n", 
           CLASS_NAMES[class_id], 
           probability_buffer[class_id] * 100.0f);
    printf("Inference time: %lu µs\n\n", end_time - start_time);
    
    // Run performance benchmark
    printf("Running comprehensive benchmark...\n");
    pizza_benchmark_results_t results;
    if (pizza_benchmark_run(&results)) {
        printf("Benchmark completed successfully.\n");
        printf("Speedup with CMSIS-NN: %.2fx\n", results.speedup_factor);
    } else {
        printf("Benchmark failed!\n");
    }
    
    // Run continuous detection loop with hardware acceleration
    printf("\nStarting continuous detection with CMSIS-NN...\n");
    printf("Press Ctrl+C to exit\n\n");
    
    pizza_model_set_hardware_optimization(true);
    
    while (true) {
        // Capture image
        if (camera_capture_image(camera_buffer)) {
            // Preprocess
            pizza_model_preprocess(camera_buffer, tensor_buffer);
            
            // Run inference
            start_time = time_us_32();
            pizza_model_infer(tensor_buffer, probability_buffer);
            end_time = time_us_32();
            
            // Get prediction
            class_id = pizza_model_get_prediction(probability_buffer);
            
            // Print result
            printf("Detected: %s (%.1f%%), Inference: %lu µs\n", 
                   CLASS_NAMES[class_id], 
                   probability_buffer[class_id] * 100.0f,
                   end_time - start_time);
        }
        
        // Wait a bit between detections
        sleep_ms(500);
    }
    
    return 0;
}