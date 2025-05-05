/**
 * Pizza Detection Example Implementation with
 * Enhanced Preprocessing and Temporal Smoothing
 * 
 * This file demonstrates how to integrate the image preprocessing
 * and temporal smoothing into a complete pizza detection workflow.
 */

#include "pizza_model.h"
#include "pizza_preprocess.h"
#include "pizza_temporal.h"
#include <stdio.h>
#include <stdbool.h>

// Camera buffer for raw image capture
static uint8_t camera_buffer[CAMERA_WIDTH * CAMERA_HEIGHT * 3]; // RGB888 format

// Buffer for input tensor
static float model_input_tensor[MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * MODEL_INPUT_CHANNELS];

// Buffer for model output
static float model_output_probs[MODEL_NUM_CLASSES];

/**
 * Main loop example for pizza detection with preprocessing
 * and temporal smoothing
 */
void pizza_detection_example(void) {
    // Initialize pizza model (only needs to be done once)
    pizza_model_init();
    
    // Configure temporal smoother strategy if default is not desired
    // ts_init(TS_STRATEGY_MOVING_AVERAGE, 0.7f);
    
    while (true) {
        // 1. Capture image from camera (implementation depends on camera driver)
        bool capture_success = capture_camera_image(camera_buffer);
        if (!capture_success) {
            printf("Error capturing image\n");
            continue;
        }
        
        // 2. Preprocess image (includes CLAHE lighting enhancement)
        pizza_model_preprocess(camera_buffer, model_input_tensor);
        
        // 3. Run model inference (includes temporal smoothing)
        pizza_model_infer(model_input_tensor, model_output_probs);
        
        // 4. Get final prediction (class index with highest probability)
        int class_idx = pizza_model_get_prediction(model_output_probs);
        float confidence = model_output_probs[class_idx];
        
        // 5. Use the prediction for your application
        printf("Detected: %s (%.1f%%)\n", 
               CLASS_NAMES[class_idx], 
               confidence * 100.0f);
               
        // Optional: Check if a significant change was detected
        if (confidence > 0.9f && !strcmp(CLASS_NAMES[class_idx], "burnt")) {
            printf("Warning: Pizza is burning!\n");
            // Trigger an alarm, notification, etc.
        }
        
        // Sleep to control inference rate and save power
        sleep_ms(200); // Run at ~5 FPS
    }
}

/**
 * Example for scenarios where you need to reset the temporal smoother
 * For example, when a new pizza is placed in the oven
 */
void handle_new_detection_scenario(void) {
    // Reset temporal smoother to start fresh
    pizza_model_reset_temporal();
    
    printf("Temporal smoother reset - starting new detection sequence\n");
    
    // Continue with normal detection
    pizza_detection_example();
}

/**
 * Example for advanced use-case: manually combining smoothing strategies
 */
void advanced_detection_example(void) {
    // Initialize with majority vote for stable detection
    ts_init(TS_STRATEGY_MAJORITY_VOTE, 0.7f);
    
    // First detection phase with majority voting (5 frames)
    for (int i = 0; i < 5; i++) {
        // Capture and process image
        capture_camera_image(camera_buffer);
        pizza_model_preprocess(camera_buffer, model_input_tensor);
        
        // Run inference (without applying temporal smoothing yet)
        // This is a low-level approach that bypasses the built-in smoothing in pizza_model_infer
        
        // Get raw predictions
        float raw_probs[MODEL_NUM_CLASSES];
        // ... model inference code ...
        
        // Find max class
        int raw_class = 0;
        float max_prob = raw_probs[0];
        for (int j = 1; j < MODEL_NUM_CLASSES; j++) {
            if (raw_probs[j] > max_prob) {
                max_prob = raw_probs[j];
                raw_class = j;
            }
        }
        
        // Add to temporal smoother
        ts_add_prediction(raw_class, max_prob, raw_probs);
        
        // Sleep between frames
        sleep_ms(100);
    }
    
    // Get initial classification with majority vote
    ts_result_t initial_result;
    ts_get_smoothed_prediction(&initial_result);
    
    // If we detect a pizza, switch to EMA for tracking progression
    if (initial_result.predicted_class != 0) { // Assuming 0 = no pizza
        printf("Pizza detected, switching to EMA for tracking progression\n");
        
        // Switch to exponential moving average for tracking progression
        ts_init(TS_STRATEGY_EXP_MOVING_AVERAGE, 0.8f);
        
        // Continue monitoring with EMA
        while (true) {
            // Normal processing flow
            capture_camera_image(camera_buffer);
            pizza_model_preprocess(camera_buffer, model_input_tensor);
            pizza_model_infer(model_input_tensor, model_output_probs);
            
            int class_idx = pizza_model_get_prediction(model_output_probs);
            printf("Tracking: %s\n", CLASS_NAMES[class_idx]);
            
            // Check for completion condition
            if (class_idx == 2) { // Assuming 2 = "done"
                printf("Pizza is done!\n");
                break;
            }
            
            sleep_ms(200);
        }
    }
}