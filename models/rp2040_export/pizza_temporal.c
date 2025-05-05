/**
 * Pizza Temporal Smoothing Implementation
 * Optimized for resource-constrained RP2040 microcontroller
 */

#include "pizza_temporal.h"
#include <string.h>
#include <math.h>

// Internal state
typedef struct {
    // Configuration
    ts_strategy_t strategy;
    int max_window_size;
    int current_window_size;
    float decay_factor;
    
    // Circular buffer for prediction history
    int prediction_history[TS_MAX_WINDOW_SIZE];
    float confidence_history[TS_MAX_WINDOW_SIZE];
    float probability_history[TS_MAX_WINDOW_SIZE][TS_MAX_CLASSES];
    
    // Buffer state
    int history_count;
    int history_index;
    int num_classes;
    
    // Flag to indicate if the smoother is initialized
    bool initialized;
} ts_state_t;

// Global state (static to avoid exposing implementation details)
static ts_state_t ts_state = {
    .strategy = TS_STRATEGY_MAJORITY_VOTE,
    .max_window_size = TS_MAX_WINDOW_SIZE,
    .current_window_size = 5,  // Default window size
    .decay_factor = 0.7f,      // Default decay factor
    .history_count = 0,
    .history_index = 0,
    .num_classes = TS_MAX_CLASSES,
    .initialized = false
};

/**
 * Initialize the temporal smoother with the specified strategy
 */
void ts_init(ts_strategy_t strategy, float decay_factor) {
    // Validate and set strategy
    if (strategy <= TS_STRATEGY_CONFIDENCE_WEIGHTED) {
        ts_state.strategy = strategy;
    } else {
        ts_state.strategy = TS_STRATEGY_MAJORITY_VOTE; // Default fallback
    }
    
    // Validate and set decay factor
    if (decay_factor >= 0.0f && decay_factor <= 1.0f) {
        ts_state.decay_factor = decay_factor;
    } else {
        ts_state.decay_factor = 0.7f; // Default value
    }
    
    // Reset the buffer
    ts_reset();
    
    // Mark as initialized
    ts_state.initialized = true;
}

/**
 * Add a new prediction to the temporal buffer
 */
void ts_add_prediction(int predicted_class, float confidence, const float *probabilities) {
    // Initialize if not already done
    if (!ts_state.initialized) {
        ts_init(TS_STRATEGY_MAJORITY_VOTE, 0.7f);
    }
    
    // Validate class index
    if (predicted_class < 0 || predicted_class >= ts_state.num_classes) {
        return; // Invalid class
    }
    
    // Add to history buffer
    ts_state.prediction_history[ts_state.history_index] = predicted_class;
    ts_state.confidence_history[ts_state.history_index] = confidence;
    
    // Add probabilities if provided
    if (probabilities != NULL) {
        for (int i = 0; i < ts_state.num_classes; i++) {
            ts_state.probability_history[ts_state.history_index][i] = probabilities[i];
        }
    } else {
        // If no probabilities provided, create a one-hot encoding
        for (int i = 0; i < ts_state.num_classes; i++) {
            ts_state.probability_history[ts_state.history_index][i] = (i == predicted_class) ? confidence : 0.0f;
        }
    }
    
    // Update circular buffer state
    ts_state.history_index = (ts_state.history_index + 1) % ts_state.current_window_size;
    if (ts_state.history_count < ts_state.current_window_size) {
        ts_state.history_count++;
    }
}

/**
 * Apply majority vote smoothing
 */
static int apply_majority_vote(ts_result_t *result) {
    // Count occurrences of each class
    int counts[TS_MAX_CLASSES] = {0};
    int max_count = 0;
    int max_class = 0;
    
    // Count predictions
    for (int i = 0; i < ts_state.history_count; i++) {
        int pred = ts_state.prediction_history[i];
        counts[pred]++;
        
        if (counts[pred] > max_count) {
            max_count = counts[pred];
            max_class = pred;
        }
    }
    
    // If result structure provided, fill in confidence and probabilities
    if (result != NULL) {
        // Calculate average confidence for winning class
        float total_confidence = 0.0f;
        int conf_count = 0;
        
        for (int i = 0; i < ts_state.history_count; i++) {
            if (ts_state.prediction_history[i] == max_class) {
                total_confidence += ts_state.confidence_history[i];
                conf_count++;
            }
        }
        
        // Set confidence
        result->class_index = max_class;
        result->confidence = (conf_count > 0) ? (total_confidence / conf_count) : 0.0f;
        
        // Calculate average probabilities
        for (int c = 0; c < ts_state.num_classes; c++) {
            float sum = 0.0f;
            for (int i = 0; i < ts_state.history_count; i++) {
                sum += ts_state.probability_history[i][c];
            }
            result->probabilities[c] = sum / ts_state.history_count;
        }
    }
    
    return max_class;
}

/**
 * Apply moving average smoothing
 */
static int apply_moving_average(ts_result_t *result) {
    float avg_probabilities[TS_MAX_CLASSES] = {0.0f};
    
    // Calculate average probabilities
    for (int c = 0; c < ts_state.num_classes; c++) {
        for (int i = 0; i < ts_state.history_count; i++) {
            avg_probabilities[c] += ts_state.probability_history[i][c];
        }
        avg_probabilities[c] /= ts_state.history_count;
    }
    
    // Find class with highest average probability
    int max_class = 0;
    float max_prob = avg_probabilities[0];
    
    for (int c = 1; c < ts_state.num_classes; c++) {
        if (avg_probabilities[c] > max_prob) {
            max_prob = avg_probabilities[c];
            max_class = c;
        }
    }
    
    // Fill result structure if provided
    if (result != NULL) {
        result->class_index = max_class;
        result->confidence = max_prob;
        
        // Copy average probabilities
        memcpy(result->probabilities, avg_probabilities, 
               ts_state.num_classes * sizeof(float));
    }
    
    return max_class;
}

/**
 * Apply exponential moving average smoothing
 */
static int apply_exponential_ma(ts_result_t *result) {
    float ema_probabilities[TS_MAX_CLASSES] = {0.0f};
    float weights[TS_MAX_WINDOW_SIZE];
    float weight_sum = 0.0f;
    
    // Calculate exponentially decaying weights
    // Most recent frames get higher weight
    for (int i = 0; i < ts_state.history_count; i++) {
        // Calculate index in history buffer (most recent first)
        int idx = (ts_state.history_index - 1 - i + ts_state.current_window_size) 
                  % ts_state.current_window_size;
        if (idx < 0) idx += ts_state.current_window_size;
        
        // Calculate weight
        float weight = powf(ts_state.decay_factor, i);
        weights[idx] = weight;
        weight_sum += weight;
    }
    
    // Normalize weights to sum to 1.0
    for (int i = 0; i < ts_state.history_count; i++) {
        weights[i] /= weight_sum;
    }
    
    // Calculate weighted probabilities
    for (int c = 0; c < ts_state.num_classes; c++) {
        for (int i = 0; i < ts_state.history_count; i++) {
            ema_probabilities[c] += weights[i] * ts_state.probability_history[i][c];
        }
    }
    
    // Find class with highest weighted probability
    int max_class = 0;
    float max_prob = ema_probabilities[0];
    
    for (int c = 1; c < ts_state.num_classes; c++) {
        if (ema_probabilities[c] > max_prob) {
            max_prob = ema_probabilities[c];
            max_class = c;
        }
    }
    
    // Fill result structure if provided
    if (result != NULL) {
        result->class_index = max_class;
        result->confidence = max_prob;
        
        // Copy weighted probabilities
        memcpy(result->probabilities, ema_probabilities, 
               ts_state.num_classes * sizeof(float));
    }
    
    return max_class;
}

/**
 * Apply confidence-weighted smoothing
 */
static int apply_confidence_weighted(ts_result_t *result) {
    float weighted_probabilities[TS_MAX_CLASSES] = {0.0f};
    float total_confidence = 0.0f;
    
    // Calculate sum of confidences
    for (int i = 0; i < ts_state.history_count; i++) {
        total_confidence += ts_state.confidence_history[i];
    }
    
    // If all confidences are zero, fallback to simple averaging
    if (total_confidence <= 0.001f) {
        return apply_moving_average(result);
    }
    
    // Calculate confidence-weighted probabilities
    for (int c = 0; c < ts_state.num_classes; c++) {
        for (int i = 0; i < ts_state.history_count; i++) {
            float weight = ts_state.confidence_history[i] / total_confidence;
            weighted_probabilities[c] += weight * ts_state.probability_history[i][c];
        }
    }
    
    // Find class with highest weighted probability
    int max_class = 0;
    float max_prob = weighted_probabilities[0];
    
    for (int c = 1; c < ts_state.num_classes; c++) {
        if (weighted_probabilities[c] > max_prob) {
            max_prob = weighted_probabilities[c];
            max_class = c;
        }
    }
    
    // Fill result structure if provided
    if (result != NULL) {
        result->class_index = max_class;
        result->confidence = max_prob;
        
        // Copy weighted probabilities
        memcpy(result->probabilities, weighted_probabilities, 
               ts_state.num_classes * sizeof(float));
    }
    
    return max_class;
}

/**
 * Get the smoothed prediction based on the current buffer
 */
int ts_get_smoothed_prediction(ts_result_t *result) {
    // Initialize if not already done
    if (!ts_state.initialized) {
        ts_init(TS_STRATEGY_MAJORITY_VOTE, 0.7f);
    }
    
    // If no history or only one frame, return the most recent prediction
    if (ts_state.history_count <= 1) {
        if (ts_state.history_count == 0) {
            // No predictions yet
            if (result != NULL) {
                result->class_index = 0;
                result->confidence = 0.0f;
                memset(result->probabilities, 0, ts_state.num_classes * sizeof(float));
            }
            return 1; // Error: no predictions yet
        }
        
        // Just one prediction
        int last_idx = (ts_state.history_index - 1 + ts_state.current_window_size) 
                        % ts_state.current_window_size;
        
        if (result != NULL) {
            result->class_index = ts_state.prediction_history[last_idx];
            result->confidence = ts_state.confidence_history[last_idx];
            memcpy(result->probabilities, ts_state.probability_history[last_idx], 
                   ts_state.num_classes * sizeof(float));
        }
        
        return 0; // Success
    }
    
    // Apply the selected smoothing strategy
    int smoothed_class;
    
    switch (ts_state.strategy) {
        case TS_STRATEGY_MOVING_AVERAGE:
            smoothed_class = apply_moving_average(result);
            break;
        case TS_STRATEGY_EXPONENTIAL_MA:
            smoothed_class = apply_exponential_ma(result);
            break;
        case TS_STRATEGY_CONFIDENCE_WEIGHTED:
            smoothed_class = apply_confidence_weighted(result);
            break;
        case TS_STRATEGY_MAJORITY_VOTE:
        default:
            smoothed_class = apply_majority_vote(result);
            break;
    }
    
    return 0; // Success
}

/**
 * Reset the temporal smoother
 */
void ts_reset(void) {
    ts_state.history_count = 0;
    ts_state.history_index = 0;
}

/**
 * Get the current window size
 */
int ts_get_window_size(void) {
    return ts_state.current_window_size;
}

/**
 * Set the desired window size for smoothing
 */
int ts_set_window_size(int window_size) {
    // Validate window size
    if (window_size < 1 || window_size > ts_state.max_window_size) {
        return 1; // Error: invalid window size
    }
    
    // If reducing window size, we might need to adjust history
    if (window_size < ts_state.current_window_size && ts_state.history_count > 0) {
        // Create temporary buffer to hold most recent entries
        int temp_count = (window_size < ts_state.history_count) ? 
                           window_size : ts_state.history_count;
        
        int temp_predictions[TS_MAX_WINDOW_SIZE];
        float temp_confidences[TS_MAX_WINDOW_SIZE];
        float temp_probabilities[TS_MAX_WINDOW_SIZE][TS_MAX_CLASSES];
        
        // Copy most recent entries
        for (int i = 0; i < temp_count; i++) {
            int src_idx = (ts_state.history_index - temp_count + i + ts_state.current_window_size) 
                           % ts_state.current_window_size;
            
            temp_predictions[i] = ts_state.prediction_history[src_idx];
            temp_confidences[i] = ts_state.confidence_history[src_idx];
            memcpy(temp_probabilities[i], ts_state.probability_history[src_idx], 
                   ts_state.num_classes * sizeof(float));
        }
        
        // Update window size
        ts_state.current_window_size = window_size;
        
        // Reset buffer state
        ts_state.history_count = temp_count;
        ts_state.history_index = 0;
        
        // Copy back the entries
        for (int i = 0; i < temp_count; i++) {
            ts_state.prediction_history[i] = temp_predictions[i];
            ts_state.confidence_history[i] = temp_confidences[i];
            memcpy(ts_state.probability_history[i], temp_probabilities[i], 
                   ts_state.num_classes * sizeof(float));
            
            ts_state.history_index = (ts_state.history_index + 1) % window_size;
        }
    } else {
        // Just update the window size
        ts_state.current_window_size = window_size;
        
        // Adjust history count if it exceeds new window size
        if (ts_state.history_count > window_size) {
            ts_state.history_count = window_size;
        }
    }
    
    return 0; // Success
}

/**
 * Change the smoothing strategy
 */
int ts_set_strategy(ts_strategy_t strategy, float decay_factor) {
    // Validate strategy
    if (strategy > TS_STRATEGY_CONFIDENCE_WEIGHTED) {
        return 1; // Error: invalid strategy
    }
    
    // Validate decay factor for EMA
    if (strategy == TS_STRATEGY_EXPONENTIAL_MA) {
        if (decay_factor < 0.0f || decay_factor > 1.0f) {
            return 2; // Error: invalid decay factor
        }
        ts_state.decay_factor = decay_factor;
    }
    
    ts_state.strategy = strategy;
    return 0; // Success
}