/**
 * Pizza Temporal Smoothing Library for RP2040
 * 
 * Implements various strategies for temporal smoothing of inference results
 * to improve recognition accuracy and stability:
 * 
 * - MAJORITY_VOTE: Selects the most frequent class across window frames
 * - MOVING_AVERAGE: Averages probabilities over window frames
 * - EXPONENTIAL_MA: Applies weighted averaging with more weight on recent frames
 * - CONFIDENCE_WEIGHTED: Weights predictions by their confidence
 * 
 * Designed for minimal memory usage and efficient operation on RP2040.
 */

#ifndef PIZZA_TEMPORAL_H
#define PIZZA_TEMPORAL_H

#include <stdint.h>
#include <stdbool.h>

// Configuration
#define TS_MAX_WINDOW_SIZE 10   // Maximum temporal window size
#define TS_MAX_CLASSES 10       // Maximum number of supported classes

// Smoothing strategies
typedef enum {
    TS_STRATEGY_MAJORITY_VOTE = 0,
    TS_STRATEGY_MOVING_AVERAGE = 1,
    TS_STRATEGY_EXPONENTIAL_MA = 2,
    TS_STRATEGY_CONFIDENCE_WEIGHTED = 3
} ts_strategy_t;

// Result structure
typedef struct {
    int class_index;               // Index of the predicted class
    float confidence;              // Confidence value (0.0-1.0)
    float probabilities[TS_MAX_CLASSES]; // Class probabilities
} ts_result_t;

/**
 * Initialize the temporal smoother with the specified strategy
 * 
 * @param strategy The smoothing strategy to use
 * @param decay_factor Weight factor for exponential smoothing (0.0-1.0)
 */
void ts_init(ts_strategy_t strategy, float decay_factor);

/**
 * Add a new prediction to the temporal buffer
 * 
 * @param predicted_class Index of the predicted class
 * @param confidence Confidence of the prediction (0.0-1.0)
 * @param probabilities Array of class probabilities (can be NULL)
 */
void ts_add_prediction(int predicted_class, float confidence, const float *probabilities);

/**
 * Get the smoothed prediction based on the current buffer
 * 
 * @param result Pointer to store the result (can be NULL if only interested in class)
 * @return 0 if successful, non-zero on error
 */
int ts_get_smoothed_prediction(ts_result_t *result);

/**
 * Reset the temporal smoother
 * Call when the scene changes significantly or when
 * starting a new detection sequence
 */
void ts_reset(void);

/**
 * Get the current window size (number of frames in buffer)
 * 
 * @return Current number of frames in the buffer
 */
int ts_get_window_size(void);

/**
 * Set the desired window size for smoothing
 * 
 * @param window_size Number of frames to use (1-TS_MAX_WINDOW_SIZE)
 * @return 0 if successful, non-zero on error
 */
int ts_set_window_size(int window_size);

/**
 * Change the smoothing strategy
 * 
 * @param strategy New strategy to use
 * @param decay_factor New decay factor for exponential smoothing
 * @return 0 if successful, non-zero on error
 */
int ts_set_strategy(ts_strategy_t strategy, float decay_factor);

#endif // PIZZA_TEMPORAL_H