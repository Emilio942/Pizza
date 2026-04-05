/**
 * @file frequent_directions.h
 * @brief O(1) SRAM Jacobian Sketching (Frequent Directions)
 */
#ifndef FREQUENT_DIRECTIONS_H
#define FREQUENT_DIRECTIONS_H

#include <stdint.h>

#define FEATURE_DIM 128 // Dimension of the penultimate layer
#define SKETCH_K 16     // Size of the sketch (k << FEATURE_DIM)

/**
 * @brief Initialize the sketch matrix.
 */
void fd_init(void);

/**
 * @brief Updates the Jacobian sketch with a new gradient row.
 * Runs in O(k * FEATURE_DIM) time and requires only O(k * FEATURE_DIM) memory.
 * 
 * @param row New gradient vector (e.g., from a misclassified sample)
 */
void fd_update(const float* row);

/**
 * @brief Computes the adapted weights using the sketched covariance.
 */
void fd_adapt_weights(float* weights, float learning_rate);

#endif // FREQUENT_DIRECTIONS_H
