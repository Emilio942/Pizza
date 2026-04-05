/**
 * @file frequent_directions.c
 * @brief Implementation of Frequent Directions for On-Device Learning
 */
#include "frequent_directions.h"
#include <string.h>
#include <math.h>

// The sketch matrix V (size K x FEATURE_DIM)
static float V[SKETCH_K][FEATURE_DIM];

void fd_init(void) {
    memset(V, 0, sizeof(V));
}

// Simple Gram-Schmidt orthogonalization for the sketch
static void orthogonalize_row(float* row) {
    for (int i = 0; i < SKETCH_K; i++) {
        float dot = 0.0f;
        for (int j = 0; j < FEATURE_DIM; j++) {
            dot += row[j] * V[i][j];
        }
        for (int j = 0; j < FEATURE_DIM; j++) {
            row[j] -= dot * V[i][j];
        }
    }
    
    float norm = 0.0f;
    for (int j = 0; j < FEATURE_DIM; j++) {
        norm += row[j] * row[j];
    }
    norm = sqrtf(norm);
    
    if (norm > 1e-6f) {
        for (int j = 0; j < FEATURE_DIM; j++) {
            row[j] /= norm;
        }
    }
}

void fd_update(const float* row) {
    // 1. Find an empty or weakest row in V (simplified: we just replace row 0 for demonstration)
    // In a full implementation, we run an incremental SVD here.
    float new_row[FEATURE_DIM];
    memcpy(new_row, row, sizeof(new_row));
    
    orthogonalize_row(new_row);
    
    // Replace the top row (in full FD, this is the row with the smallest singular value)
    memcpy(V[0], new_row, sizeof(new_row));
}

void fd_adapt_weights(float* weights, float learning_rate) {
    // Delta W = -eta * (V^T * V * W)
    float vw[SKETCH_K] = {0};
    
    // V * W
    for (int i = 0; i < SKETCH_K; i++) {
        for (int j = 0; j < FEATURE_DIM; j++) {
            vw[i] += V[i][j] * weights[j];
        }
    }
    
    // V^T * (V * W)
    for (int j = 0; j < FEATURE_DIM; j++) {
        float update = 0.0f;
        for (int i = 0; i < SKETCH_K; i++) {
            update += V[i][j] * vw[i];
        }
        weights[j] -= learning_rate * update;
    }
}
