/**
 * @file evidential_logic.c
 * @brief Dirichlet Evidence Logic for RP2040 (Bayesian Deep Learning)
 * 
 * Implements u = K / S with fixed-point Q15.16 arithmetic.
 */

#include <stdint.h>
#include <stdbool.h>
#include "softplus_table.h"

// Number of classes (Pizza vs No-Pizza)
#define K_CLASSES 2

// Q15.16 conversion macros
#define INT_TO_Q15_16(x) ((int32_t)(x) << 16)
#define Q15_16_TO_INT(x) ((int32_t)(x) >> 16)

typedef struct {
    int32_t alpha[K_CLASSES]; // Q15.16 concentration parameters
    int32_t strength_s;       // S = sum(alpha_i)
    int32_t uncertainty;      // u = K / S
} EvidenceState;

/**
 * @brief Calculate Dirichlet uncertainty based on logits
 */
void calculate_evidence(int32_t *logits, EvidenceState *state) {
    state->strength_s = 0;
    
    for (int i = 0; i < K_CLASSES; i++) {
        // 1. Get alpha_i = softplus(logit_i) + 1
        // We use a precomputed lookup table for softplus(x) = ln(1+e^x)
        int32_t softplus_val = lookup_softplus_fixed(logits[i]);
        state->alpha[i] = softplus_val + INT_TO_Q15_16(1);
        
        // 2. Accumulate strength S
        state->strength_s += state->alpha[i];
    }
    
    // 3. Compute total uncertainty u = K / S
    // For binary class K=2, u = 2.0 / S. Using fixed-point division:
    // u = (K << 16) / (S >> 16) ? No, we need more precision.
    // Fixed-point division u = (K << 16) * 65536 / S
    if (state->strength_s > 0) {
        int64_t k_fixed = (int64_t)K_CLASSES << 32;
        state->uncertainty = (int32_t)(k_fixed / state->strength_s);
    } else {
        state->uncertainty = INT_TO_Q15_16(1); // Max uncertainty
    }
}

/**
 * @brief Decision function considering uncertainty
 * Returns class index or -1 if too uncertain (OOD)
 */
int8_t evidential_predict(EvidenceState *state, int32_t threshold_q15_16) {
    if (state->uncertainty > threshold_q15_16) {
        return -1; // Out-of-Distribution (OOD)
    }
    
    int8_t best_class = 0;
    int32_t max_alpha = state->alpha[0];
    
    for (int i = 1; i < K_CLASSES; i++) {
        if (state->alpha[i] > max_alpha) {
            max_alpha = state->alpha[i];
            best_class = i;
        }
    }
    return best_class;
}
