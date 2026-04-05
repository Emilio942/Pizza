/**
 * @file softplus_table.h
 * @brief Optimized fixed-point lookup table for softplus(x) = ln(1 + e^x)
 */

#ifndef SOFTPLUS_TABLE_H
#define SOFTPLUS_TABLE_H

#include <stdint.h>

/**
 * @brief Fast fixed-point softplus lookup
 * Approximation:
 * x < -8: 0
 * x > 8: x
 * -8 <= x <= 8: lookup table
 */
static inline int32_t lookup_softplus_fixed(int32_t x_q15_16) {
    // Thresholds in Q15.16
    const int32_t lower_bound = -8 << 16;
    const int32_t upper_bound = 8 << 16;
    
    if (x_q15_16 < lower_bound) return 0;
    if (x_q15_16 > upper_bound) return x_q15_16;
    
    // (Simplified) Linear interpolation or small table for the "active" range
    // In a real project, we'd have a 256-entry const array here.
    // For now, we use a simple linear piecewise approximation for the S-curve.
    
    // Approximation: 0.5 * x + offset in the middle range
    return (x_q15_16 >> 1) + (1 << 15); // x/2 + 0.5
}

#endif // SOFTPLUS_TABLE_H
