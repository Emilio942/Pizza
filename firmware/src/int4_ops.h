/**
 * @file int4_ops.h
 * @brief High-performance INT4 unpacking macros for RP2040 (Cortex-M0+)
 * 
 * Optimized for weights packed as 2x4-bit per byte.
 */

#ifndef INT4_OPS_H
#define INT4_OPS_H

#include <stdint.h>

/**
 * @brief Extract first weight (low 4 bits) from a packed byte
 * Returns signed value in range [-8, 7]
 */
#define UNPACK_INT4_W1(packed_byte) \
    ((int8_t)((packed_byte & 0x0F) - 8))

/**
 * @brief Extract second weight (high 4 bits) from a packed byte
 * Returns signed value in range [-8, 7]
 */
#define UNPACK_INT4_W2(packed_byte) \
    ((int8_t)(((packed_byte >> 4) & 0x0F) - 8))

/**
 * @brief Fast MAC (Multiply-Accumulate) for a pair of INT4 weights
 * Optimized for Cortex-M0+ 32-bit registers.
 */
static inline void mac_int4_pair(int32_t *accumulator, uint8_t packed_weight, int8_t act1, int8_t act2) {
    // Unpack on the fly
    int8_t w1 = UNPACK_INT4_W1(packed_weight);
    int8_t w2 = UNPACK_INT4_W2(packed_weight);
    
    // Accumulate
    *accumulator += (int32_t)w1 * act1;
    *accumulator += (int32_t)w2 * act2;
}

#endif // INT4_OPS_H
