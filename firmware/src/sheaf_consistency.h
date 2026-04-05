/**
 * @file sheaf_consistency.h
 * @brief Sheaf Theory and Topos Logic for Sensor Fusion
 */
#ifndef SHEAF_CONSISTENCY_H
#define SHEAF_CONSISTENCY_H

#include <stdint.h>
#include <stdbool.h>

#define NUM_PATCHES 16 // Example: 4x4 grid of patches

// Heyting Algebra Truth Values (Topos Logic)
typedef enum { 
    BOOL_BOT = 0x00, // False (No evidence)
    BOOL_U   = 0x01, // Unknown (Partial/Conflicting evidence)
    BOOL_TOP = 0x02  // True (Full evidence)
} bool_h;

typedef struct {
    uint8_t edge_mask;      // From PIO
    uint8_t topping_class;  // From CNN (int4)
    bool_h  edge_truth;
    bool_h  topping_truth;
} PatchData;

/**
 * @brief Initializes the sheaf structure
 */
void sheaf_init(void);

/**
 * @brief Updates a local patch (local section of the sheaf)
 */
void sheaf_update_patch(uint8_t patch_idx, uint8_t edge, uint8_t topping);

/**
 * @brief Evaluates the Cech Cohomology (H^1) to find ghost pizzas.
 * @return true if consistent (H^1 = 0), false if ghost pizza detected.
 */
bool sheaf_check_global_consistency(void);

/**
 * @brief Computes the global Heyting truth value for the entire image.
 */
bool_h sheaf_evaluate_global_truth(void);

#endif // SHEAF_CONSISTENCY_H
