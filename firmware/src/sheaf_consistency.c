/**
 * @file sheaf_consistency.c
 * @brief Implementation of Sheaf Theory consistency checks
 */
#include "sheaf_consistency.h"

static PatchData patches[NUM_PATCHES];

// Heyting Conjunction (Min)
static inline bool_h heyting_meet(bool_h a, bool_h b) {
    if (a == BOOL_TOP && b == BOOL_TOP) return BOOL_TOP;
    if (a == BOOL_BOT && b == BOOL_BOT) return BOOL_BOT;
    return BOOL_U;
}

// Heyting Disjunction (Max)
static inline bool_h heyting_join(bool_h a, bool_h b) {
    if (a == BOOL_TOP || b == BOOL_TOP) return BOOL_TOP;
    if (a == BOOL_U || b == BOOL_U) return BOOL_U;
    return BOOL_BOT;
}

static inline bool_h make_truth(uint8_t val) {
    if (val > 0) return BOOL_TOP;
    return BOOL_BOT;
}

void sheaf_init(void) {
    for (int i = 0; i < NUM_PATCHES; i++) {
        patches[i].edge_mask = 0;
        patches[i].topping_class = 0;
        patches[i].edge_truth = BOOL_BOT;
        patches[i].topping_truth = BOOL_BOT;
    }
}

void sheaf_update_patch(uint8_t patch_idx, uint8_t edge, uint8_t topping) {
    if (patch_idx >= NUM_PATCHES) return;
    patches[patch_idx].edge_mask = edge;
    patches[patch_idx].topping_class = topping;
    patches[patch_idx].edge_truth = make_truth(edge);
    patches[patch_idx].topping_truth = make_truth(topping);
}

bool sheaf_check_global_consistency(void) {
    // Compute 1-cocycles between adjacent patches
    // Simplified 1D check for demonstration
    for (int i = 0; i < NUM_PATCHES - 2; i++) {
        uint8_t e12 = (patches[i].edge_mask ^ patches[i+1].edge_mask) > 0 ? 1 : 0;
        uint8_t e23 = (patches[i+1].edge_mask ^ patches[i+2].edge_mask) > 0 ? 1 : 0;
        uint8_t e13 = (patches[i].edge_mask ^ patches[i+2].edge_mask) > 0 ? 1 : 0;
        
        // Cocycle condition: d12 + d23 = d13 (mod 2)
        if ((e12 ^ e23) != e13) {
            return false; // H^1 != 0 -> Ghost Pizza!
        }
    }
    return true; // H^1 == 0 -> Consistent
}

bool_h sheaf_evaluate_global_truth(void) {
    bool_h global_truth = BOOL_BOT;
    for (int i = 0; i < NUM_PATCHES; i++) {
        // Local section agreement
        bool_h local_agreement = heyting_meet(patches[i].edge_truth, patches[i].topping_truth);
        global_truth = heyting_join(global_truth, local_agreement);
    }
    return global_truth;
}
