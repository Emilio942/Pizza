/**
 * @file optimal_transport.c
 * @brief Fixed-point Sinkhorn implementation
 */
#include "optimal_transport.h"

// Q15.16 Helpers
#define Q_MUL(a, b) (int32_t)(((int64_t)(a) * (b)) >> 16)
#define Q_DIV(a, b) (int32_t)((((int64_t)(a)) << 16) / (b))
#define Q_ONE (1 << 16)

// Precomputed kernel K and its inverse mapping (Dummy values for structure)
static int32_t K_matrix[OT_DIM][OT_DIM];
static int32_t u_vec[OT_DIM];
static int32_t v_vec[OT_DIM];

void sinkhorn_init(void) {
    for (int i = 0; i < OT_DIM; i++) {
        u_vec[i] = Q_ONE / OT_DIM;
        v_vec[i] = Q_ONE / OT_DIM;
        for (int j = 0; j < OT_DIM; j++) {
            K_matrix[i][j] = (i == j) ? Q_ONE : (Q_ONE >> 2); // Simple diagonal-heavy kernel
        }
    }
}

void sinkhorn_compute(const int32_t* mu, const int32_t* nu, int iterations) {
    for (int iter = 0; iter < iterations; iter++) {
        // Update u = mu ./ (K * v)
        for (int i = 0; i < OT_DIM; i++) {
            int32_t kv = 0;
            for (int j = 0; j < OT_DIM; j++) {
                kv += Q_MUL(K_matrix[i][j], v_vec[j]);
            }
            if (kv > 0) {
                u_vec[i] = Q_DIV(mu[i], kv);
            }
        }
        
        // Update v = nu ./ (K^T * u)
        for (int j = 0; j < OT_DIM; j++) {
            int32_t ktu = 0;
            for (int i = 0; i < OT_DIM; i++) {
                ktu += Q_MUL(K_matrix[i][j], u_vec[i]);
            }
            if (ktu > 0) {
                v_vec[j] = Q_DIV(nu[j], ktu);
            }
        }
    }
}
