/**
 * @file optimal_transport.h
 * @brief Sinkhorn algorithm for domain adaptation (Q15.16 fixed-point)
 */
#ifndef OPTIMAL_TRANSPORT_H
#define OPTIMAL_TRANSPORT_H

#include <stdint.h>

#define OT_DIM 8 // Size of feature vectors for calibration

/**
 * @brief Pre-computes the exponential kernel K = exp(-C/epsilon)
 * Note: In a real scenario, this is done offline, but here we simulate the structure.
 */
void sinkhorn_init(void);

/**
 * @brief Computes the optimal transport plan iteratively using Q15.16 arithmetic.
 * Helps calibrate camera ISP parameters against domain shifts.
 * 
 * @param mu Source distribution (Q15.16)
 * @param nu Target distribution (Q15.16)
 * @param iterations Number of Sinkhorn iterations
 */
void sinkhorn_compute(const int32_t* mu, const int32_t* nu, int iterations);

#endif // OPTIMAL_TRANSPORT_H
