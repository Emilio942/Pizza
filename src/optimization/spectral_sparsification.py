import torch
import numpy as np

class SpectralSparsifier:
    """
    Implements Spectral Sparsification using Effective Resistance.
    (Spielman-Srivastava Algorithm)
    """
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def compute_effective_resistance(self, weight_matrix):
        """
        Approximates effective resistance for each 'edge' (weight).
        R_uv = (u - v)^T L^+ (u - v)
        """
        W = np.abs(weight_matrix)
        # Degree matrix
        D = np.diag(np.sum(W, axis=1))
        # Laplacian
        L = D - W
        
        # Pseudo-inverse of Laplacian
        L_pinv = np.linalg.pinv(L)
        
        rows, cols = W.shape
        resistances = np.zeros_like(W)
        
        for i in range(rows):
            for j in range(cols):
                if W[i, j] > 0:
                    # e_i - e_j
                    e_ij = np.zeros(rows)
                    e_ij[i] = 1
                    if j < rows: # For non-square, this is an approximation
                        e_ij[j] = -1
                    
                    R_eff = e_ij.T @ L_pinv @ e_ij
                    resistances[i, j] = R_eff
                    
        return resistances

    def sparsify(self, weight_tensor):
        """
        Takes a PyTorch weight tensor and zeroes out weights with low effective resistance.
        """
        orig_shape = weight_tensor.shape
        if len(orig_shape) < 2:
            return weight_tensor # Cannot sparsify 1D
            
        # Flatten to 2D for graph interpretation
        W_2d = weight_tensor.view(orig_shape[0], -1).detach().cpu().numpy()
        
        # Calculate resistances
        R = self.compute_effective_resistance(W_2d)
        
        # Sampling probabilities p_e proportional to w_e * R_e
        probabilities = W_2d * R
        prob_sum = np.sum(probabilities)
        
        if prob_sum > 0:
            probabilities /= prob_sum
        else:
            return weight_tensor

        # Determine number of edges to keep: O(n * log(n) / eps^2)
        n = W_2d.shape[0]
        k = int((n * np.log(n)) / (self.epsilon ** 2))
        k = min(k, W_2d.size) # Cap at total size
        
        # Sample edges
        flat_probs = probabilities.flatten()
        sampled_indices = np.random.choice(W_2d.size, size=k, p=flat_probs, replace=True)
        
        # Create sparse mask
        mask = np.zeros(W_2d.size)
        mask[sampled_indices] = 1
        mask = mask.reshape(W_2d.shape)
        
        # Apply mask and reshape
        sparsified_W = torch.tensor(W_2d * mask).view(orig_shape)
        
        print(f"Spectral Sparsification: Reduced edges to {k} (Sparsity: {100 * (1 - k/W_2d.size):.1f}%)")
        return sparsified_W
