import numpy as np
import json

class DirichletCalibrator:
    """
    Implements Bayesian Evidential Deep Learning for on-device uncertainty.
    Converts logits to Dirichlet concentration parameters: alpha = softplus(z) + 1.
    """
    def __init__(self, num_classes: int = 2):
        self.K = num_classes  # e.g., Pizza vs No-Pizza

    def calculate_alpha(self, logits: np.ndarray) -> np.ndarray:
        """
        alpha_i = softplus(logits_i) + 1
        Higher alpha means more evidence for that class.
        """
        # softplus(x) = log(1 + exp(x))
        # Approximation for fixed-point: if x > 20, softplus(x) approx x
        alpha = np.log1p(np.exp(np.clip(logits, -20, 20))) + 1
        return alpha

    def get_uncertainty(self, alpha: np.ndarray) -> float:
        """
        Total Uncertainty u = K / S, where S = sum(alpha)
        Range: (0, 1]. u=1 means max uncertainty (OOD).
        """
        S = np.sum(alpha)
        return self.K / S

    def get_belief_mass(self, alpha: np.ndarray) -> np.ndarray:
        """
        Belief mass m_i = (alpha_i - 1) / S
        Represents the portion of evidence assigned to class i.
        """
        S = np.sum(alpha)
        return (alpha - 1) / S

if __name__ == "__main__":
    # Test with OOD (Out-of-Distribution) sample: low logits
    ood_logits = np.array([-5.0, -5.0])
    calibrator = DirichletCalibrator()
    alpha_ood = calibrator.calculate_alpha(ood_logits)
    print(f"OOD Alpha: {alpha_ood}, Uncertainty: {calibrator.get_uncertainty(alpha_ood)}")

    # Test with clear Pizza detection
    pizza_logits = np.array([10.0, -10.0])
    alpha_pizza = calibrator.calculate_alpha(pizza_logits)
    print(f"Pizza Alpha: {alpha_pizza}, Uncertainty: {calibrator.get_uncertainty(alpha_pizza)}")
