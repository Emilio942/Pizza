"""
Pizza recognition verification components.

This module provides quality assessment and verification for pizza recognition
results, including neural network-based verifiers and data handling utilities.
"""

from .pizza_verifier import (
    PizzaVerifier,
    PizzaVerifierNet,
    VerifierData,
    load_verifier_data,
    save_verifier_data
)

__all__ = [
    'PizzaVerifier',
    'PizzaVerifierNet', 
    'VerifierData',
    'load_verifier_data',
    'save_verifier_data'
]
