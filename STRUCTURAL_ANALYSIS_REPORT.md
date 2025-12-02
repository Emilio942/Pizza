# Structural Analysis Report

## 1. Critical Logic & Type Errors (mypy)
**Total Errors:** 445
The static analysis revealed severe structural issues that will likely cause runtime crashes:

*   **Broken Function Calls:**
    *   `src/augmentation/diffusion_pizza_generator.py` calls `validate_image_quality` from `src.utils.validation`, but this function **does not exist**.
    *   `src/integration/diffusion_training_integration.py` calls `train_microcontroller_model` with arguments (`num_epochs`, `learning_rate`, etc.) that were removed or changed during refactoring.

*   **Potential Crashes (NoneType):**
    *   `src/rl/pizza_rl_environment.py`: Multiple accesses to attributes of `current_task` without checking if it is `None`. This will crash the RL training loop if a task isn't assigned.

*   **Type Confusion:**
    *   `src/exceptions.py`: Exception classes are being assigned incompatible types, suggesting a circular dependency or mix-up between `src.emulation` and `src.utils` exceptions.
    *   `src/visualization/status_display.py`: Confusing `OLEDDisplay` with `RP2040RGBLED`.

## 2. Architectural Flaws (pydeps)
*   **"God Object" Dependency:**
    *   `src/pizza_detector.py` (the main entry point) is being imported by:
        *   `src.rl.environment`
        *   `src.verification.pizza_verifier`
        *   `src.rl.pizza_rl_environment`
    *   **Impact:** This is a circular dependency risk. The main script should be a "leaf" node. Shared logic must be moved to `src/utils` or `src/models`.

## 3. Complexity Hotspots (radon)
Code sections with "D" (High Complexity) rating, indicating high risk of bugs:
*   `src/pizza_detector.py`: `detailed_evaluation` (Score: D).
*   `src/augmentation/augmentation.py`: `apply_combination_augmentation`, `EnhancedOvenEffect.forward`.
*   `src/augmentation/advanced_pizza_diffusion_control.py`: `generate_balanced_dataset`.

## 4. Dead Code & Zombies (vulture)
*   **Legacy/Duplicate Files:**
    *   `src/pizza-baking-detection-final.py`: Appears to be a massive duplicate of an older version of `pizza_detector.py`.
    *   `src/legacy/augment_legacy.py`: Unused.
*   **Unused Imports in Main:**
    *   `src/pizza_detector.py` still imports `datasets`, `ctypes`, `struct`, `BalancedPizzaDataset` which are no longer used.
*   **Redundant Modules:**
    *   `src/metrics.py` appears largely unused (likely superseded by `src/analysis/metrics.py`).

## Recommendations
1.  **Fix Critical Bugs:** Immediately fix the broken function calls in `diffusion_pizza_generator.py` and `diffusion_training_integration.py`.
2.  **Break Dependency Cycles:** Refactor `src.rl` and `src.verification` to stop importing `src.pizza_detector`. Move necessary logic to `src/common` or `src/utils`.
3.  **Delete Zombie Files:** Remove `src/pizza-baking-detection-final.py` and `src/legacy/`.
4.  **Type Fixes:** Address the `None` safety issues in `pizza_rl_environment.py`.
