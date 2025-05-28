# RP2040 Pizza-Erkennungssystem - Next Steps

**Datum:** 2025-05-27

This document outlines the recommended next steps for the RP2040 Pizza-Erkennungssystem, based on the findings from the recent status and error reports. It is intended to guide developers and other agents involved in the project.

## 1. Immediate Actions/Fixes

These actions should be prioritized to ensure the project documentation is accurate and critical dependencies are met.

### 1.1. Review and Update `PROJECT_STATUS.txt`

The main project status document (`docs/status/PROJECT_STATUS.txt`) requires several updates to reflect the current state accurately:

*   **Status of `scripts/augment_dataset.py` and `scripts/classify_images.py`:**
    *   **Action:** Modify "Offener Punkt 5" to reflect that these scripts are now considered functional and not empty/incomplete. Reference the completion of tasks `DATEN-1.1` and `DATEN-1.2` from `aufgaben.txt`.
*   **Class Name Loading in `analyze_performance_logs.py`:**
    *   **Action:** Update "Offener Punkt 4" to indicate that `analyze_performance_logs.py` now loads class names from a configuration file, as per the completion of task `PERF-2.2`.
*   **SD Card Logging Status:**
    *   **Action:** Clarify the status of SD card logging.
        *   Verify if the `ENABLE_SD_CARD` preprocessor macro is active by default in the main/release firmware builds.
        *   Confirm if comprehensive hardware validation for SD card logging has been completed.
        *   Update the "On-Device Performance-Logging" section and potentially `PERF-1.1` in `aufgaben.txt` to accurately reflect whether this is fully complete, in hardware validation, or if firmware enablement is pending for certain build targets.

### 1.2. Verify Task Tracking for `DATEN-1.3` (Unify class names)

*   **Action:** Review the completion status of `DATEN-1.3` in `aufgaben.txt`. While the underlying class name inconsistency was recently fixed (as confirmed by updates to `src/utils/constants.py`), the task tracking should accurately represent the timeline. If it was marked complete prematurely, consider adding a note or ensuring the resolution date aligns with the actual fix.

### 1.3. Graphviz Installation

*   **Context:** One test in the automated test suite is currently skipped due to the missing Graphviz dependency.
*   **Action:**
    1.  **Development Environment:** Instruct developers to install Graphviz.
        *   For Debian/Ubuntu: `sudo apt-get update && sudo apt-get install -y graphviz`
        *   For macOS (using Homebrew): `brew install graphviz`
        *   For other systems: Refer to official Graphviz installation guides.
    2.  **Python Bindings:** Ensure the `graphviz` Python package is listed in the project's `requirements.txt` (or equivalent dependency management file) and installed in the virtual environment: `pip install graphviz`
    3.  **CI/CD Environment:** Update the CI/CD workflow (`.github/workflows/model_pipeline.yml` or similar) to include steps for installing the Graphviz system package.
    4.  **Verification:** After installation, run the full test suite (`scripts/run_pizza_tests.py`) to confirm that all 50 tests pass and none are skipped.

### 1.4. Investigate 'segment' Augmentation Placeholder

*   **Context:** The augmentation function for the "segment" class in `scripts/augment_dataset.py` currently contains a `pass` statement.
*   **Action:**
    1.  **Determine Intent:** Clarify with the team or responsible developer if specific data augmentation techniques are planned for the "segment" class.
    2.  **If Planned:** Create a new task/issue in the project's issue tracker for implementing the "segment"-specific augmentations. Assign it accordingly.
    3.  **If Not Planned:** If no specific augmentation is deemed necessary for this class, remove the placeholder `elif aug_type == 'segment': pass` block from `scripts/augment_dataset.py` to avoid confusion.

## 2. Further Investigation & Verification

These items require deeper technical investigation or validation on actual hardware.

### 2.1. SQLAlchemy Warnings (`src/chatlist_ki/`)

*   **Context:** The `PROJECT_STATUS.txt` mentions pending SQLAlchemy warnings. Runtime observation was previously hindered by script execution errors.
*   **Action for Developers:**
    1.  **Resolve Execution Errors:** First, ensure the `src/chatlist_ki/__main__.py` script (or relevant test suites for this module) can be executed without the "Internal error occurred when running command" issue. This may involve debugging the script's environment setup or its interaction with the execution sandbox.
    2.  **Local Monitoring:** Run the `src/chatlist_ki/` application or its associated tests in a local development environment.
    3.  **Observe Warnings:** Monitor the console output closely for any `DeprecationWarning` or `SAWarning` messages originating from SQLAlchemy.
    4.  **Address Warnings:** If warnings are identified, consult the SQLAlchemy documentation to understand the cause and update the code to use the recommended newer patterns. Task `PERF-2.3` in `aufgaben.txt` tracks this.

### 2.2. SD Card Logging (Hardware Validation)

*   **Context:** Ensure the SD card logging functionality is robust and reliable on the physical RP2040 device.
*   **Action:**
    1.  **Firmware Build Configuration:** Confirm that the `ENABLE_SD_CARD` macro (or equivalent build flag) is correctly set for firmware builds intended to have SD card logging enabled.
    2.  **Hardware Testing:** Conduct thorough tests of the SD card logging feature on the actual RP2040 hardware setup. This should include:
        *   Writing logs over extended periods.
        *   Testing behavior when the SD card is full or nearly full.
        *   Verifying log file integrity.
        *   Measuring any performance impact on the main application.
    3.  **Documentation:** Update all relevant documentation (`PROJECT_STATUS.txt`, `aufgaben.txt` for `PERF-1.1`) based on the hardware validation results.

## 3. Consult Project Backlog/Roadmap

The project has a substantial backlog of defined tasks and planned features. These should be reviewed and prioritized.

*   **Action:**
    1.  **Review `PROJECT_STATUS.txt`:** Systematically go through the "Offene Punkte" and "Geplante Aufgaben (KI-Agent Roadmap)" sections.
    2.  **Review `aufgaben.txt` Checklists:** Examine all unchecked or partially checked (`[ ]`, `[~]`, `[x]?`) items in the detailed checklists (SPEICHER, DATEN, MODELL, PERF, DIFFUSION, ENERGIE, HWEMU).
    3.  **Prioritization:** Based on current project goals, dependencies, and resource availability, prioritize the implementation of these items. Key areas often highlighted include:
        *   **Efficient Energy Management:** This is a critical goal for battery-powered devices. Address any remaining tasks in the `ENERGIE-*` checklist and the general open point.
        *   **CNN Optimizations:** Finalize any pending tasks from `MODELL-*` (e.g., full implementation and validation of Early Exit if still beneficial, further Pruning/Clustering if required).
        *   **Hardware/Emulator Integration:** Complete tasks like `HWEMU-1.2` (DMA-Transfer for Kameradaten im Emulator) and any other pending hardware bring-up or emulation tasks.
        *   **Diffusion Model Data Generation:** If the generated data significantly improves model performance (based on `DIFFUSION-4.2` A/B tests), complete any remaining generation (`DIFFUSION-3.2`) and evaluation (`DIFFUSION-4.1`) tasks.
        *   **Automated Regression Tests:** Implement `PERF-4.1` to ensure long-term code quality and performance stability.

## 4. Documentation Maintenance

Consistent and accurate documentation is crucial for project continuity and collaboration.

*   **Action:**
    1.  **Regular Reviews:** Schedule periodic reviews of all key documentation files (`PROJECT_STATUS.txt`, `aufgaben.txt`, `COMPLETED_TASKS.md`, `README.md`, and other docs in `docs/`).
    2.  **Synchronization:** Ensure that task statuses, completed work, and open issues are consistent across all documents.
    3.  **Clarity and Accuracy:** Update information to reflect the latest code changes, architectural decisions, and experimental findings. Remove outdated information or clearly mark it as such.

By systematically addressing these next steps, the RP2040 Pizza-Erkennungssystem can progress towards its goals with increased clarity, stability, and efficiency.
