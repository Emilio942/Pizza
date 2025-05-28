# RP2040 Pizza-Erkennungssystem - Error Report

**Datum:** 2025-05-27

This report details confirmed bugs, inconsistencies, potential issues requiring further investigation, and external dependencies affecting the RP2040 Pizza-Erkennungssystem.

## 1. Confirmed Bugs/Inconsistencies

### 1.1. Class Name Inconsistency (Resolved)

*   **Issue:** A discrepancy existed in the definition of pizza classes across different project files. `src/utils/constants.py` utilized a 4-class system, whereas `src/constants.py` and `data/class_definitions.json` (the source of truth for labeling) employed a 6-class system ("basic", "burnt", "combined", "mixed", "progression", "segment").
*   **Status:** **Resolved**. `src/utils/constants.py` was updated in a previous subtask to align with the 6-class system, ensuring consistency.
*   **Affected Files (Initially):**
    *   `src/utils/constants.py` (now fixed)
    *   `src/constants.py`
    *   `data/class_definitions.json`

### 1.2. Outdated Information in `PROJECT_STATUS.txt`

The main project status document (`docs/status/PROJECT_STATUS.txt`) contained some information that was found to be outdated during analysis:

*   **Status of `scripts/augment_dataset.py` and `scripts/classify_images.py`:**
    *   **Issue:** `PROJECT_STATUS.txt` (Offener Punkt 5) stated: "Problem: Skripte `augment_dataset.py` und `classify_images.py` scheinen leer oder unvollständig zu sein. Überprüfung notwendig."
    *   **Finding:** Analysis and review of `aufgaben.txt` (tasks `DATEN-1.1` and `DATEN-1.2` marked as complete) and successful generation of a status report (which relied on these scripts being functional for data analysis tasks) indicate these scripts are now considered complete and functional.
    *   **Status:** Documentation in `PROJECT_STATUS.txt` requires updating.

*   **Hardcoding of Class Names in `analyze_performance_logs.py`:**
    *   **Issue:** `PROJECT_STATUS.txt` (Offener Punkt 4) stated: "`analyze_performance_logs.py`: Klassennamen aus Konfigurationsdatei laden statt hartcodieren."
    *   **Finding:** Task `PERF-2.2` ("analyze_performance_logs.py robust machen (Klassennamen)") in `aufgaben.txt` is marked as complete (`[X]`). This implies the script now loads class names dynamically.
    *   **Status:** Documentation in `PROJECT_STATUS.txt` requires updating.

### 1.3. Contradictory Task Completion Status in Documentation

*   **Task `DATEN-1.3` (Unify class names):**
    *   **Issue:** `aufgaben.txt` marked `DATEN-1.3` as complete (`[x]`). However, the class name inconsistency (detailed in 1.1) persisted until it was actively resolved in a separate subtask. This indicates a potential premature marking of completion or a re-introduction of the issue.
    *   **Status:** The underlying class inconsistency is resolved. The documentation discrepancy highlights a need for careful status tracking.

*   **SD Card Logging Status:**
    *   **Issue:** `PROJECT_STATUS.txt` (Offene Punkte & Geplante Aufgaben) implies that SD card logging is still a pending task ("On-Device Performance-Logging (Punkt 8, teilweise): SD-Karten-Logging implementieren für umfassendere Datenerfassung"). However, `aufgaben.txt` marks `PERF-1.1` (SD-Karten-Logging implementieren) as complete (`[x]`).
    *   **Status:** This represents a documentation inconsistency. Further clarification is needed on whether the existing implementation meets all requirements or if further hardware validation/firmware enablement is the pending part (see Section 2.2).

### 1.4. Placeholder in Augmentation Script

*   **Issue:** The augmentation function for the class "segment" within `scripts/augment_dataset.py` currently contains a `pass` statement. This means no specific augmentation is applied for this class, effectively acting as a placeholder.
    ```python
    # In scripts/augment_dataset.py (illustrative)
    # ...
    # elif aug_type == 'segment':
    #     # TODO: Implement segment-specific augmentation
    #     pass
    # ...
    ```
*   **Impact:** If specific augmentations are desired or necessary for the "segment" class to improve model robustness or address class imbalances, this functionality is currently missing.
*   **Status:** Confirmed placeholder. Requires implementation if segment-specific augmentation is needed.
*   **Affected Files:** `scripts/augment_dataset.py`

## 2. Potential Issues & Items Requiring Further Investigation

### 2.1. SQLAlchemy Warnings

*   **Context:** `PROJECT_STATUS.txt` lists "Behebung von Warnungen in SQLAlchemy (veraltete Funktionen)" as an open point. A previous subtask aimed to observe these warnings by executing `src/chatlist_ki/__main__.py`.
*   **Issue:** Runtime verification of SQLAlchemy warnings was **inconclusive** due to persistent "Internal error occurred when running command" errors during the execution of the `src/chatlist_ki/__main__.py` script. These errors prevented observation of any script output, including potential SQLAlchemy warnings.
*   **Static Analysis Finding:** A static review of the `src/chatlist_ki/` codebase (specifically `database.py` and `models.py`) did not reveal obvious uses of deprecated SQLAlchemy patterns that would *guarantee* warnings in a standard SQLAlchemy 2.x environment. The usage of `datetime.now(timezone.UTC)` for `default` and `onupdate` in `models.py` appears correct and is the recommended way to handle timezone-aware default timestamps, mitigating a common source of warnings related to naive datetimes.
*   **Recommendation:**
    *   The "Internal error" preventing script execution needs to be diagnosed and resolved first.
    *   Once the script is runnable, local developer testing is essential to confirm the presence or absence of any SQLAlchemy warnings under the actual runtime conditions and SQLAlchemy version used by the project.
    *   If warnings are observed, they should be addressed by updating to current SQLAlchemy idioms. `PERF-2.3` in `aufgaben.txt` is correctly marked as not completed.

### 2.2. SD Card Logging (Firmware Enablement & Hardware Validation)

*   **Context:** As noted in 1.3, there's a discrepancy in the documented status of SD card logging. `aufgaben.txt` (PERF-1.1) suggests completion, while `PROJECT_STATUS.txt` implies it's pending.
*   **Issue:** While code for SD card logging exists (indicated by `PERF-1.1` completion, likely referring to emulator/software-side logic and conditional firmware code via `ENABLE_SD_CARD` macro), its default enablement in production firmware builds and comprehensive validation on actual RP2040 hardware is unclear.
*   **Potential Concern:** The functionality might be implemented but not actively used in all builds, or it may not have been fully tested for reliability and performance on the physical hardware with an SD card.
*   **Recommendation:**
    *   Clarify if the `ENABLE_SD_CARD` macro is set by default in release firmware builds.
    *   Perform thorough testing of SD card logging functionality on the physical RP2040 hardware to validate performance, reliability, and impact on overall system resources.
    *   Update documentation consistently based on these findings.

## 3. External Dependencies/Environment Issues

### 3.1. Graphviz Installation for Full Test Coverage

*   **Issue:** The `PROJECT_STATUS.txt` ("Test-Status") reports: "Alle Tests sind erfolgreich (49 bestanden, 1 übersprungen aufgrund fehlender Graphviz-Installation)." Task `PERF-2.1` in `aufgaben.txt` is marked `[x]?`, indicating uncertainty or external dependency for completion.
*   **Impact:** The absence of Graphviz in the testing/development environment prevents one test case from running, leading to incomplete test coverage. This means a part of the system's functionality, likely related to visualization or graph generation, is not being automatically verified.
*   **Recommendation:**
    *   Install Graphviz and its Python bindings in the development and CI/CD environments.
    *   Ensure the skipped test is then executed and passes.
    *   Update documentation to reflect full test coverage.
*   **Affected Components:** Test Suite, potentially visualization scripts.

This report summarizes the current known errors and areas needing attention. Addressing these points will improve the robustness, reliability, and maintainability of the RP2040 Pizza-Erkennungssystem.
