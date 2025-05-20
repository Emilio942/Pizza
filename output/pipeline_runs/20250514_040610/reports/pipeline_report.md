# Pizza Detection Pipeline Report

Generated on: 2025-05-14 04:09:30

## Pipeline Summary

- **Duration:** 123.33 minutes
- **Total Scripts:** 5
- **Success Rate:** 60.00% (3 successful, 2 failed)
- **Overall Status:** Failure

## Script Execution Details

| Script | Status | Duration (s) | Retries |
|--------|--------|--------------|---------|
| test_image_preprocessing.py | Success | 3.00 | 0 |
| augment_dataset.py | Failed | 4.00 | 3 |
| run_pizza_tests.py | Success | 0.00 | 0 |
| verify_model.py | Failed | 4.00 | 3 |
| test_temporal_smoothing.py | Success | 4.00 | 0 |

## Validation Results

- **Total Scripts Validated:** 75
- **Passed:** 28
- **With Warnings:** 47
- **Failed:** 0
- **Pass Rate:** 37.33%


## Integration Analysis

- **Overall Success Rate:** 50.0%
- **Total Categories:** 3
- **Total Issues:** 2

### Category Success Rates

| Category | Success Rate | Successful Scripts | Total Scripts |
|----------|--------------|-------------------|---------------|
| preprocessing | 50.0% | 1 | 2 |
| testing | 100.0% | 2 | 2 |
| unknown | 0.0% | 0 | 1 |

### Integration Issues

| Type | Details |
|------|--------|
| Failed Script | Script failed after 3 retries |
| Failed Script | Script failed after 3 retries |

## Recommendations

- Review and fix the failed scripts to improve pipeline reliability.
- Address integration issues between script categories.
- Focus on improving scripts in the 'preprocessing' category.
- Focus on improving scripts in the 'unknown' category.

## Failed Scripts Details

### augment_dataset.py

- **Duration:** 4.00 seconds
- **Retries:** 3
- **Log File:** /home/emilio/Documents/ai/pizza/output/pipeline_runs/20250514_040610/logs/augment_dataset.log

### verify_model.py

- **Duration:** 4.00 seconds
- **Retries:** 3
- **Log File:** /home/emilio/Documents/ai/pizza/output/pipeline_runs/20250514_040610/logs/verify_model.log

