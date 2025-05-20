# Pizza Dataset Class Distribution Analysis

## Summary
This report presents the distribution of images across different pizza classes in the training dataset.

## Data Sources
The analysis includes images from two dataset directories:
- `augmented_pizza`: Organized with class subdirectories
- `augmented_pizza_legacy`: Contains images with class information in their filenames

## Class Distribution Results

| Class | Count | Percentage |
|-------|-------|------------|
| basic | 31 | 50.00% |
| burnt | 27 | 43.55% |
| combined | 1 | 1.61% |
| mixed | 1 | 1.61% |
| progression | 1 | 1.61% |
| segment | 1 | 1.61% |
| **Total** | **62** | **100%** |

## Observations
- The dataset shows a significant imbalance with most images belonging to only two classes: "basic" (50%) and "burnt" (43.55%)
- The remaining classes ("combined", "mixed", "progression", "segment") are severely underrepresented with only one image each (1.61%)
- This imbalance could lead to poor model performance on underrepresented classes

## Recommendations
1. Collect or generate more data for underrepresented classes
2. Consider data augmentation techniques to artificially increase the number of samples in minority classes
3. Use class weighting during model training to account for class imbalance
4. Consider evaluation metrics that are robust to class imbalance (e.g., precision, recall, F1-score)

## Visualizations
The distribution is visualized in bar chart and pie chart format in the following files:
- `class_distribution_train.png`
- `class_distribution_pie.png`
