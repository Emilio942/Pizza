# DIFFUSION-3.1: Dataset Balancing Strategy Using Generated Images

**Date:** 2025-01-28  
**Status:** Complete Strategy Definition  
**Previous Task:** DIFFUSION-2.1 (Targeted Diffusion Pipeline Implementation) ✅  

## Executive Summary

This document defines a comprehensive strategy for balancing the pizza dataset using AI-generated synthetic images from the targeted diffusion pipeline implemented in DIFFUSION-2.1. The strategy addresses current class imbalances by generating specific quantities of images with targeted properties to achieve optimal dataset balance.

## Current Dataset Analysis (DATEN-2.1 Findings)

### Dataset State Overview
Based on the latest class distribution analysis:

```json
Current Class Distribution:
{
  "basic": 30 images (53.57%),
  "burnt": 26 images (46.43%),
  "combined": 0 images (0.0%),
  "mixed": 0 images (0.0%),
  "progression": 0 images (0.0%),
  "segment": 0 images (0.0%)
}

Recent Diffusion Analysis:
{
  "basic": 29 images,
  "burnt": 4 images,
  "combined": 17 images,
  "mixed": 7 images,
  "progression": 0 images,
  "segment": 0 images
}
```

### Identified Issues
1. **Severe Class Imbalance**: Only 2 out of 6 classes have substantial representation
2. **Missing Classes**: 3 classes (`combined`, `mixed`, `progression`, `segment`) have 0-17 images
3. **Underrepresented Classes**: `burnt`, `mixed`, `progression`, `segment` need significant augmentation
4. **Target Balance**: Achieve minimum 80% of largest class for all classes

## Balancing Strategy Framework

### Target Distribution Goals
- **Target Size**: 50 images per class (balanced across all 6 classes)
- **Minimum Threshold**: 40 images per class (80% of target)
- **Total Target Dataset**: 300 images
- **Quality Standards**: All generated images must pass property verification

### Generation Requirements by Class

#### 1. Basic Class
- **Current**: 29-30 images
- **Target**: 50 images
- **Generate**: 20-21 additional images
- **Focus**: Foundation pizza images with varied lighting

#### 2. Burnt Class  
- **Current**: 4-26 images (varying by analysis)
- **Target**: 50 images
- **Generate**: 24-46 additional images
- **Focus**: All burn levels with realistic progression

#### 3. Combined Class
- **Current**: 0-17 images
- **Target**: 50 images  
- **Generate**: 33-50 additional images
- **Focus**: Complete pizzas with toppings

#### 4. Mixed Class
- **Current**: 0-7 images
- **Target**: 50 images
- **Generate**: 43-50 additional images
- **Focus**: Mixed states and varied configurations

#### 5. Progression Class
- **Current**: 0 images
- **Target**: 50 images
- **Generate**: 50 additional images
- **Focus**: Cooking progression sequences

#### 6. Segment Class
- **Current**: 0 images
- **Target**: 50 images
- **Generate**: 50 additional images
- **Focus**: Pizza segments and portions

## Detailed Generation Parameters

### Prompt Templates by Class

#### Basic Class Prompts
```python
BASIC_CLASS_PROMPTS = [
    "simple pizza dough, basic ingredients, professional food photography",
    "plain pizza base with minimal toppings, clean presentation",
    "basic pizza with simple cheese and sauce, studio lighting",
    "traditional pizza with basic toppings, restaurant quality"
]

BASIC_VARIATIONS = [
    "different crust textures: thin, thick, medium",
    "sauce variations: light, moderate, heavy",
    "cheese distributions: even, uneven, minimal"
]
```

#### Burnt Class Prompts
```python
BURNT_CLASS_PROMPTS = {
    "slightly_burnt": [
        "pizza slightly burnt with light browning on edges, still appetizing",
        "pizza with gentle browning and minimal burn spots, golden-brown",
        "pizza lightly overcooked with subtle burn marks, artisanal appearance"
    ],
    "moderately_burnt": [
        "pizza moderately burnt with visible brown and dark spots",
        "pizza with noticeable burn marks and darker browning, rustic style",
        "pizza with moderate burning, mix of golden and dark brown areas"
    ],
    "severely_burnt": [
        "pizza severely burnt with black charred areas, overcooked",
        "pizza with heavy burning and black spots, charred crust",
        "pizza completely overcooked with dark charred patterns"
    ]
}
```

#### Combined Class Prompts
```python
COMBINED_CLASS_PROMPTS = [
    "complete pizza with sauce, cheese, and toppings, professional photography",
    "fully assembled pizza with multiple toppings, restaurant presentation",
    "finished pizza with all ingredients combined, appetizing appearance",
    "gourmet pizza with complete toppings configuration, food styling"
]
```

#### Mixed Class Prompts
```python
MIXED_CLASS_PROMPTS = [
    "pizza with mixed cooking states, uneven preparation patterns",
    "pizza with varying toppings distribution, irregular configuration",
    "pizza with mixed ingredient applications, diverse surface patterns",
    "pizza showing varied cooking stages across surface, documentary style"
]
```

#### Progression Class Prompts
```python
PROGRESSION_CLASS_PROMPTS = [
    "pizza in cooking progression, transitioning from raw to cooked",
    "pizza showing cooking sequence, time-lapse style documentation",
    "pizza with gradient cooking effect, progression from uncooked areas",
    "pizza demonstrating cooking stages, educational food photography"
]
```

#### Segment Class Prompts
```python
SEGMENT_CLASS_PROMPTS = [
    "pizza slice or segment, detailed close-up view, professional photography",
    "individual pizza portion, cross-section view, studio lighting",
    "pizza wedge showing internal structure, food photography",
    "cut pizza segment with visible layers, appetizing presentation"
]
```

### Lighting Condition Distribution

Each class should include varied lighting conditions:

#### Lighting Distribution Strategy
- **Overhead Harsh**: 25% of images (dramatic shadows, high contrast)
- **Side Dramatic**: 25% of images (strong lateral lighting, depth)
- **Dim Ambient**: 25% of images (restaurant atmosphere, soft shadows)
- **Backlit Rim**: 25% of images (rim lighting, artistic effect)

#### Burn Level Distribution (for applicable classes)
- **Slightly Burnt**: 40% of burn-related images
- **Moderately Burnt**: 35% of burn-related images  
- **Severely Burnt**: 25% of burn-related images

## Implementation Plan

### Phase 1: Infrastructure Preparation (Completed ✅)
- ✅ Targeted diffusion pipeline implemented
- ✅ Property verification system active
- ✅ Template systems for lighting and burn levels
- ✅ Memory optimization features enabled

### Phase 2: Systematic Generation (To Execute)

#### Generation Sequence
1. **Basic Class** (20 images)
   - 5 images per lighting condition
   - Focus on fundamental pizza appearances
   - Verify basic pizza properties

2. **Burnt Class** (25-45 images)
   - Distribute across burn levels: 10 slight, 8 moderate, 7 severe
   - Apply all lighting conditions to each burn level
   - Strong emphasis on burn verification

3. **Combined Class** (35-50 images)
   - Complete pizzas with full toppings
   - Varied lighting for appetizing presentation
   - Focus on completeness verification

4. **Mixed Class** (45-50 images)
   - Irregular patterns and mixed states
   - Creative lighting for documentary style
   - Emphasis on variation and authenticity

5. **Progression Class** (50 images)
   - Cooking transition sequences
   - Time-progression documentation style
   - Focus on gradient cooking verification

6. **Segment Class** (50 images)
   - Close-up pizza slices and portions
   - Detailed internal structure visibility
   - Cross-section and texture verification

### Phase 3: Quality Control and Verification

#### Verification Criteria
- **Property Verification Score**: ≥0.6 for all images
- **Lighting Verification**: Matches target lighting condition
- **Burn Level Verification**: Appropriate burn characteristics (where applicable)
- **Class Characteristics**: Meets class-specific requirements

#### Quality Metrics
- **Target Success Rate**: ≥80% first-attempt success
- **Maximum Retries**: 3 attempts per image
- **Verification Threshold**: 0.6 for property matching
- **Overall Quality**: Professional food photography standards

### Phase 4: Integration and Validation

#### Dataset Integration Steps
1. Organize generated images by class directories
2. Update metadata with generation parameters
3. Run comprehensive class distribution analysis
4. Validate balanced distribution achievement
5. Document generation statistics and success rates

## Expected Outcomes

### Quantitative Results
- **Total Dataset Size**: ~300 images (up from 56)
- **Class Balance**: All classes at 50±10 images
- **Balance Ratio**: All classes ≥80% of largest class
- **Quality Assurance**: All images verified for target properties

### Qualitative Improvements
- **Model Training**: Improved performance across all classes
- **Class Recognition**: Better representation of rare classes
- **Robustness**: Enhanced model generalization
- **Dataset Diversity**: Rich variety in lighting and cooking states

## Generation Commands and Scripts

### CLI Command Template
```bash
# Basic class generation
python -m src.augmentation.targeted_diffusion_pipeline \
    --mode generate_lighting \
    --lighting overhead_harsh \
    --count 5 \
    --stage basic \
    --output-dir data/synthetic/balanced/basic \
    --seed 42

# Burnt class with burn level
python -m src.augmentation.targeted_diffusion_pipeline \
    --mode generate_burn_level \
    --burn-level moderately_burnt \
    --lighting side_dramatic \
    --count 8 \
    --stage burnt \
    --output-dir data/synthetic/balanced/burnt \
    --seed 123
```

### Automation Script
```python
# Full balancing automation (to be implemented)
python balance_with_diffusion.py \
    --target-per-class 50 \
    --min-threshold 40 \
    --verify-properties \
    --output-dir data/synthetic/balanced \
    --config-file diffusion_balance_config.json
```

## Risk Mitigation and Considerations

### Technical Risks
1. **Generation Quality**: Monitor verification scores, increase retries if needed
2. **Memory Constraints**: Use CPU offloading and attention slicing
3. **Processing Time**: Estimate 2-5 minutes per image, plan accordingly
4. **Storage Requirements**: ~1-2GB for complete balanced dataset

### Quality Assurance
1. **Manual Spot Checks**: Review 10% of generated images manually
2. **Distribution Validation**: Ensure even lighting/burn distribution
3. **Model Compatibility**: Test with existing classification pipelines
4. **Metadata Integrity**: Verify all generation parameters stored

### Fallback Strategies
1. **Lower Quality Threshold**: Reduce to 0.5 if generation fails repeatedly
2. **Alternative Prompts**: Use backup prompt variations for difficult classes
3. **Traditional Augmentation**: Supplement with rotation/brightness if needed
4. **Progressive Targets**: Achieve 80% balance first, then optimize to 100%

## Success Metrics and Validation

### Primary Success Criteria
- [ ] All 6 classes have ≥40 images (80% of target)
- [ ] No class has <80% representation relative to largest class
- [ ] ≥80% of generated images pass property verification
- [ ] Complete metadata documentation for all generated images

### Secondary Success Criteria
- [ ] All 6 classes have ≥50 images (target achieved)
- [ ] Even distribution of lighting conditions across classes
- [ ] Realistic burn progression in burn-related classes
- [ ] High-quality visual appearance suitable for model training

### Validation Process
1. **Automated Analysis**: Run class distribution analysis post-generation
2. **Visual Inspection**: Manual review of sample images per class
3. **Model Testing**: Quick classification test on balanced dataset
4. **Documentation Review**: Ensure complete generation metadata

## Conclusion

This strategy provides a comprehensive roadmap for achieving dataset balance using the sophisticated targeted diffusion pipeline implemented in DIFFUSION-2.1. By generating approximately 240-250 additional images across underrepresented classes, we will transform the current imbalanced dataset (56 images, 2 major classes) into a well-balanced training corpus (300 images, 6 balanced classes).

The combination of targeted property generation, comprehensive verification systems, and systematic quality control ensures that the synthetic images will enhance rather than degrade the overall dataset quality. This approach represents a significant advancement in automated dataset balancing using state-of-the-art generative AI techniques.

**Next Steps**: Execute the generation plan using the targeted diffusion pipeline, starting with the most underrepresented classes and working systematically through the defined generation sequence.
