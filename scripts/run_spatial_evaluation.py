#!/usr/bin/env python3
"""
Quick launcher for SPATIAL-3.1 evaluation with progress tracking
"""
import sys
import os
sys.path.append('.')

print("🚀 Starting SPATIAL-3.1 Evaluation...")
print("=" * 60)

try:
    # Import the main evaluation script
    print("📦 Loading evaluation framework...")
    from scripts.spatial_vs_standard_evaluation import main
    
    print("✅ Framework loaded successfully!")
    print("🔄 Running comprehensive evaluation...")
    
    # Run the evaluation
    result = main()
    
    if result == 0:
        print("✅ SPATIAL-3.1 Evaluation completed successfully!")
    else:
        print("❌ Evaluation failed!")
        
except Exception as e:
    print(f"❌ Error during evaluation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
