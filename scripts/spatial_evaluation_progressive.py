#!/usr/bin/env python3
"""
SPATIAL-3.1 Evaluation - Progressive Execution with Error Handling
"""
import os
import sys
import json
import torch
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_prerequisites():
    """Check if all required files and dependencies are available."""
    print("🔍 Checking prerequisites...")
    
    # Check model files
    spatial_model = Path("models/spatial_mllm/pizza_finetuned_v1.pth")
    if not spatial_model.exists():
        print(f"❌ Spatial model not found: {spatial_model}")
        return False
    print(f"✅ Spatial model found: {spatial_model} ({spatial_model.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Check test data
    test_data = Path("data/test")
    if not test_data.exists():
        print(f"❌ Test data not found: {test_data}")
        return False
    print(f"✅ Test data found: {test_data}")
    
    # Check output directories
    output_dir = Path("output")
    eval_dir = output_dir / "evaluation"
    viz_dir = output_dir / "visualizations"
    
    eval_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Output directories ready: {eval_dir}, {viz_dir}")
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")
    
    return True

def test_model_loading():
    """Test if we can load the spatial model."""
    print("\n🔄 Testing model loading...")
    
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        print("✅ Transformers imported successfully")
        
        # Try to load the base model first
        print("🔄 Loading base BLIP model...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        print("✅ Base BLIP processor loaded")
        
        # Try to load our spatial model
        print("🔄 Loading spatial model...")
        model_path = Path("models/spatial_mllm/pizza_finetuned_v1.pth")
        
        # First, let's check what's actually in the model file
        print(f"🔍 Examining model file: {model_path}")
        model_data = torch.load(model_path, map_location='cpu')
        
        if isinstance(model_data, dict):
            print(f"✅ Model data is a dictionary with keys: {list(model_data.keys())}")
            if 'model_state_dict' in model_data:
                print("✅ Found model_state_dict")
            if 'metadata' in model_data:
                print(f"✅ Found metadata: {model_data['metadata']}")
        else:
            print(f"✅ Model data type: {type(model_data)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_lightweight_evaluation():
    """Run a lightweight version of the evaluation to test the framework."""
    print("\n🔄 Running lightweight evaluation...")
    
    try:
        # Create a simple comparison report with mock data
        eval_dir = Path("output/evaluation")
        
        # Mock comparison results (in a real scenario, this would come from actual model evaluation)
        mock_results = {
            "evaluation_info": {
                "task": "SPATIAL-3.1: Spatial-MLLM vs Standard-MLLM Comparison",
                "timestamp": datetime.now().isoformat(),
                "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            },
            "models_compared": {
                "spatial_mllm": {
                    "name": "Spatial-MLLM Pizza Classifier",
                    "model_path": "models/spatial_mllm/pizza_finetuned_v1.pth",
                    "status": "loaded_successfully"
                },
                "standard_blip": {
                    "name": "Standard BLIP Classifier", 
                    "status": "baseline_comparison"
                }
            },
            "test_dataset": {
                "total_samples": 0,  # Will be updated when we count actual samples
                "challenging_cases": {
                    "burnt": 0,
                    "uneven": 0,
                    "mixed": 0,
                    "progression": 0,
                    "segment": 0
                }
            },
            "placeholder_metrics": {
                "spatial_mllm": {
                    "accuracy": "pending_evaluation",
                    "precision": "pending_evaluation", 
                    "recall": "pending_evaluation",
                    "f1_score": "pending_evaluation"
                },
                "standard_model": {
                    "accuracy": "pending_evaluation",
                    "precision": "pending_evaluation",
                    "recall": "pending_evaluation", 
                    "f1_score": "pending_evaluation"
                }
            },
            "status": "framework_ready_for_full_evaluation",
            "next_steps": [
                "Load and test spatial model inference",
                "Process test dataset with challenging cases",
                "Compare performance metrics",
                "Generate attention visualizations",
                "Complete quantitative analysis"
            ]
        }
        
        # Count actual test samples
        test_dir = Path("data/test")
        if test_dir.exists():
            total_samples = 0
            challenging_cases = {}
            
            for challenge_type in ["burnt", "uneven", "mixed", "progression", "segment"]:
                challenge_dir = test_dir / challenge_type
                if challenge_dir.exists():
                    image_files = list(challenge_dir.glob("*.jpg")) + list(challenge_dir.glob("*.png"))
                    challenging_cases[challenge_type] = len(image_files)
                    total_samples += len(image_files)
                    print(f"✅ Found {len(image_files)} {challenge_type} samples")
            
            mock_results["test_dataset"]["total_samples"] = total_samples
            mock_results["test_dataset"]["challenging_cases"] = challenging_cases
        
        # Save preliminary report
        report_path = eval_dir / "spatial_vs_standard_comparison.json"
        with open(report_path, 'w') as f:
            json.dump(mock_results, f, indent=2)
        
        print(f"✅ Preliminary report created: {report_path}")
        print(f"✅ Found {mock_results['test_dataset']['total_samples']} test samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Lightweight evaluation failed: {e}")
        return False

def main():
    """Main execution function."""
    print("🚀 SPATIAL-3.1: Progressive Evaluation Startup")
    print("=" * 60)
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites check failed")
        return 1
    
    # Step 2: Test model loading
    if not test_model_loading():
        print("❌ Model loading test failed")
        return 1
    
    # Step 3: Run lightweight evaluation
    if not run_lightweight_evaluation():
        print("❌ Lightweight evaluation failed")
        return 1
    
    print("\n✅ SPATIAL-3.1 Framework Validation Complete!")
    print("=" * 60)
    print("📊 Summary:")
    print("   • Prerequisites: ✅ Verified")
    print("   • Model Loading: ✅ Tested")
    print("   • Framework: ✅ Ready")
    print("   • Report: ✅ Created")
    print("\n🔄 Next: Run full evaluation with actual model inference")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
