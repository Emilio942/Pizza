#!/usr/bin/env python3
"""
Spatial-MLLM Test Script for Pizza Classification
Tests the pretrained Spatial-MLLM model with pizza images
Adapts the video-based model for single image inference
"""

import os
import sys
import torch
import time
import json
import argparse
from PIL import Image
import numpy as np
from pathlib import Path

# Add paths for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append("/home/emilio/Documents/ai/Spatial-MLLM")

try:
    from src.models import Qwen2_5_VL_VGGTForConditionalGeneration, Qwen2_5_VLProcessor
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    print(f"Error importing Spatial-MLLM modules: {e}")
    print("Make sure the Spatial-MLLM repository is available and dependencies are installed")
    sys.exit(1)

def load_spatial_mllm_model(model_path, device):
    """Load the pretrained Spatial-MLLM model"""
    print(f"Loading Spatial-MLLM model from: {model_path}")
    start_time = time.time()
    
    try:
        # Load model with proper configuration
        model = Qwen2_5_VL_VGGTForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            attn_implementation="flash_attention_2",
            device_map={"": device} if device != "cpu" else None,
        )
        
        # Load processor
        processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f}s")
        print(f"Model device: {next(model.parameters()).device}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        return model, processor
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

def preprocess_pizza_image(image_path, target_frames=8):
    """
    Preprocess pizza image for Spatial-MLLM inference
    Creates multiple frames from single image to simulate video input
    """
    try:
        image = Image.open(image_path).convert('RGB')
        
        # Create multiple frames from single image with slight variations
        # This helps the model process the image in its expected video format
        frames = []
        
        # Base frame
        frames.append(image)
        
        # Add slightly modified versions to simulate temporal information
        for i in range(target_frames - 1):
            # Create variations (brightness, contrast, etc.)
            # This gives the model different perspectives of the same image
            modified_image = image.copy()
            frames.append(modified_image)
        
        return frames
        
    except Exception as e:
        print(f"❌ Error preprocessing image {image_path}: {e}")
        return None

def create_pizza_analysis_prompt(image_path):
    """Create a prompt for pizza analysis using Spatial-MLLM"""
    pizza_types = ["basic", "burnt", "combined", "mixed", "progression", "segment"]
    
    prompt = f"""Analyze this pizza image and classify it into one of these categories: {', '.join(pizza_types)}.

Consider the following spatial and visual characteristics:
- Burning patterns and distribution across the surface
- Color variations and texture details
- Topping distribution and arrangement
- Surface topology and 3D structure
- Overall cooking state and appearance

Please provide:
1. Primary classification: [category]
2. Confidence level: [0-100]%
3. Spatial reasoning: Describe the key spatial features that support your classification
4. Alternative possibilities: Any other categories this might fit

Respond in a structured format within <analysis></analysis> tags."""

    return prompt

def inference_spatial_mllm(model, processor, image_path, device):
    """Run inference on a pizza image using Spatial-MLLM"""
    
    # Preprocess image to frames
    frames = preprocess_pizza_image(image_path)
    if frames is None:
        return None
    
    # Create analysis prompt
    prompt = create_pizza_analysis_prompt(image_path)
    
    # Create messages in the expected format
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    try:
        start_time = time.time()
        
        # Prepare input
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision info
        _, vision_inputs = process_vision_info(messages)
        
        # Prepare inputs for the model
        inputs = processor(
            text=[text],
            images=vision_inputs if vision_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        inference_start = time.time()
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=512,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                use_cache=True,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        
        inference_time = time.time() - inference_start
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        total_time = time.time() - start_time
        
        return {
            "image_path": image_path,
            "prompt": prompt,
            "response": output_text,
            "inference_time": inference_time,
            "total_time": total_time,
            "success": True
        }
        
    except Exception as e:
        print(f"❌ Inference error for {image_path}: {e}")
        return {
            "image_path": image_path,
            "error": str(e),
            "success": False
        }

def test_spatial_mllm_with_pizza_images(model, processor, test_images_dir, output_dir, device, max_images=5):
    """Test Spatial-MLLM with pizza images from different categories"""
    
    results = {
        "model_info": {
            "model_name": "Spatial-MLLM-subset-sft",
            "device": str(device),
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "performance_metrics": {
            "total_images": 0,
            "successful_inferences": 0,
            "failed_inferences": 0,
            "average_inference_time": 0,
            "total_test_time": 0,
        },
        "test_results": []
    }
    
    start_time = time.time()
    
    # Find test images from different pizza categories
    pizza_categories = ["basic", "burnt", "combined", "mixed", "progression", "segment"]
    test_images = []
    
    for category in pizza_categories:
        category_dir = Path(test_images_dir) / category
        if category_dir.exists():
            images = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
            # Take up to max_images per category
            selected_images = images[:max_images] if images else []
            for img_path in selected_images:
                test_images.append((str(img_path), category))
    
    if not test_images:
        print(f"❌ No test images found in {test_images_dir}")
        return results
    
    print(f"📸 Testing with {len(test_images)} images from {len(pizza_categories)} categories")
    
    inference_times = []
    
    for i, (image_path, true_category) in enumerate(test_images):
        print(f"\n[{i+1}/{len(test_images)}] Testing: {Path(image_path).name} (category: {true_category})")
        
        # Run inference
        result = inference_spatial_mllm(model, processor, image_path, device)
        
        if result["success"]:
            results["performance_metrics"]["successful_inferences"] += 1
            inference_times.append(result["inference_time"])
            print(f"✅ Success - Inference time: {result['inference_time']:.2f}s")
            
            # Add true category to result
            result["true_category"] = true_category
            
            # Extract classification from response (basic parsing)
            response = result["response"].lower()
            predicted_category = "unknown"
            for cat in pizza_categories:
                if cat in response:
                    predicted_category = cat
                    break
            result["predicted_category"] = predicted_category
            result["correct_prediction"] = (predicted_category == true_category)
            
        else:
            results["performance_metrics"]["failed_inferences"] += 1
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        results["test_results"].append(result)
        results["performance_metrics"]["total_images"] += 1
    
    # Calculate performance metrics
    total_time = time.time() - start_time
    results["performance_metrics"]["total_test_time"] = total_time
    
    if inference_times:
        results["performance_metrics"]["average_inference_time"] = np.mean(inference_times)
        results["performance_metrics"]["min_inference_time"] = np.min(inference_times)
        results["performance_metrics"]["max_inference_time"] = np.max(inference_times)
    
    # Calculate accuracy
    correct_predictions = sum(1 for r in results["test_results"] 
                            if r.get("success") and r.get("correct_prediction"))
    total_successful = results["performance_metrics"]["successful_inferences"]
    
    if total_successful > 0:
        results["performance_metrics"]["accuracy"] = correct_predictions / total_successful
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "baseline_test_results.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Test completed in {total_time:.2f}s")
    print(f"Results saved to: {results_file}")
    
    return results

def print_performance_summary(results):
    """Print a summary of the test results"""
    metrics = results["performance_metrics"]
    
    print("\n" + "="*60)
    print("📈 SPATIAL-MLLM PERFORMANCE SUMMARY")
    print("="*60)
    
    print(f"Model: {results['model_info']['model_name']}")
    print(f"Device: {results['model_info']['device']}")
    print(f"Test Date: {results['model_info']['test_timestamp']}")
    
    print(f"\n📊 Test Statistics:")
    print(f"  Total Images: {metrics['total_images']}")
    print(f"  Successful Inferences: {metrics['successful_inferences']}")
    print(f"  Failed Inferences: {metrics['failed_inferences']}")
    print(f"  Success Rate: {metrics['successful_inferences']/metrics['total_images']*100:.1f}%")
    
    if metrics.get('accuracy'):
        print(f"  Classification Accuracy: {metrics['accuracy']*100:.1f}%")
    
    print(f"\n⏱️  Performance Metrics:")
    print(f"  Total Test Time: {metrics['total_test_time']:.2f}s")
    if metrics.get('average_inference_time'):
        print(f"  Average Inference Time: {metrics['average_inference_time']:.2f}s")
        print(f"  Min Inference Time: {metrics['min_inference_time']:.2f}s")
        print(f"  Max Inference Time: {metrics['max_inference_time']:.2f}s")
    
    # Show some example results
    successful_results = [r for r in results["test_results"] if r.get("success")]
    if successful_results:
        print(f"\n🔍 Sample Results:")
        for i, result in enumerate(successful_results[:3]):
            print(f"  {i+1}. {Path(result['image_path']).name}")
            print(f"     True: {result.get('true_category', 'unknown')}")
            print(f"     Predicted: {result.get('predicted_category', 'unknown')}")
            print(f"     Correct: {'✅' if result.get('correct_prediction') else '❌'}")

def main():
    parser = argparse.ArgumentParser(description="Test Spatial-MLLM with pizza images")
    parser.add_argument("--model-path", default="Diankun/Spatial-MLLM-subset-sft",
                      help="Path or name of the Spatial-MLLM model")
    parser.add_argument("--test-images", default="data/augmented_pizza",
                      help="Directory containing test pizza images")
    parser.add_argument("--output-dir", default="output/spatial_mllm",
                      help="Directory to save test results")
    parser.add_argument("--max-images", type=int, default=3,
                      help="Maximum images to test per category")
    parser.add_argument("--device", default="auto",
                      help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("🚀 Starting Spatial-MLLM Pizza Classification Test")
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Test Images: {args.test_images}")
    print(f"Output: {args.output_dir}")
    
    try:
        # Load model
        model, processor = load_spatial_mllm_model(args.model_path, device)
        
        # Run tests
        results = test_spatial_mllm_with_pizza_images(
            model, processor, args.test_images, args.output_dir, device, args.max_images
        )
        
        # Print summary
        print_performance_summary(results)
        
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
