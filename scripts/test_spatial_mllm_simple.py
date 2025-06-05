#!/usr/bin/env python3
"""
Simplified Spatial-MLLM Test Script for Pizza Classification
Tests the pretrained Spatial-MLLM model directly from Hugging Face
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

def test_basic_imports():
    """Test if all required dependencies are available"""
    print("🔍 Testing basic imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            print(f"   Current GPU: {torch.cuda.get_device_name()}")
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    try:
        from transformers import AutoTokenizer, AutoModel, AutoProcessor
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not available")
        return False
    
    try:
        from qwen_vl_utils import process_vision_info
        print("✅ Qwen VL utils available")
    except ImportError:
        print("⚠️  Qwen VL utils not available - will use fallback methods")
    
    try:
        from PIL import Image
        print("✅ PIL/Pillow available")
    except ImportError:
        print("❌ PIL/Pillow not available")
        return False
    
    return True

def test_model_availability():
    """Test if the Spatial-MLLM model is available on Hugging Face"""
    print("\n🔍 Testing model availability...")
    
    model_name = "Diankun/Spatial-MLLM-subset-sft"
    
    try:
        from transformers import AutoConfig
        
        print(f"Checking model: {model_name}")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        print(f"✅ Model config loaded successfully")
        print(f"   Model type: {config.model_type}")
        print(f"   Architecture: {config.architectures}")
        
        return True, model_name
        
    except Exception as e:
        print(f"❌ Error accessing model: {e}")
        return False, None

def test_model_loading(model_name, device):
    """Test loading the actual model"""
    print(f"\n🔍 Testing model loading on {device}...")
    
    try:
        from transformers import AutoTokenizer, AutoModel, AutoProcessor
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✅ Tokenizer loaded")
        
        print("Loading processor...")
        try:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            print("✅ Processor loaded")
        except Exception as e:
            print(f"⚠️  Processor loading failed: {e}")
            processor = None
        
        print("Loading model...")
        start_time = time.time()
        
        # Try different Auto classes for vision-language models
        model = None
        for auto_class_name in ["AutoModelForVision2Seq", "AutoModelForCausalLM", "AutoModel"]:
            try:
                print(f"   Trying {auto_class_name}...")
                if auto_class_name == "AutoModelForVision2Seq":
                    from transformers import AutoModelForVision2Seq
                    auto_class = AutoModelForVision2Seq
                elif auto_class_name == "AutoModelForCausalLM":
                    from transformers import AutoModelForCausalLM
                    auto_class = AutoModelForCausalLM
                else:
                    from transformers import AutoModel
                    auto_class = AutoModel
                
                model = auto_class.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                    device_map="auto" if device != "cpu" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                print(f"✅ Model loaded with {auto_class_name}")
                print(f"   Model class: {model.__class__.__name__}")
                break
                
            except Exception as e:
                print(f"   {auto_class_name} failed: {e}")
                continue
        
        if model is None:
            raise Exception("All Auto model classes failed")
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Parameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"   Model device: {next(model.parameters()).device}")
        
        return model, tokenizer, processor, load_time
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def test_vision_language_inference(model, tokenizer, processor, device):
    """Test vision-language inference with an actual pizza image"""
    print(f"\n🔍 Testing vision-language inference...")
    
    try:
        # Find a sample pizza image from the test data
        pizza_image_path = "/home/emilio/Documents/ai/pizza/data/test/sample_pizza_image.jpg"
        basic_image_path = "/home/emilio/Documents/ai/pizza/data/test/basic/sample_basic.jpg"
        
        # Try to find an available test image
        test_image_path = None
        for path in [pizza_image_path, basic_image_path]:
            if os.path.exists(path):
                test_image_path = path
                break
        
        if not test_image_path:
            # Fall back to creating a simple test image
            print("⚠️  No test images found, creating a simple test image...")
            test_image = Image.new('RGB', (224, 224), color=(255, 200, 100))  # Pizza-like orange color
            test_image_path = "/tmp/test_pizza.jpg"
            test_image.save(test_image_path)
        
        # Load the image
        image = Image.open(test_image_path).convert('RGB')
        print(f"   Using test image: {test_image_path}")
        print(f"   Image size: {image.size}")
        
        # Create a text prompt for pizza classification
        text_prompt = "What type of pizza cooking state is shown in this image? Classify as: basic, burnt, mixed, or ready."
        
        # Process inputs using the processor (for vision-language models)
        if processor:
            print("   Using processor for vision-language input...")
            try:
                # Try different processor input formats for Qwen VL models
                # Format 1: Standard format with messages
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": text_prompt}
                        ]
                    }
                ]
                
                # Apply chat template
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                
                # Process with image and text
                inputs = processor(
                    text=[text],
                    images=[image],
                    padding=True,
                    return_tensors="pt"
                )
                
            except Exception as e:
                print(f"   Qwen VL format failed: {e}")
                try:
                    # Format 2: Simple processor with both image and text
                    inputs = processor(
                        images=image,
                        text=text_prompt,
                        return_tensors="pt"
                    )
                except Exception as e2:
                    print(f"   Simple format failed: {e2}")
                    try:
                        # Format 3: Only image input
                        inputs = processor(images=image, return_tensors="pt")
                        # Add text separately
                        text_inputs = processor.tokenizer(text_prompt, return_tensors="pt")
                        inputs.update(text_inputs)
                    except Exception as e3:
                        print(f"   All processor formats failed: {e3}")
                        raise e3
        else:
            print("   No processor available, using manual image processing...")
            # Manual processing if no processor
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            pixel_values = transform(image).unsqueeze(0)
            input_ids = tokenizer(text_prompt, return_tensors="pt").input_ids
            
            inputs = {
                "pixel_values": pixel_values,
                "input_ids": input_ids
            }
        
        # Move inputs to device
        print("   Moving inputs to device...")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        print("   Running vision-language inference...")
        start_time = time.time()
        
        with torch.no_grad():
            try:
                # Try generate method
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
                )
                
                # Decode output
                if hasattr(outputs, 'sequences'):
                    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                else:
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
            except Exception as e:
                print(f"   Generate method failed: {e}")
                # Try forward pass instead
                outputs = model(**inputs)
                
                if hasattr(outputs, 'logits'):
                    # Get predicted class from logits
                    predicted_class = torch.argmax(outputs.logits, dim=-1)
                    generated_text = f"Predicted class: {predicted_class.item()}"
                else:
                    generated_text = f"Model output keys: {list(outputs.keys())}"
        
        inference_time = time.time() - start_time
        
        print(f"✅ Vision-language inference completed in {inference_time:.2f}s")
        print(f"   Input image: {test_image_path}")
        print(f"   Text prompt: {text_prompt}")
        print(f"   Model output: {generated_text}")
        
        return True, inference_time, generated_text
        
    except Exception as e:
        print(f"❌ Vision-language inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_pizza_classification(model, tokenizer, processor, device):
    """Test pizza classification on multiple images from different classes"""
    print(f"\n🔍 Testing pizza classification on different pizza types...")
    
    try:
        # Define pizza classes and test images
        pizza_classes = ["basic", "burnt", "mixed", "combined"]
        test_base_dir = "/home/emilio/Documents/ai/pizza/data/test"
        
        classification_results = []
        
        for pizza_class in pizza_classes:
            class_dir = os.path.join(test_base_dir, pizza_class)
            
            if not os.path.exists(class_dir):
                print(f"   ⚠️  Class directory not found: {class_dir}")
                continue
            
            # Find a sample image in this class
            sample_image = None
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    sample_image = os.path.join(class_dir, img_file)
                    break
            
            if not sample_image:
                print(f"   ⚠️  No sample images found in {class_dir}")
                continue
            
            print(f"   Testing {pizza_class} classification...")
            
            # Load image
            image = Image.open(sample_image).convert('RGB')
            
            # Create classification prompt
            text_prompt = f"Is this pizza {pizza_class}? Analyze the cooking state and classify this pizza image. What type of pizza cooking state is shown?"
            
            # Process inputs
            if processor:
                try:
                    # Try Qwen VL chat template format
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": text_prompt}
                            ]
                        }
                    ]
                    
                    text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    
                    inputs = processor(
                        text=[text],
                        images=[image],
                        padding=True,
                        return_tensors="pt"
                    )
                except Exception as e:
                    try:
                        inputs = processor(
                            images=image,
                            text=text_prompt,
                            return_tensors="pt"
                        )
                    except:
                        inputs = processor(images=image, return_tensors="pt")
            else:
                # Manual processing
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                pixel_values = transform(image).unsqueeze(0)
                input_ids = tokenizer(text_prompt, return_tensors="pt").input_ids
                
                inputs = {
                    "pixel_values": pixel_values,
                    "input_ids": input_ids
                }
            
            # Move to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Run inference
            start_time = time.time()
            
            with torch.no_grad():
                try:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
                    )
                    
                    if hasattr(outputs, 'sequences'):
                        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                    else:
                        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                except Exception as e:
                    print(f"     Generation failed for {pizza_class}: {e}")
                    # Try forward pass
                    outputs = model(**inputs)
                    if hasattr(outputs, 'logits'):
                        predicted_class = torch.argmax(outputs.logits, dim=-1)
                        generated_text = f"Predicted class: {predicted_class.item()}"
                    else:
                        generated_text = f"Forward pass output keys: {list(outputs.keys())}"
            
            inference_time = time.time() - start_time
            
            classification_results.append({
                "class": pizza_class,
                "image_path": sample_image,
                "inference_time": inference_time,
                "output": generated_text,
                "prompt": text_prompt
            })
            
            print(f"     ✅ {pizza_class}: {inference_time:.2f}s")
            print(f"     Output: {generated_text[:80]}...")
        
        print(f"✅ Pizza classification test completed")
        print(f"   Tested {len(classification_results)} pizza classes")
        
        return True, classification_results
        
    except Exception as e:
        print(f"❌ Pizza classification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_image_processing():
    """Test image processing capabilities"""
    print(f"\n🔍 Testing image processing...")
    
    # Create a test image
    test_image = Image.new('RGB', (224, 224), color='red')
    
    try:
        # Test basic image operations
        test_image_resized = test_image.resize((384, 384))
        test_image_array = np.array(test_image)
        
        print(f"✅ Image processing works")
        print(f"   Original size: {test_image.size}")
        print(f"   Resized: {test_image_resized.size}")
        print(f"   Array shape: {test_image_array.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing failed: {e}")
        return False

def save_test_results(results, output_dir):
    """Save test results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "spatial_mllm_test_results.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {results_file}")
    return results_file

def main():
    parser = argparse.ArgumentParser(description="Test Spatial-MLLM setup and availability")
    parser.add_argument("--output-dir", default="output/spatial_mllm",
                      help="Directory to save test results")
    parser.add_argument("--device", default="auto",
                      help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--skip-model-loading", action="store_true",
                      help="Skip actual model loading (for quick tests)")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("🚀 SPATIAL-MLLM INTEGRATION TEST")
    print("="*50)
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    print(f"Skip model loading: {args.skip_model_loading}")
    
    # Initialize results
    results = {
        "test_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "python_version": sys.version,
        },
        "tests": {},
        "overall_success": False
    }
    
    success_count = 0
    total_tests = 6 if not args.skip_model_loading else 3
    
    # Test 1: Basic imports
    import_success = test_basic_imports()
    results["tests"]["basic_imports"] = {
        "success": import_success,
        "description": "Test basic dependency imports"
    }
    if import_success:
        success_count += 1
    
    # Test 2: Model availability
    model_available, model_name = test_model_availability()
    results["tests"]["model_availability"] = {
        "success": model_available,
        "model_name": model_name,
        "description": "Test if model is available on Hugging Face"
    }
    if model_available:
        success_count += 1
    
    # Test 3: Image processing
    image_success = test_image_processing()
    results["tests"]["image_processing"] = {
        "success": image_success,
        "description": "Test basic image processing capabilities"
    }
    if image_success:
        success_count += 1
    
    if not args.skip_model_loading and model_available:
        # Test 4: Model loading
        model, tokenizer, processor, load_time = test_model_loading(model_name, device)
        model_load_success = model is not None
        results["tests"]["model_loading"] = {
            "success": model_load_success,
            "load_time": load_time,
            "description": "Test actual model loading"
        }
        if model_load_success:
            success_count += 1
        
        # Test 5: Vision-language inference
        if model_load_success:
            inference_success, inference_time, output_text = test_vision_language_inference(
                model, tokenizer, processor, device
            )
            results["tests"]["vision_language_inference"] = {
                "success": inference_success,
                "inference_time": inference_time,
                "output_sample": output_text[:100] if output_text else None,
                "description": "Test vision-language model inference with pizza image"
            }
            if inference_success:
                success_count += 1
        
        # Test 6: Pizza classification on multiple classes
        if model_load_success:
            pizza_success, pizza_results = test_pizza_classification(
                model, tokenizer, processor, device
            )
            results["tests"]["pizza_classification"] = {
                "success": pizza_success,
                "results": pizza_results,
                "description": "Test pizza classification on different pizza types"
            }
            if pizza_success:
                success_count += 1
    
    # Calculate overall success
    results["overall_success"] = success_count == total_tests
    results["success_rate"] = success_count / total_tests
    
    # Print summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    print(f"Tests passed: {success_count}/{total_tests}")
    print(f"Success rate: {results['success_rate']*100:.1f}%")
    print(f"Overall success: {'✅' if results['overall_success'] else '❌'}")
    
    # Save results
    save_test_results(results, args.output_dir)
    
    if results["overall_success"]:
        print("\n🎉 All tests passed! Spatial-MLLM is ready for integration.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the results for details.")
        return 1

if __name__ == "__main__":
    exit(main())
