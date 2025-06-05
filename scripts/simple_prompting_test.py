#!/usr/bin/env python3
"""
Simple test of different prompting strategies for Spatial-MLLM
Testing one image with multiple prompting approaches
"""

import os
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor

def main():
    print("🍕 Simple Prompting Strategy Test")
    print("=" * 40)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Diankun/Spatial-MLLM-subset-sft"
    
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    
    # Load model
    print("\n🔄 Loading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto" if device != "cpu" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test image path
    test_image_dir = "/home/emilio/Documents/ai/pizza/data/test/basic"
    if not os.path.exists(test_image_dir):
        print(f"❌ Test directory not found: {test_image_dir}")
        return
    
    # Get first available image
    image_files = [f for f in os.listdir(test_image_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"❌ No images found in {test_image_dir}")
        return
    
    test_image = os.path.join(test_image_dir, image_files[0])
    print(f"\n🖼️  Test image: {image_files[0]}")
    
    # Load image
    image = Image.open(test_image).convert('RGB')
    
    # Test 1: Original multiple choice (biased)
    print("\n1️⃣  Original Multiple Choice (A/B/C/D)...")
    try:
        prompt1 = """Look at this pizza image and analyze its cooking state.

Which cooking state best describes this pizza?

A) basic - Raw or undercooked pizza with pale dough and unmelted cheese
B) burnt - Overcooked pizza with dark/black areas and charred edges  
C) mixed - Pizza with mixed cooking levels (some areas cooked, others not)
D) ready - Perfectly cooked pizza with golden-brown crust and melted cheese

Answer:"""
        
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt1}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, temperature=1.0, pad_token_id=tokenizer.eos_token_id)
        
        if hasattr(outputs, 'sequences'):
            full_response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        else:
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "assistant\n" in full_response:
            response = full_response.split("assistant\n")[-1].strip()
        else:
            response = full_response.strip()
        
        print(f"   Response: {response}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Open-ended question
    print("\n2️⃣  Open-ended Classification...")
    try:
        prompt2 = """Look at this pizza image. Describe its cooking state in one word. Is it: basic, ready, burnt, or mixed?"""
        
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt2}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=15, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        
        if hasattr(outputs, 'sequences'):
            full_response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        else:
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "assistant\n" in full_response:
            response = full_response.split("assistant\n")[-1].strip()
        else:
            response = full_response.strip()
        
        print(f"   Response: {response}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Yes/No question  
    print("\n3️⃣  Binary Classification (Is it basic?)...")
    try:
        prompt3 = """Look at this pizza image. Is this pizza undercooked with pale dough and unmelted cheese? Answer YES or NO."""
        
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt3}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, temperature=1.0, pad_token_id=tokenizer.eos_token_id)
        
        if hasattr(outputs, 'sequences'):
            full_response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        else:
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "assistant\n" in full_response:
            response = full_response.split("assistant\n")[-1].strip()
        else:
            response = full_response.strip()
        
        print(f"   Response: {response}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    main()
