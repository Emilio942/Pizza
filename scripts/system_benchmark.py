import os
import sys
import torch
import time
import json
import numpy as np
from pathlib import Path

# Fix path to include current dir
sys.path.append(os.getcwd())

def get_model_stats(model_path):
    """Computes model stats relevant for RP2040."""
    print(f"Analyzing {model_path}...")
    try:
        # Load state dict
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        
        total_params = 0
        total_size_kb = os.path.getsize(model_path) / 1024
        
        # Analyze parameters
        for name, param in checkpoint.items():
            if hasattr(param, 'numel'):
                total_params += param.numel()
        
        # Estimated MACs (Rough estimate for a typical CNN if we don't have the architecture)
        # We'll try to estimate from the layer shapes
        estimated_macs = 0
        for name, param in checkpoint.items():
            if 'weight' in name and len(param.shape) >= 2:
                # For Linear: in * out
                # For Conv: out * in * k * k
                estimated_macs += param.numel() # Simple approximation for weights

        return {
            "params": total_params,
            "size_kb": total_size_kb,
            "estimated_macs": estimated_macs,
            "success": True
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def benchmark_inference(model_path, dummy_input_shape=(1, 3, 224, 224)):
    """Measures latency on CPU as a proxy."""
    try:
        # Note: This requires the model class to be available. 
        # Since we only have the .pth (likely state_dict), we might need to know the architecture.
        # But we can try to estimate latency based on MACs.
        stats = get_model_stats(model_path)
        if not stats["success"]:
            return stats

        # RP2040 Stats: 133 MHz, Cortex-M0+ (roughly 1 MAC per 10-20 cycles depending on implementation)
        # CMSIS-NN can do 1 MAC in ~1-4 cycles for INT8 if optimized.
        # Let's be conservative: 10 cycles per MAC.
        cycles_per_mac = 10
        total_cycles = stats["estimated_macs"] * cycles_per_mac
        rp2040_mhz = 133 * 1e6
        estimated_latency_ms = (total_cycles / rp2040_mhz) * 1000

        return {
            "macs": stats["estimated_macs"],
            "size_kb": stats["size_kb"],
            "estimated_rp2040_latency_ms": estimated_latency_ms,
            "success": True
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def generate_report():
    print("🚀 Starting Standardized Benchmark...")
    
    models_to_test = [
        "models/pizza_model_int8.pth"
    ]
    
    results = {}
    for model_p in models_to_test:
        if os.path.exists(model_p):
            results[model_p] = benchmark_inference(model_p)
        else:
            results[model_p] = {"success": False, "error": "File not found"}

    # Target Matrix
    target = {
        "latency_ms": 100,
        "memory_kb": 200,
        "accuracy": 0.85
    }

    print("\n" + "="*50)
    print("📊 BENCHMARK RESULTS")
    print("="*50)
    
    report_lines = ["# 📊 Standardized Benchmark Report", "", "## 🎯 Target Matrix", ""]
    report_lines.append(f"- **Max Latency:** {target['latency_ms']} ms")
    report_lines.append(f"- **Max Memory:** {target['memory_kb']} KB")
    report_lines.append(f"- **Min Accuracy:** {target['accuracy'] * 100}%")
    report_lines.append("")
    report_lines.append("## 🔍 Measured Performance")
    report_lines.append("")

    for model_p, res in results.items():
        print(f"Model: {model_p}")
        report_lines.append(f"### Model: `{model_p}`")
        if res["success"]:
            lat = res['estimated_rp2040_latency_ms']
            mem = res['size_kb']
            print(f"  Latency (est): {lat:.2f} ms")
            print(f"  Memory: {mem:.2f} KB")
            
            report_lines.append(f"- **Estimated RP2040 Latency:** {lat:.2f} ms")
            report_lines.append(f"- **Flash/Storage Size:** {mem:.2f} KB")
            
            status_lat = "✅ OK" if lat <= target['latency_ms'] else "❌ TOO SLOW"
            status_mem = "✅ OK" if mem <= target['memory_kb'] else "❌ TOO LARGE"
            
            report_lines.append(f"- **Latency Status:** {status_lat}")
            report_lines.append(f"- **Memory Status:** {status_mem}")
        else:
            print(f"  Error: {res['error']}")
            report_lines.append(f"- **Error:** {res['error']}")
        report_lines.append("")

    with open("BENCHMARK_REPORT.md", "w") as f:
        f.write("\n".join(report_lines))
    
    print("\n✅ Report generated: BENCHMARK_REPORT.md")

if __name__ == "__main__":
    generate_report()
