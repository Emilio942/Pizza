import torch
import torch.nn as nn
import logging
from src.config import RP2040Config

logger = logging.getLogger(__name__)

class MemoryEstimator:
    """Schätzt Speicherverbrauch von Modellen und Operationen für RP2040"""
    
    @staticmethod
    def estimate_model_size(model, bits=32, custom_bits=None):
        """
        Schätzt die Modellgröße in KB
        
        Args:
            model: PyTorch-Modell
            bits: Standard-Bitbreite für Parameter (8, 16, 32)
            custom_bits: Dictionary mit spezifischen Bitbreiten für bestimmte Layer, z.B.
                        {'int4_layers': 4} für INT4-quantisierte Layer
        """
        param_size = 0
        
        # Wenn spezifische Layer mit eigener Bitbreite angegeben sind
        if custom_bits and 'int4_layers' in custom_bits:
            int4_bit_width = custom_bits.get('int4_layers', 4)
            
            # Für jedes benannte Modul prüfen, ob es ein INT4-Layer ist
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    # Für Gewichte die reduzierte Bitbreite verwenden
                    weight_param = getattr(module, 'weight', None)
                    if weight_param is not None:
                        param_size += weight_param.nelement() * (int4_bit_width / 8)
                    
                    # Für Bias die Standardbitbreite verwenden
                    bias_param = getattr(module, 'bias', None)
                    if bias_param is not None:
                        param_size += bias_param.nelement() * (bits / 8)
                else:
                    # Für alle anderen Module die Standardparameter verwenden
                    for param in module.parameters(recurse=False):
                        param_size += param.nelement() * (bits / 8)
        else:
            # Standardschätzung für alle Parameter
            for param in model.parameters():
                param_size += param.nelement() * (bits / 8)  # Größe in Bytes
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * (bits / 8)
            
        total_size_kb = (param_size + buffer_size) / 1024
        return total_size_kb
    
    @staticmethod
    def estimate_activation_memory(model, input_size):
        """Estimates the maximum memory usage by activations during inference"""
        device = next(model.parameters()).device
        input_tensor = torch.rand(1, *input_size).to(device)  # Batch size 1 for inference
        
        # Track activation sizes per layer and memory usage over time
        activation_sizes = []
        memory_timeline = [0]  # Track memory usage throughout execution
        current_memory = 0     # Current memory usage
        
        # Dictionary to track tensor lifetimes
        tensor_lifetimes = {}
        tensor_counter = 0
        
        def forward_hook(module, input, output):
            nonlocal current_memory, tensor_counter
            
            if isinstance(output, torch.Tensor):
                # Calculate memory for output tensor
                tensor_size = output.nelement() * output.element_size()
                activation_sizes.append(tensor_size)
                
                # Assign ID to this tensor and track its creation
                tensor_id = tensor_counter
                tensor_counter += 1
                tensor_lifetimes[tensor_id] = {
                    'size': tensor_size,
                    'created_at': len(memory_timeline),
                    'freed_at': None
                }
                
                # Add memory for this tensor
                current_memory += tensor_size
                memory_timeline.append(current_memory)
                
            elif isinstance(output, tuple):
                # For multiple outputs
                for o in output:
                    if isinstance(o, torch.Tensor):
                        tensor_size = o.nelement() * o.element_size()
                        activation_sizes.append(tensor_size)
                        
                        # Assign ID and track
                        tensor_id = tensor_counter
                        tensor_counter += 1
                        tensor_lifetimes[tensor_id] = {
                            'size': tensor_size,
                            'created_at': len(memory_timeline),
                            'freed_at': None
                        }
                        
                        # Add memory
                        current_memory += tensor_size
                        memory_timeline.append(current_memory)
        
        # Hooks for input tensors
        input_hooks = []
        
        def input_hook(module, input):
            nonlocal current_memory
            
            for tensor in input:
                if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                    # For simplicity, assume input tensors with gradients
                    # persist until the backward pass
                    tensor_size = tensor.nelement() * tensor.element_size()
                    current_memory += tensor_size
                    memory_timeline.append(current_memory)
        
        # Register hooks for all modules
        forward_hooks = []
        for name, module in model.named_modules():
            if not list(module.children()):  # Only leaf modules
                forward_hooks.append(module.register_forward_hook(forward_hook))
                input_hooks.append(module.register_forward_pre_hook(input_hook))
        
        # Forward pass for hook activation
        with torch.no_grad():
            model(input_tensor)
        
        # Remove hooks
        for hook in forward_hooks:
            hook.remove()
        for hook in input_hooks:
            hook.remove()
        
        # Simulate tensor deallocation based on scope exit
        # In real execution, tensors would be freed at different points
        # This is a simple heuristic that assumes tensors are freed when they're no longer needed
        
        # Simulate lifetime - mark tensors as freed when their operation completes
        # (this is very simplified compared to real memory management)
        layer_depth = 0
        for i in range(1, len(memory_timeline)):
            # Increase depth when memory increases (new tensor created)
            if memory_timeline[i] > memory_timeline[i-1]:
                layer_depth += 1
            
            # Find tensors to free based on simple heuristic
            # In this simplified model, tensors created at depth D are freed 
            # when we go back to depth D-1
            for tensor_id, info in tensor_lifetimes.items():
                if info['freed_at'] is None and info['created_at'] < i and layer_depth < info['created_at']:
                    tensor_lifetimes[tensor_id]['freed_at'] = i
                    current_memory -= info['size']
                    memory_timeline.append(current_memory)
        
        # Calculate maximum memory usage
        max_activation_kb = max(memory_timeline) / 1024
        total_activation_kb = sum(activation_sizes) / 1024
        
        return {
            'total_kb': total_activation_kb,           # Sum of all activations
            'max_activation_kb': max_activation_kb,    # Estimated peak memory usage
            'memory_timeline': memory_timeline,        # Full timeline for analysis
            'largest_layer_kb': max(activation_sizes) / 1024  # Largest single layer
        }

    @staticmethod
    def check_memory_requirements(model, input_size, config):
        """Checks if the model fits within the memory constraints of the RP2040"""
        # Float32 size (for training)
        float32_size_kb = MemoryEstimator.estimate_model_size(model, bits=32)
        
        # Int8 size (for deployment)
        int8_size_kb = MemoryEstimator.estimate_model_size(model, bits=8)
        
        # Activation memory during inference
        activation_memory = MemoryEstimator.estimate_activation_memory(model, input_size)
        
        # Use peak memory for RAM estimate rather than total sum
        peak_runtime_memory_kb = int8_size_kb + activation_memory['max_activation_kb']
        
        # Also calculate the conservative estimate for comparison
        conservative_memory_kb = int8_size_kb + activation_memory['total_kb']
        
        # Check against memory constraints
        flash_ok = int8_size_kb <= config.MAX_MODEL_SIZE_KB
        ram_ok = peak_runtime_memory_kb <= config.MAX_RUNTIME_RAM_KB
        
        report = {
            'model_size_float32_kb': float32_size_kb,
            'model_size_int8_kb': int8_size_kb,
            'activation_memory_kb': activation_memory,
            'peak_runtime_memory_kb': peak_runtime_memory_kb,
            'conservative_runtime_memory_kb': conservative_memory_kb,
            'flash_requirement_met': flash_ok,
            'ram_requirement_met': ram_ok,
            'flash_usage_percent': (int8_size_kb / config.MAX_MODEL_SIZE_KB) * 100,
            'ram_usage_percent': (peak_runtime_memory_kb / config.MAX_RUNTIME_RAM_KB) * 100,
            'ram_usage_percent_conservative': (conservative_memory_kb / config.MAX_RUNTIME_RAM_KB) * 100,
            'total_flash_percent': (int8_size_kb / config.RP2040_FLASH_SIZE_KB) * 100,
            'total_ram_percent': (peak_runtime_memory_kb / config.RP2040_RAM_SIZE_KB) * 100
        }
        
        logger.info(f"Memory Analysis:")
        logger.info(f"  Model size (Float32): {float32_size_kb:.2f} KB")
        logger.info(f"  Model size (Int8): {int8_size_kb:.2f} KB ({report['flash_usage_percent']:.1f}% of allocated flash)")
        logger.info(f"  Peak activation memory: {activation_memory['max_activation_kb']:.2f} KB")
        logger.info(f"  Total activation memory (sum): {activation_memory['total_kb']:.2f} KB")
        logger.info(f"  Peak runtime memory: {peak_runtime_memory_kb:.2f} KB ({report['ram_usage_percent']:.1f}% of allocated RAM)")
        logger.info(f"  Conservative memory estimate: {conservative_memory_kb:.2f} KB ({report['ram_usage_percent_conservative']:.1f}% of allocated RAM)")
        logger.info(f"  Resource usage of total hardware: {report['total_flash_percent']:.1f}% flash, {report['total_ram_percent']:.1f}% RAM")
        
        if not flash_ok:
            logger.warning(f"Warning: Model exceeds flash constraint ({int8_size_kb:.2f}KB > {config.MAX_MODEL_SIZE_KB}KB)")
        
        if not ram_ok:
            logger.warning(f"Warning: Runtime memory exceeds RAM constraint ({peak_runtime_memory_kb:.2f}KB > {config.MAX_RUNTIME_RAM_KB}KB)")
            
        return report
