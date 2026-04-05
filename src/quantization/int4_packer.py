import numpy as np
import logging

logger = logging.getLogger(__name__)

class Int4Packer:
    """
    Quantizes float32 weights to 4-bit signed integers and packs them.
    Mathematical basis: Laplacian weight distribution clipping.
    """
    def __init__(self, target_range=(-8, 7)):
        self.min_val, self.max_val = target_range

    def quantize_to_int4(self, weights):
        """
        Quantizes weights to 4-bit signed range [-8, 7].
        Uses symmetric clipping for better LSB stability (2-adic theory).
        """
        # Find optimal scale using 3*sigma (Laplacian assumption)
        std = np.std(weights)
        scale = (3 * std) / 7.0
        
        # Clip and quantize
        q_weights = np.clip(np.round(weights / scale), self.min_val, self.max_val).astype(np.int8)
        return q_weights, scale

    def pack_weights(self, q_weights):
        """
        Packs two 4-bit weights into one 8-bit byte.
        Layout: [W2 (high 4 bits) | W1 (low 4 bits)]
        """
        # Ensure even length
        if len(q_weights) % 2 != 0:
            q_weights = np.append(q_weights, 0)
            
        # Shift weights into unsigned 4-bit range for packing (0-15)
        u4_weights = (q_weights + 8).astype(np.uint8)
        
        packed = []
        for i in range(0, len(u4_weights), 2):
            w1 = u4_weights[i] & 0x0F
            w2 = u4_weights[i+1] & 0x0F
            packed_byte = (w2 << 4) | w1
            packed.append(packed_byte)
            
        return np.array(packed, dtype=np.uint8)

if __name__ == "__main__":
    # Test execution
    weights = np.random.randn(100).astype(np.float32)
    packer = Int4Packer()
    q_weights, scale = packer.quantize_to_int4(weights)
    packed = packer.pack_weights(q_weights)
    
    print(f"Original elements: {len(weights)} (float32, ~400 bytes)")
    print(f"Packed bytes: {len(packed)} (int4, 50 bytes)")
    print(f"Compression ratio: {len(weights)*4 / len(packed):.1f}x")
    print(f"Scale factor for dequantization: {scale:.6f}")
