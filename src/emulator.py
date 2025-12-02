"""
Compatibility module exposing emulator classes at `src.emulator`.
"""

from .emulation.emulator import CameraEmulator, RP2040Emulator, HardwareError, ResourceError

__all__ = [
    "CameraEmulator",
    "RP2040Emulator",
    "HardwareError",
    "ResourceError",
]

