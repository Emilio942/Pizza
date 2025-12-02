"""Compatibility shim exposing the emulator package at the repository root.

The legacy test suite expects the modules to be importable as ``emulation.*``.
These re-exports forward to the canonical implementations under ``src.emulation``.
"""

from src.emulation.emulator import (
    RP2040Emulator,
    CameraEmulator,
    ResourceError,
    HardwareError,
)
from src.emulation.frame_buffer import FrameBuffer, PixelFormat

__all__ = [
    "RP2040Emulator",
    "CameraEmulator",
    "ResourceError",
    "HardwareError",
    "FrameBuffer",
    "PixelFormat",
]
