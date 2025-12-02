"""
Compatibility layer re-exporting exceptions for tests and external imports.

Ensures that tests catching ResourceError/HardwareError see the same classes
raised by the emulator by aliasing to the emulator’s exception types when available.
"""

# Base exports
from .utils.exceptions import *  # type: ignore[F403]

# Prefer the emulator-defined exception classes for identity in tests
try:
    from .emulation.emulator import ResourceError as _EmuResourceError, HardwareError as _EmuHardwareError
    ResourceError = _EmuResourceError  # type: ignore[F405]
    HardwareError = _EmuHardwareError  # type: ignore[F405]
except Exception:
    # Fall back to utils exceptions if emulator not importable
    pass

__all__ = [name for name in dir() if not name.startswith("_")]
