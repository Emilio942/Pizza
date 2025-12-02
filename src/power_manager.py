"""
Compatibility module exposing power manager classes at `src.power_manager`.
"""

from .emulation.simple_power_manager import PowerManager, PowerUsage, AdaptiveMode

__all__ = [
    "PowerManager",
    "PowerUsage",
    "AdaptiveMode",
]

