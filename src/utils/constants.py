"""
Compatibility proxy for constants.

This module re-exports all constants from `src/constants.py` to avoid
duplicated and conflicting definitions across the codebase. Importing
from `src.utils.constants` will now return the same values as
`src.constants`.
"""

# Re-export everything from the canonical constants module
from ..constants import *  # noqa: F401,F403
