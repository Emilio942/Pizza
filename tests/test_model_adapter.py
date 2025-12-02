"""Unit tests for the early-exit model adapter."""

from pathlib import Path

import pytest
import torch

from scripts.early_exit.model_adapter import (
    load_model_with_compatibility,
    ImprovedMicroPizzaNetWithEarlyExit,
    MicroPizzaNetWithEarlyExit,
)


@pytest.fixture
def tmp_weights_dir(tmp_path: Path) -> Path:
    """Temporary directory to store serialized model weights."""
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    return weights_dir


def _save_model_state(model: torch.nn.Module, path: Path) -> Path:
    torch.save(model.state_dict(), path)
    return path


def test_load_model_prefers_improved_architecture(tmp_weights_dir: Path) -> None:
    """Loading an improved model should return the improved architecture."""

    model = ImprovedMicroPizzaNetWithEarlyExit(num_classes=6)
    weights_path = _save_model_state(model, tmp_weights_dir / "improved.pt")

    loaded = load_model_with_compatibility(str(weights_path), num_classes=6, device="cpu")

    assert isinstance(loaded, ImprovedMicroPizzaNetWithEarlyExit)


def test_load_model_falls_back_to_original(tmp_weights_dir: Path) -> None:
    """When improved loading fails, the adapter should fall back to the original model."""

    original_model = MicroPizzaNetWithEarlyExit(num_classes=6)
    weights_path = _save_model_state(original_model, tmp_weights_dir / "original.pt")

    loaded = load_model_with_compatibility(str(weights_path), num_classes=6, device="cpu")

    assert isinstance(loaded, MicroPizzaNetWithEarlyExit)
