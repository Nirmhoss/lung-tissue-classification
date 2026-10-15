"""Model factory for selecting one of several CNN architectures."""

import torch
import torch.nn as nn

from src.model import SimpleLungCNN
from src.model_improved import ImprovedLungCNN
from src.model_teammate import TeammateCNN


AVAILABLE_MODELS = ["baseline", "teammate", "improved"]


def get_model(model_name: str, num_classes: int = 3, img_size: int = 128) -> nn.Module:
    """Create a CNN model by name."""
    if model_name == "baseline":
        return SimpleLungCNN(num_classes=num_classes)
    if model_name == "teammate":
        return TeammateCNN(num_classes=num_classes, img_size=img_size)
    if model_name == "improved":
        return ImprovedLungCNN(num_classes=num_classes)

    raise ValueError(
        f"Unknown model_name='{model_name}'. Choose one of: {AVAILABLE_MODELS}"
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_shape_explanation(model: nn.Module, img_size: int, num_classes: int) -> None:
    """Print input/output tensor shapes using a dummy forward pass."""
    model_was_training = model.training
    model.eval()

    device = next(model.parameters()).device

    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size, device=device)
        output = model(dummy)

    if model_was_training:
        model.train()

    print("\nModel tensor shapes:")
    print(f"Input batch:   [batch_size, 3, {img_size}, {img_size}]")
    print(f"Output logits: [batch_size, {num_classes}]")
    print(f"Dummy output:  {list(output.shape)}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}\n")
