"""Utility functions for reproducible training and plotting."""

import json
import os
import random
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # More reproducible, but sometimes slower on GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def denormalize_image(tensor: torch.Tensor, mean=None, std=None) -> torch.Tensor:
    """Undo normalization for visualization.

    Input tensor shape is expected to be [C, H, W].
    """
    if mean is None:
        mean = DEFAULT_MEAN
    if std is None:
        std = DEFAULT_STD

    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)

    image = tensor.detach().cpu() * std_tensor + mean_tensor
    return torch.clamp(image, 0.0, 1.0)


def plot_training_curves(history: Dict[str, list], output_dir: str) -> None:
    """Save loss and accuracy curves."""
    ensure_dir(output_dir)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], marker="o", label="Train accuracy")
    plt.plot(epochs, history["val_acc"], marker="o", label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"), dpi=200)
    plt.close()


def make_json_serializable(obj: Any) -> Any:
    """Convert numpy and torch objects into JSON-friendly Python objects."""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(value) for value in obj]
    if isinstance(obj, tuple):
        return [make_json_serializable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save dictionary as a readable JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as file:
        json.dump(make_json_serializable(data), file, indent=2)
