"""Training and validation loops."""

import os
import time
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import ensure_dir


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple:
    """Train the model for one epoch and return average loss and accuracy."""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    average_loss = running_loss / total
    accuracy = correct / total
    return average_loss, accuracy


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Evaluate the model for one epoch and return average loss and accuracy."""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

    average_loss = running_loss / total
    accuracy = correct / total
    return average_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    model_path: str,
    class_names: list,
    class_to_idx: dict,
    img_size: int,
    seed: int,
    model_name: str = "baseline",
) -> Dict[str, list]:
    """Train model and save the best checkpoint based on validation loss."""
    ensure_dir(os.path.dirname(model_path))

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "class_to_idx": class_to_idx,
                "img_size": img_size,
                "seed": seed,
                "model_name": model_name,
                "best_val_loss": best_val_loss,
                "history": history,
            }
            torch.save(checkpoint, model_path)
            print(f"  Saved new best model to {model_path}")

    total_minutes = (time.time() - start_time) / 60
    print(f"\nTraining finished in {total_minutes:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}\n")

    return history
