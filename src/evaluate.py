"""Evaluation metrics and result visualizations."""

import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import denormalize_image, ensure_dir


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect true labels, predicted labels, and class probabilities."""
    model.eval()

    all_true = []
    all_pred = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_true.extend(labels.numpy())
            all_pred.extend(predictions.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    return np.array(all_true), np.array(all_pred), np.array(all_probs)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list,
) -> Dict[str, object]:
    """Evaluate model on a dataloader and return metrics."""
    y_true, y_pred, y_probs = collect_predictions(model, dataloader, device)
    labels = list(range(len(class_names)))

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    print("Test results:")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall:    {recall:.4f}")
    print(f"Macro F1-score:  {f1:.4f}")
    print("\nClassification report:")
    print(report_text)

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "classification_report_text": report_text,
        "classification_report": report_dict,
        "confusion_matrix": matrix,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
    }


def save_confusion_matrix_plot(matrix: np.ndarray, class_names: list, output_path: str) -> None:
    """Save confusion matrix plot."""
    ensure_dir(os.path.dirname(output_path))

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix)
    fig.colorbar(image, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_sample_predictions_plot(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: list,
    device: torch.device,
    output_path: str,
    max_images: int = 9,
) -> None:
    """Save a grid with sample predictions from the test set."""
    ensure_dir(os.path.dirname(output_path))
    model.eval()

    images_to_plot = []
    true_labels = []
    predicted_labels = []
    predicted_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images.to(device))
            probabilities = torch.softmax(outputs, dim=1).cpu()
            predictions = torch.argmax(probabilities, dim=1)

            for image, true_label, pred_label, probs in zip(images, labels, predictions, probabilities):
                images_to_plot.append(image)
                true_labels.append(int(true_label))
                predicted_labels.append(int(pred_label))
                predicted_probs.append(float(probs[pred_label]))

                if len(images_to_plot) >= max_images:
                    break

            if len(images_to_plot) >= max_images:
                break

    if not images_to_plot:
        print("WARNING: No images available for sample prediction plot.")
        return

    cols = 3
    rows = int(np.ceil(len(images_to_plot) / cols))
    _, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, image, true_label, pred_label, prob in zip(
        axes,
        images_to_plot,
        true_labels,
        predicted_labels,
        predicted_probs,
    ):
        image = denormalize_image(image).permute(1, 2, 0).numpy()
        ax.imshow(image)
        ax.set_title(
            f"True: {class_names[true_label]}\n"
            f"Pred: {class_names[pred_label]} ({prob:.2f})"
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
