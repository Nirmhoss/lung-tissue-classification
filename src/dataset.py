"""Dataset loading, transforms, stratified splitting, and visualization."""

import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.utils import DEFAULT_MEAN, DEFAULT_STD, denormalize_image, ensure_dir


def get_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return train and evaluation transforms.

    Train transforms include light augmentation. Validation/test transforms are
    deterministic because evaluation should not be random.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
    ])

    return train_transform, eval_transform


def get_labels_from_imagefolder(dataset: ImageFolder) -> np.ndarray:
    """Extract integer labels from an ImageFolder dataset."""
    return np.array([label for _, label in dataset.samples])


def stratified_train_val_test_split(
    labels: np.ndarray,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create train/validation/test indices with stratification if possible."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    indices = np.arange(len(labels))

    try:
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=seed,
            stratify=labels,
        )

        train_val_labels = labels[train_val_idx]
        relative_val_ratio = val_ratio / (train_ratio + val_ratio)

        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val_ratio,
            random_state=seed,
            stratify=train_val_labels,
        )

    except ValueError as error:
        print("WARNING: Stratified split failed. Falling back to random split.")
        print(f"Reason: {error}")

        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(indices)

        train_end = int(train_ratio * len(shuffled))
        val_end = train_end + int(val_ratio * len(shuffled))

        train_idx = shuffled[:train_end]
        val_idx = shuffled[train_end:val_end]
        test_idx = shuffled[val_end:]

    return train_idx, val_idx, test_idx


def class_distribution_from_indices(
    labels: np.ndarray,
    indices: np.ndarray,
    class_names: list,
) -> Dict[str, int]:
    """Count class samples for a given set of indices."""
    selected_labels = labels[indices]
    counts = {}
    for class_index, class_name in enumerate(class_names):
        counts[class_name] = int(np.sum(selected_labels == class_index))
    return counts


def print_class_distributions(distributions: Dict[str, Dict[str, int]]) -> None:
    """Print class distributions as a small table."""
    table = pd.DataFrame(distributions).fillna(0).astype(int)
    table.loc["TOTAL"] = table.sum(axis=0)
    print("\nClass distribution:")
    print(table)
    print()


def create_data_loaders(
    data_dir: str,
    img_size: int,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> Tuple[Dict[str, DataLoader], Dict[str, object]]:
    """Create ImageFolder datasets, stratified splits, and DataLoaders."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            "Expected structure: data/lung_subset_small/lung_aca, "
            "data/lung_subset_small/lung_n, data/lung_subset_small/lung_scc"
        )

    train_transform, eval_transform = get_transforms(img_size)

    # Dataset without transform is used only to read labels and class names.
    base_dataset = ImageFolder(root=data_dir)
    labels = get_labels_from_imagefolder(base_dataset)
    class_names = base_dataset.classes
    class_to_idx = base_dataset.class_to_idx

    train_idx, val_idx, test_idx = stratified_train_val_test_split(labels, seed=seed)

    # Separate ImageFolder objects allow different transforms for train and eval.
    train_full = ImageFolder(root=data_dir, transform=train_transform)
    eval_full = ImageFolder(root=data_dir, transform=eval_transform)

    train_dataset = Subset(train_full, train_idx)
    val_dataset = Subset(eval_full, val_idx)
    test_dataset = Subset(eval_full, test_idx)

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }

    distributions = {
        "full": class_distribution_from_indices(labels, np.arange(len(labels)), class_names),
        "train": class_distribution_from_indices(labels, train_idx, class_names),
        "val": class_distribution_from_indices(labels, val_idx, class_names),
        "test": class_distribution_from_indices(labels, test_idx, class_names),
    }

    metadata = {
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "labels": labels,
        "indices": {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        },
        "distributions": distributions,
    }

    return dataloaders, metadata


def save_class_distribution_csv(distributions: Dict[str, Dict[str, int]], output_path: str) -> None:
    """Save class distribution table as CSV."""
    ensure_dir(os.path.dirname(output_path))
    table = pd.DataFrame(distributions).fillna(0).astype(int)
    table.loc["TOTAL"] = table.sum(axis=0)
    table.to_csv(output_path)


def save_sample_images(dataset: Subset, class_names: list, output_path: str, max_per_class: int = 3) -> None:
    """Save a grid with a few example images from each class."""
    ensure_dir(os.path.dirname(output_path))

    selected = []
    counts = {class_index: 0 for class_index in range(len(class_names))}

    for image, label in dataset:
        if counts[label] < max_per_class:
            selected.append((image, label))
            counts[label] += 1
        if all(count >= max_per_class for count in counts.values()):
            break

    if not selected:
        print("WARNING: No images available for sample visualization.")
        return

    cols = max_per_class
    rows = len(class_names)
    _, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for ax in axes.ravel():
        ax.axis("off")

    position = {class_index: 0 for class_index in range(len(class_names))}

    for image, label in selected:
        col = position[label]
        ax = axes[label][col]
        image = denormalize_image(image).permute(1, 2, 0).numpy()
        ax.imshow(image)
        ax.set_title(class_names[label])
        ax.axis("off")
        position[label] += 1

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
