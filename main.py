"""Main training script for the lung biopsy CNN project."""

import argparse
import os

import torch
import torch.nn as nn

from src.dataset import (
    create_data_loaders,
    print_class_distributions,
    save_class_distribution_csv,
    save_sample_images,
)
from src.evaluate import (
    evaluate_model,
    save_confusion_matrix_plot,
    save_sample_predictions_plot,
)
from src.model_factory import AVAILABLE_MODELS, get_model, print_model_shape_explanation
from src.train import train_model
from src.utils import ensure_dir, get_device, plot_training_curves, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a custom CNN for lung biopsy image classification."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/lung_subset_small",
        help="Path to dataset root folder used by ImageFolder.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for DataLoaders.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=128,
        help="Image size after resizing. Images become img_size x img_size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers. Use 0 for maximum compatibility.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="baseline",
        choices=AVAILABLE_MODELS,
        help="CNN architecture to train: baseline, teammate, or improved.",
    )
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        help="Evaluate the saved checkpoint without training; preserve historical reports.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = get_device()

    figures_dir = os.path.join("reports", "figures", args.model_name)
    models_dir = "models"
    model_path = os.path.join(models_dir, f"best_model_{args.model_name}.pt")
    metrics_path = os.path.join("reports", f"test_metrics_{args.model_name}.json")
    distribution_path = os.path.join("reports", "class_distribution.csv")

    if args.evaluate_only:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if checkpoint.get("model_name", "baseline") != args.model_name:
            raise ValueError("Checkpoint architecture does not match --model_name.")
        # Reconstruct the historical split and preprocessing from the checkpoint.
        args.seed = checkpoint["seed"]
        args.img_size = checkpoint["img_size"]
        metrics_path = os.path.join(
            "reports", f"reevaluated_metrics_{args.model_name}.json"
        )

    set_seed(args.seed)

    ensure_dir(figures_dir)
    ensure_dir(models_dir)
    ensure_dir("reports")

    print(f"Using device: {device}")
    print(f"Dataset directory: {args.data_dir}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Seed: {args.seed}")
    print(f"Model architecture: {args.model_name}\n")

    dataloaders, metadata = create_data_loaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    class_names = metadata["class_names"]
    class_to_idx = metadata["class_to_idx"]
    distributions = metadata["distributions"]

    print("class_to_idx:")
    print(class_to_idx)

    print_class_distributions(distributions)
    if not args.evaluate_only:
        save_class_distribution_csv(distributions, distribution_path)
        save_sample_images(
            dataset=dataloaders["val"].dataset,
            class_names=class_names,
            output_path=os.path.join(figures_dir, "sample_images.png"),
        )

    model = get_model(
        model_name=args.model_name,
        num_classes=len(class_names),
        img_size=args.img_size,
    ).to(device)
    print(model)
    print_model_shape_explanation(model, args.img_size, len(class_names))

    if args.evaluate_only:
        if (
            checkpoint["class_names"] != class_names
            or checkpoint["class_to_idx"] != class_to_idx
        ):
            raise ValueError("Dataset class labels do not match the checkpoint.")
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate_model(
            model=model,
            dataloader=dataloaders["test"],
            device=device,
            class_names=class_names,
        )
        save_json(
            {
                "checkpoint": model_path,
                "seed": args.seed,
                "img_size": args.img_size,
                "class_to_idx": class_to_idx,
                "class_distribution": distributions,
                "test_metrics": test_metrics,
            },
            metrics_path,
        )
        print(f"Re-evaluation metrics: {metrics_path}")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = train_model(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        model_path=model_path,
        class_names=class_names,
        class_to_idx=class_to_idx,
        img_size=args.img_size,
        seed=args.seed,
        model_name=args.model_name,
    )

    plot_training_curves(history, figures_dir)

    # Load the best validation checkpoint before final test evaluation.
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_model(
        model=model,
        dataloader=dataloaders["test"],
        device=device,
        class_names=class_names,
    )

    save_confusion_matrix_plot(
        matrix=test_metrics["confusion_matrix"],
        class_names=class_names,
        output_path=os.path.join(figures_dir, "confusion_matrix.png"),
    )

    save_sample_predictions_plot(
        model=model,
        dataloader=dataloaders["test"],
        class_names=class_names,
        device=device,
        output_path=os.path.join(figures_dir, "sample_predictions.png"),
    )

    save_json(
        {
            "args": vars(args),
            "class_to_idx": class_to_idx,
            "class_distribution": distributions,
            "history": history,
            "test_metrics": test_metrics,
            "educational_warning": (
                "This model is for educational purposes only and is not a clinical "
                "diagnostic system."
            ),
        },
        metrics_path,
    )

    print("Saved outputs:")
    print(f"Best model:             {model_path}")
    print(f"Metrics JSON:           {metrics_path}")
    print(f"Class distribution CSV: {distribution_path}")
    print(f"Figures directory:      {figures_dir}")


if __name__ == "__main__":
    main()
