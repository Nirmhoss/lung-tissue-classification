"""Single-image prediction script for the trained lung CNN model."""

import argparse

import torch
from PIL import Image

from src.dataset import get_transforms
from src.model_factory import AVAILABLE_MODELS, get_model
from src.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict one lung biopsy image using a trained CNN checkpoint."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to one image file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/best_model_baseline.pt",
        help="Path to trained checkpoint.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=None,
        help="Optional image size. If omitted, uses the value stored in checkpoint.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        choices=AVAILABLE_MODELS,
        help="Optional model architecture. If omitted, uses the value stored in checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint format is not supported. Expected a dictionary with "
            "'model_state_dict', 'class_names', and 'img_size'."
        )

    class_names = checkpoint["class_names"]
    img_size = args.img_size if args.img_size is not None else checkpoint["img_size"]
    model_name = (
        args.model_name
        if args.model_name is not None
        else checkpoint.get("model_name", "baseline")
    )

    model = get_model(
        model_name=model_name,
        num_classes=len(class_names),
        img_size=img_size,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, eval_transform = get_transforms(img_size)

    with Image.open(args.image_path) as image:
        image_tensor = eval_transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()
        predicted_index = int(torch.argmax(probabilities).item())

    print(f"Model architecture: {model_name}")
    print(f"Predicted class: {class_names[predicted_index]}")
    print("Class probabilities:")
    for class_name, probability in zip(class_names, probabilities):
        print(f"  {class_name}: {float(probability):.4f}")

    print(
        "\nWarning: this prediction is produced by an educational model, "
        "not by a clinical diagnostic system."
    )


if __name__ == "__main__":
    main()
