# Lung Tissue Image Classification with CNNs

## Overview

This university group project classifies RGB lung tissue images into three classes using a compact CNN implemented in PyTorch. The baseline was trained and evaluated alongside two alternative CNN architectures. The saved results describe the original student experiment, including the duplicate-image leakage found during a publication audit.

## Dataset

The original README identifies the source as a Google Drive folder supplied for the course. No download link, upstream dataset attribution, or license is provided, so those details could not be verified. Images are excluded from this repository. Obtain the course-provided `lung_subset_small` folder from the course provider; the exact subset is needed to reproduce the historical split.

All 2,400 local images were verified as RGB JPEGs measuring 128 × 128 pixels:

| Class | Images | Train | Validation | Test |
|---|---:|---:|---:|---:|
| `lung_aca` (adenocarcinoma) | 800 | 560 | 120 | 120 |
| `lung_n` (normal tissue) | 800 | 560 | 120 | 120 |
| `lung_scc` (squamous cell carcinoma) | 800 | 560 | 120 | 120 |
| Total | 2,400 | 1,680 | 360 | 360 |

`ImageFolder` assigns labels as `lung_aca=0`, `lung_n=1`, `lung_scc=2`. The code makes a stratified 70%/15%/15% split in two stages with seed 42. A warned random-split fallback exists for datasets too small for stratification; it was not used here.

**Duplicate-image leakage:** 22 image pairs are identical both as files and as decoded pixels. Eleven pairs cross split boundaries: six train/validation, four train/test, and one validation/test. The original split is preserved to keep historical results traceable. [Duplicate image evidence](reports/duplicate_images.json) records the filenames, hashes, and split membership.

## Model

The baseline `SimpleLungCNN` has **23,779 trainable parameters**. Three 3 × 3 convolution layers with 16, 32, and 64 channels are each followed by ReLU and 2 × 2 max pooling. Global average pooling, dropout with probability 0.3, and a linear layer produce three logits. No pretrained weights are used.

The existing `teammate` comparison uses 5 × 5 convolutions and a larger fully connected classifier. The `improved` comparison uses two convolutions per block, 32/64/128 channels, batch normalization, global average pooling, and dropout with probability 0.4.

## Training

The original runs used 25 epochs, batch size 32, Adam with learning rate 0.001, and cross-entropy loss. Training images are resized to 128 × 128, randomly flipped horizontally with probability 0.5, and randomly rotated by up to 10 degrees. Tensors use normalization mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

Validation, test, and prediction use the same resizing and normalization without augmentation. The existing seed setup covers Python, NumPy, and PyTorch, with deterministic cuDNN settings and a seeded training-loader generator. Exact retraining across hardware or package versions is not established. There is no early stopping; the checkpoint with the lowest validation loss is saved.

## Evaluation

The best validation checkpoint is loaded and evaluated on the 360-image test split. Scikit-learn computes accuracy, macro precision, macro recall, macro F1, per-class metrics, and a confusion matrix. Macro F1 averages the three class F1 scores.

The test loader is not passed to training or checkpoint selection. The files cannot establish whether test results influenced manual choices between experiments. During the audit, all three original checkpoints reproduced their saved test labels, predictions, confusion matrices, and aggregate metrics. This verifies the calculations but does not remove duplicate-image leakage.

## Results

These are **historical results from the original split, which contains duplicate-image leakage**. They are not estimates from a leakage-free evaluation.

| Model | Trainable parameters | Test accuracy | Macro F1 | Best epoch |
|---|---:|---:|---:|---:|
| Baseline | 23,779 | 91.39% | 91.44% | 24 |
| Teammate | 2,046,571 | 94.17% | 94.14% | 22 |
| Improved | 287,843 | 97.50% | 97.49% | 24 |

The baseline correctly classified 329/360 test images: accuracy `0.9138888888888889`, macro F1 `0.9143646646023487`. Original metrics, predictions, probabilities, and training histories are preserved in `reports/test_metrics_<model_name>.json`.

![Baseline confusion matrix from the original split](reports/figures/baseline/confusion_matrix.png)

![Baseline training and validation loss](reports/figures/baseline/loss_curve.png)

## Project Structure

```text
lung-tissue-classification/
├── .gitignore
├── main.py
├── predict.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── model_factory.py
│   ├── model_improved.py
│   ├── model_teammate.py
│   ├── train.py
│   └── utils.py
├── models/
│   └── best_model_baseline.pt
└── reports/
    ├── class_distribution.csv
    ├── duplicate_images.json
    ├── test_metrics_baseline.json
    ├── test_metrics_improved.json
    ├── test_metrics_teammate.json
    └── figures/
        ├── baseline/
        ├── improved/
        └── teammate/
```

Each figure directory contains `accuracy_curve.png`, `loss_curve.png`, and `confusion_matrix.png`. Figures containing dataset images are excluded because redistribution rights are unknown.

The baseline checkpoint is included because it is only 100,469 bytes and permits inference without retraining. Alternative checkpoints (1,165,168 and 8,192,937 bytes) are excluded to keep the repository small; training generates them locally. Their original reports are retained.

## Running the Project

Run commands from the repository root.

1. Clone and enter the repository:

   ```bash
   git clone https://github.com/Nirmhoss/lung-tissue-classification.git
   cd lung-tissue-classification
   ```

2. Create an environment:

   ```bash
   python -m venv .venv
   ```

   Activate it with `.\.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on macOS/Linux, then install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

   Validation used Python 3.14.0 and these existing environment versions: PyTorch `2.12.0+cu126`, torchvision `0.27.0+cu126`, NumPy `2.4.6`, pandas `3.0.3`, scikit-learn `1.9.0`, matplotlib `3.10.9`, Pillow `12.2.0`. Requirements pin the tested base versions; CPU/CUDA wheels depend on the installation source. A fresh installation and other Python versions were not tested.

3. Obtain the course dataset and place its class folders here:

   ```text
   data/lung_subset_small/
   ├── lung_aca/
   ├── lung_n/
   └── lung_scc/
   ```

4. Re-evaluate the included checkpoint without training:

   ```bash
   python main.py --evaluate_only
   ```

   This uses the checkpoint's seed and image size, checks class labels, and writes `reports/reevaluated_metrics_baseline.json`, preserving original reports and plots. The same dataset files and ordering are required; the checkpoint has no dataset manifest.

5. Predict one image, replacing the placeholder with an existing image path:

   ```bash
   python predict.py --image_path path/to/image.jpeg
   ```

6. Repeat the original training procedure if needed:

   ```bash
   python main.py --data_dir data/lung_subset_small --epochs 25 --batch_size 32 --lr 0.001 --img_size 128 --seed 42 --model_name baseline
   ```

   This repeats the original split, including duplicate leakage, and overwrites that model's local checkpoint, metrics, and figures. Use `--model_name teammate` or `--model_name improved` for comparisons. `--data_dir` accepts a different dataset location. Both scripts support `--help`.

## Development Note

AI coding tools were used extensively during implementation. The resulting code was reviewed, tested, adapted, and evaluated by me as part of the project.

The original README lists the group members as Saienko Mykhailo, Muzychenko Oleksandr, and Nikita Skydonenko. Individual contributions are not documented in the project files.

## Limitations

- A clean rerun must remove duplicates or keep duplicate groups within a single split, then retrain and evaluate anew. Historical scores have not been replaced with invented corrected results.
- No patient, slide, or original-image grouping metadata is available. Overlap among related images or augmented variants cannot be ruled out through the exact-duplicate check.
- No external validation is present. The subset's upstream source and redistribution terms remain unverified.
- This educational model is not a clinical diagnostic system.
