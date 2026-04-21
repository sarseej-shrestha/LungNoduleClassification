#!/usr/bin/env python3
"""
Lung Nodule CAD Training V2.0 - Calibrated Framework
With WeightedRandomSampler and Calibration Monitoring (Mean Logit Magnitude)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision import models
from typing import Tuple, Optional, List
import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from src.config_v2 import config_v2
from src.architecture_v2 import create_model_v2, SharedResNet_V2

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

BATCH_SIZE = config_v2.BATCH_SIZE
LEARNING_RATE = config_v2.LEARNING_RATE
WEIGHT_DECAY = config_v2.WEIGHT_DECAY
NUM_EPOCHS = config_v2.NUM_EPOCHS
PATIENCE = config_v2.PATIENCE
MODEL_SAVE_PATH = config_v2.BEST_MODEL_PATH


class NoduleDataset(Dataset):
    def __init__(
        self, npz_dir: str, augment: bool = True, target_label: Optional[int] = None
    ):
        self.npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        self.augment = augment
        self.target_label = target_label

    def __len__(self) -> int:
        return len(self.npz_files)

    def _get_train_transform(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomResizedCrop(64, scale=(0.9, 1.1)),
            ]
        )

    def _get_val_transform(self):
        return transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.npz_files[idx])

        axial = data["axial"]
        coronal = data["coronal"]
        sagittal = data["sagittal"]
        label = int(data.get("label", 0))

        if self.target_label is not None and label != self.target_label:
            label = -1

        transform = (
            self._get_train_transform() if self.augment else self._get_val_transform()
        )

        axial_t = transform((axial * 255).astype(np.uint8))
        coronal_t = transform((coronal * 255).astype(np.uint8))
        sagittal_t = transform((sagittal * 255).astype(np.uint8))

        return axial_t, coronal_t, sagittal_t, torch.tensor(float(label))


def stratified_split(
    dataset: Dataset, val_size: float = 0.2
) -> Tuple[List[int], List[int]]:
    labels = []
    for f in dataset.npz_files:
        labels.append(int(np.load(f).get("label", 0)))
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)), test_size=val_size, random_state=SEED, stratify=labels
    )
    return train_idx.tolist(), val_idx.tolist()


def train_epoch(model, loader, criterion, optimizer, device):
    """Train one epoch with calibration monitoring."""
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for axial, coronal, sagittal, labels in loader:
        axial, coronal, sagittal, labels = (
            axial.to(device),
            coronal.to(device),
            sagittal.to(device),
            labels.to(device),
        )

        # Filter out invalid labels
        valid_mask = labels >= 0
        if not valid_mask.all():
            continue

        optimizer.zero_grad()
        logits = model(axial, coronal, sagittal)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Track logits for calibration monitoring
        all_logits.extend(logits.cpu().detach().numpy())
        all_labels.extend(labels.cpu().detach().numpy())

    avg_loss = total_loss / max(len(loader), 1)

    # Calibration Monitor: Mean Logit Magnitude
    # Values > 10.0 indicate overconfidence
    mean_logit_magnitude = np.mean(np.abs(all_logits)) if all_logits else 0.0

    return avg_loss, mean_logit_magnitude


def evaluate(model, loader, criterion, device):
    """Evaluate model and return metrics."""
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for axial, coronal, sagittal, labels in loader:
            axial, coronal, sagittal, labels = (
                axial.to(device),
                coronal.to(device),
                sagittal.to(device),
                labels.to(device),
            )
            logits = model(axial, coronal, sagittal)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Apply temperature scaling for prediction
            probs = torch.sigmoid(logits / config_v2.TEMPERATURE)
            preds = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return (
        total_loss / len(loader),
        accuracy,
        precision,
        recall,
        f1,
        tp,
        tn,
        fp,
        fn,
    )


def main():
    print("=" * 60)
    print("TRAINING V2.0 - CALIBRATED FRAMEWORK")
    print("=" * 60)
    print(f"Temperature: {config_v2.TEMPERATURE}")
    print(f"MC Samples: {config_v2.MC_SAMPLES}")

    # Load dataset
    npz_dir = config_v2.OUTPUT_DIR
    full_dataset = NoduleDataset(npz_dir, augment=True)
    print(f"Total samples: {len(full_dataset)}")

    # Get labels for weighted sampling
    all_labels = []
    for f in full_dataset.npz_files:
        all_labels.append(int(np.load(f).get("label", 0)))

    pos_count = sum(all_labels)
    neg_count = len(all_labels) - pos_count
    print(f"Positive: {pos_count}, Negative: {neg_count}")

    # Create weighted sampler (3:1 ratio)
    class_weights = torch.FloatTensor([1.0, 3.0])
    sample_weights = class_weights[torch.LongTensor(all_labels)]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Split dataset
    train_idx, val_idx = stratified_split(full_dataset, val_size=0.2)
    train_dataset = NoduleDataset(npz_dir, augment=True)
    val_dataset = NoduleDataset(npz_dir, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=config_v2.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=config_v2.NUM_WORKERS,
    )

    # Create model
    model = create_model_v2().to(DEVICE)
    print(f"Model created: V2.0 with Deep Dropout")

    # Class weights for loss (imbalanced data)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}

    for epoch in range(NUM_EPOCHS):
        train_loss, mean_logit_mag = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )

        val_loss, accuracy, precision, recall, f1, tp, tn, fp, fn = evaluate(
            model, val_loader, criterion, DEVICE
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(accuracy)
        history["val_f1"].append(f1)

        # Calibration status
        calib_status = "OK"
        if mean_logit_mag > 10.0:
            calib_status = "OVERCONFIDENT"
        elif mean_logit_mag < 1.0:
            calib_status = "UNDERCONFIDENT"

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {accuracy:.4f} | "
            f"F1: {f1:.4f} | "
            f"[CALIB] Mean Logit Mag: {mean_logit_mag:.2f} ({calib_status})"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": accuracy,
                    "val_f1": f1,
                    "tp": tp,
                    "fn": fn,
                },
                MODEL_SAVE_PATH,
            )
            print(f"  → Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.plot(history["val_accuracy"], label="Val Accuracy")
    plt.plot(history["val_f1"], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Curve V2.0 - Calibrated")
    plt.legend()
    plt.savefig(config_v2.TRAINING_CURVE_PATH, dpi=100)
    print(f"Training curve saved to {config_v2.TRAINING_CURVE_PATH}")


if __name__ == "__main__":
    main()
