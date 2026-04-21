#!/usr/bin/env python3
"""
Lung Nodule CAD Training - Stable Version
- Aggressive augmentation (minority class focus)
- Class weights (3:1 ratio)
- Lower LR with weight decay
- Confusion matrix logging
- Early stopping on val_loss
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision import models
from typing import Tuple, Optional, Dict, List
import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
DROPOUT_PROB = 0.5
PATIENCE = 10
MODEL_SAVE_PATH = "best_uncertainty_model.pth"


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

        transform = (
            self._get_train_transform() if self.augment else self._get_val_transform()
        )

        axial_t = transform((axial * 255).astype(np.uint8))
        coronal_t = transform((coronal * 255).astype(np.uint8))
        sagittal_t = transform((sagittal * 255).astype(np.uint8))

        return axial_t, coronal_t, sagittal_t, torch.tensor(float(label))


class ResNet18Branch(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.backbone.bn1 = nn.BatchNorm2d(64)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)


class MultiViewNet(nn.Module):
    def __init__(self, dropout_prob: float = DROPOUT_PROB):
        super().__init__()
        self.backbone = ResNet18Branch(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(512 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(256, 1),
        )

    def forward(self, axial, coronal, sagittal):
        f1 = self.backbone(axial)
        f2 = self.backbone(coronal)
        f3 = self.backbone(sagittal)
        fused = torch.cat([f1, f2, f3], dim=1)
        return self.classifier(fused).squeeze(1)


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
    model.train()
    total_loss = 0.0
    for axial, coronal, sagittal, labels in loader:
        axial, coronal, sagittal, labels = (
            axial.to(device),
            coronal.to(device),
            sagittal.to(device),
            labels.to(device),
        )
        optimizer.zero_grad()
        outputs = model(axial, coronal, sagittal)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
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
            outputs = model(axial, coronal, sagittal)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
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

    return {
        "loss": total_loss / len(loader),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def main():
    print("=" * 55)
    print("LUNG NODULE CAD - STABLE TRAINING")
    print("=" * 55)

    # Load dataset
    print("\n[1] Loading data...")
    full_dataset = NoduleDataset("preprocessed", augment=True)
    print(f"Total samples: {len(full_dataset)}")

    # Calculate class weights: ~3:1 ratio (521 neg / 174 pos = 3.0)
    all_labels = []
    for f in full_dataset.npz_files:
        all_labels.append(int(np.load(f).get("label", 0)))
    n_pos, n_neg = sum(all_labels), len(all_labels) - sum(all_labels)
    pos_weight = n_neg / n_pos  # ~3.0
    print(f"Positive: {n_pos}, Negative: {n_neg}, pos_weight: {pos_weight:.2f}")

    # Stratified split
    train_idx, val_idx = stratified_split(full_dataset)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Create data loaders - VALID SET UNAUGMENTED
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = NoduleDataset("preprocessed", augment=False)
    val_dataset.npz_files = [full_dataset.npz_files[i] for i in val_idx]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Create model
    print("\n[2] Creating model...")
    model = MultiViewNet(DROPOUT_PROB).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Loss with class weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    # Optimizer with weight decay
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    print("\n[3] Training...")
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_results = evaluate(model, val_loader, criterion, DEVICE)
        val_loss = val_results["loss"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_results["accuracy"])
        history["val_f1"].append(val_results["f1"])

        scheduler.step(val_loss)

        # Confusion matrix logging
        print(
            f"Epoch [{epoch + 1:02d}] "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Acc: {val_results['accuracy']:.4f} | "
            f"F1: {val_results['f1']:.4f} | "
            f"TP:{val_results['tp']} TN:{val_results['tn']} FP:{val_results['fp']} FN:{val_results['fn']}"
        )

        # Early stopping - save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_results["accuracy"],
                    "val_f1": val_results["f1"],
                    "tp": val_results["tp"],
                    "fn": val_results["fn"],
                },
                MODEL_SAVE_PATH,
            )
            marker = " <-- BEST"
        else:
            patience_counter += 1
            marker = ""

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Plot training curve
    print("\n[4] Saving training curve...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "s-", label="Val")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_acc"], "o-", label="Acc")
    axes[1].plot(epochs, history["val_f1"], "s-", label="F1")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=100)
    plt.close()

    # Summary
    print("\n" + "=" * 55)
    print("FINAL RESULTS")
    print("=" * 55)
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Final Acc: {history['val_acc'][-1]:.4f}")
    print(f"Final F1: {history['val_f1'][-1]:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
