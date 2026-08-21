#!/usr/bin/env python3
"""
Patient-level 5-fold training. Each fold is trained and evaluated independently
against its own held-out val patients (from input/splits/patient_splits.json); the
locked test set is never touched here.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config
from src.dataset import NoduleDataset, filter_by_patients, index_samples, load_splits
from src.model import create_model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_fold(fold_idx, index, splits, device, epochs=None, patience=None):
    epochs = epochs or config.NUM_EPOCHS
    patience = patience or config.PATIENCE

    fold = splits["folds"][fold_idx]
    train_samples = filter_by_patients(index, fold["train"])
    val_samples = filter_by_patients(index, fold["val"])

    n_pos = sum(1 for s in train_samples if s["label"] == 1)
    n_neg = len(train_samples) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(
        f"Fold {fold_idx}: train={len(train_samples)} (pos={n_pos}, neg={n_neg}), "
        f"val={len(val_samples)}"
    )

    train_loader = DataLoader(
        NoduleDataset(train_samples, augment=True),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        NoduleDataset(val_samples, augment=False),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    model = create_model().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.MODEL_DIR, f"fold{fold_idx}_best.pth")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for axial, coronal, sagittal, labels in train_loader:
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
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for axial, coronal, sagittal, labels in val_loader:
                axial, coronal, sagittal, labels = (
                    axial.to(device),
                    coronal.to(device),
                    sagittal.to(device),
                    labels.to(device),
                )
                outputs = model(axial, coronal, sagittal)
                val_loss += criterion(outputs, labels).item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        scheduler.step(val_loss)

        print(
            f"Fold {fold_idx} Epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "fold": fold_idx,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Fold {fold_idx}: early stopping at epoch {epoch + 1}")
                break

    return history, ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=str, default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    index = index_samples()
    splits = load_splits()
    print(f"Indexed {len(index)} samples")

    fold_ids = [int(x) for x in args.folds.split(",")]
    all_histories = {}
    for fold_idx in fold_ids:
        print(f"\n=== Fold {fold_idx} ===")
        history, ckpt_path = train_fold(fold_idx, index, splits, device, epochs=args.epochs)
        all_histories[fold_idx] = history
        print(f"Best checkpoint: {ckpt_path}")

    os.makedirs("output", exist_ok=True)
    with open("output/train_history.json", "w") as f:
        json.dump(all_histories, f, indent=2)


if __name__ == "__main__":
    main()
