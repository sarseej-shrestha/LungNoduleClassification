#!/usr/bin/env python3
"""
Lung Nodule CAD Training Script
- Triple-Branch ResNet-18 Architecture
- MC Dropout during training and validation
- Heavy augmentation
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from typing import Tuple, Optional, Dict, List
import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS = 100
DROPOUT_PROB = 0.5
NUM_MC_SAMPLES = 10
PATIENCE = 7
MODEL_SAVE_PATH = "best_nodule_model.pth"


class NoduleDataset(Dataset):
    def __init__(self, npz_dir: str, augment: bool = True):
        self.npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        self.augment = augment

    def __len__(self) -> int:
        return len(self.npz_files)

    def _get_transform(self):
        if self.augment:
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(15),
                ]
            )
        return transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.npz_files[idx])

        axial = data["axial"]
        coronal = data["coronal"]
        sagittal = data["sagittal"]

        label = 1.0

        axial_tensor = self._augment_view(axial)
        coronal_tensor = self._augment_view(coronal)
        sagittal_tensor = self._augment_view(sagittal)

        label_tensor = torch.tensor([label], dtype=torch.float32)

        return axial_tensor, coronal_tensor, sagittal_tensor, label_tensor

    def _augment_view(self, view: np.ndarray) -> torch.Tensor:
        img = (view * 255).astype(np.uint8)

        transform = self._get_transform()

        img_tensor = transform(img)

        return img_tensor.float()


class ResNet18Branch(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.backbone.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x


class MultiViewNet(nn.Module):
    def __init__(
        self, dropout_prob: float = DROPOUT_PROB, shared_backbone: bool = False
    ):
        super().__init__()

        self.shared_backbone = shared_backbone

        if shared_backbone:
            self.backbone = ResNet18Branch(in_channels=1)
        else:
            self.axial_branch = ResNet18Branch(in_channels=1)
            self.coronal_branch = ResNet18Branch(in_channels=1)
            self.sagittal_branch = ResNet18Branch(in_channels=1)

        feature_dim = 512 * 3 if not shared_backbone else 512 * 3

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(256, 1),
        )

    def _get_branch(self, branch_name):
        if self.shared_backbone:
            return self.backbone
        return getattr(self, branch_name)

    def forward(
        self, axial: torch.Tensor, coronal: torch.Tensor, sagittal: torch.Tensor
    ) -> torch.Tensor:

        if self.shared_backbone:
            feat_axial = self.backbone(axial)
            feat_coronal = self.backbone(coronal)
            feat_sagittal = self.backbone(sagittal)
        else:
            feat_axial = self.axial_branch(axial)
            feat_coronal = self.coronal_branch(coronal)
            feat_sagittal = self.sagittal_branch(sagittal)

        fused = torch.cat([feat_axial, feat_coronal, feat_sagittal], dim=1)

        output = self.classifier(fused)

        return output.squeeze(1)

    def predict_with_uncertainty(
        self,
        axial: torch.Tensor,
        coronal: torch.Tensor,
        sagittal: torch.Tensor,
        num_samples: int = NUM_MC_SAMPLES,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.train()

        outputs = []
        for _ in range(num_samples):
            with torch.no_grad():
                output = self.forward(axial, coronal, sagittal)
                outputs.append(output)

        stacked = torch.stack(outputs, dim=0)

        mean_pred = torch.mean(stacked, dim=0)
        std_pred = torch.std(stacked, dim=0)

        return mean_pred, std_pred


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for axial, coronal, sagittal, labels in dataloader:
        axial = axial.to(device)
        coronal = coronal.to(device)
        sagittal = sagittal.to(device)
        labels = labels.squeeze().to(device)

        optimizer.zero_grad()

        outputs = model(axial, coronal, sagittal)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for axial, coronal, sagittal, labels in dataloader:
        axial = axial.to(device)
        coronal = coronal.to(device)
        sagittal = sagittal.to(device)
        labels = labels.squeeze().to(device)

        outputs = model(axial, coronal, sagittal)
        loss = criterion(outputs, labels)

        total_loss += loss.item()

        preds = torch.sigmoid(outputs.detach())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    binary_preds = (all_preds > 0.5).astype(float)
    accuracy = np.mean(binary_preds == all_labels)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "predictions": all_preds,
        "labels": all_labels,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int,
    device: torch.device,
    save_path: str = MODEL_SAVE_PATH,
) -> Dict:

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        if val_loader is not None:
            val_results = evaluate(model, val_loader, criterion, device)
            val_loss = val_results["loss"]
            val_acc = val_results["accuracy"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                    },
                    save_path,
                )
                print(f"  -> Saved best model to {save_path}")
            else:
                patience_counter += 1
                print(f"  -> No improvement ({patience_counter}/{PATIENCE})")

            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        else:
            history["train_loss"].append(train_loss)
            print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f}")

    return history


def plot_training_history(history: Dict, save_path: str = "training_curve.png"):
    plt.figure(figsize=(10, 6))

    plt.plot(history["train_loss"], label="Train Loss", marker="o", markersize=3)
    plt.plot(history["val_loss"], label="Validation Loss", marker="s", markersize=3)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Training curve saved to {save_path}")


def main():
    print(f"Using device: {DEVICE}")

    print("Loading dataset...")
    dataset = NoduleDataset("preprocessed", augment=True)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"Total samples: {len(dataset)}, Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print("Creating model (3 separate ResNet-18 branches)...")
    model = MultiViewNet(dropout_prob=DROPOUT_PROB, shared_backbone=False)
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nStarting training...")
    print(f"Learning rate: {LEARNING_RATE}, Patience: {PATIENCE}")
    history = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        save_path=MODEL_SAVE_PATH,
    )

    print("\nPlotting training history...")
    plot_training_history(history, save_path="training_curve.png")

    print("\nTraining complete!")
    print(f"Best model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
