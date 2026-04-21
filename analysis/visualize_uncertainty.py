#!/usr/bin/env python3
"""
Uncertainty Analysis - Matching original architecture
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
DEVICE = torch.device("cpu")
NUM_MC_SAMPLES = 25
MODEL_PATH = "best_uncertainty_model.pth"


class NoduleDataset(Dataset):
    def __init__(self, file_list):
        self.npz_files = file_list

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        data = np.load(self.npz_files[idx])
        transform = transforms.ToTensor()

        axial = transform((data["axial"] * 255).astype(np.uint8))
        coronal = transform((data["coronal"] * 255).astype(np.uint8))
        sagittal = transform((data["sagittal"] * 255).astype(np.uint8))

        return axial, coronal, sagittal, int(data.get("label", 0)), self.npz_files[idx]


class ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        # Modify first conv for 1-channel input
        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            self.conv1.weight = nn.Parameter(resnet.conv1.weight.repeat(1, 3, 1, 1) / 3)
        self.bn1 = resnet.bn1
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = resnet.avgpool
        self.fc = nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)


class MultiViewNet(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        self.backbone = ResNet18Backbone()

        # Match original: 3 views × 512 = 1536 features
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

    def predict_with_uncertainty(self, axial, coronal, sagittal, n=25):
        self.eval()

        # Manual MC Dropout - eval mode but apply dropout
        outputs = []
        for _ in range(n):
            fused = torch.cat(
                [self.backbone(axial), self.backbone(coronal), self.backbone(sagittal)],
                dim=1,
            )

            # Apply classifier with dropout
            x = fused
            for i, layer in enumerate(self.classifier):
                if isinstance(layer, nn.Dropout):
                    x = layer(x)
                else:
                    x = layer(x)

            outputs.append(torch.sigmoid(x))

        stacked = torch.stack(outputs, dim=0)
        return torch.mean(stacked).item(), torch.std(stacked).item()


def main():
    print("=" * 55)
    print("UNCERTAINTY ANALYSIS WITH MC DROPOUT")
    print("=" * 55)

    # Load model
    print("\n[1] Loading model...")
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = MultiViewNet()
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(DEVICE)
    print(f"Epoch: {ckpt.get('epoch', 'N/A')}")

    # Load data
    print("\n[2] Loading validation data...")
    all_files = sorted(glob.glob("preprocessed/*.npz"))
    val_files = all_files[::4][:60]
    dataset = NoduleDataset(val_files)
    loader = DataLoader(dataset, batch_size=1)
    print(f"Samples: {len(dataset)}")

    # Run MC Dropout
    print(f"\n[3] Running MC Dropout ({NUM_MC_SAMPLES} passes)...")
    results = []
    for i, (ax, co, sa, lb, pt) in enumerate(loader):
        mp, un = model.predict_with_uncertainty(
            ax.to(DEVICE), co.to(DEVICE), sa.to(DEVICE), NUM_MC_SAMPLES
        )
        pred = 1 if mp > 0.5 else 0
        results.append(
            {
                "mean_prob": mp,
                "uncertainty": un,
                "prediction": pred,
                "true_label": lb.item(),
                "correct": pred == lb.item(),
                "path": pt[0],
            }
        )
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(loader)}")

    # Analysis
    print("\n" + "=" * 55)
    print("UNCERTAINTY ANALYSIS")
    print("=" * 55)

    tp = [r for r in results if r["true_label"] == 1 and r["prediction"] == 1]
    tn = [r for r in results if r["true_label"] == 0 and r["prediction"] == 0]
    fp = [r for r in results if r["true_label"] == 0 and r["prediction"] == 1]
    fn = [r for r in results if r["true_label"] == 1 and r["prediction"] == 0]

    print(f"\nConfusion Matrix:")
    print(f"  TP: {len(tp)}, TN: {len(tn)}, FP: {len(fp)}, FN: {len(fn)}")

    tp_unc = np.mean([r["uncertainty"] for r in tp]) if tp else 0
    fn_unc = np.mean([r["uncertainty"] for r in fn]) if fn else 0

    print(f"\nUncertainty by Class:")
    print(f"  True Positives (n={len(tp)}):  {tp_unc:.4f}")
    print(f"  False Negatives (n={len(fn)}): {fn_unc:.4f}")
    print(
        f"  True Negatives (n={len(tn)}):  {tn_unc:.4f}"
        if (tn_unc := np.mean([r["uncertainty"] for r in tn]))
        else ""
    )

    print("\n" + "=" * 55)
    print("THE 'AHA!' DISCOVERY")
    print("=" * 55)
    if fn and tp:
        if fn_unc > tp_unc:
            print(f">>> AHA! Missed nodules (FN) have HIGHER uncertainty!")
            print(f"    FN: {fn_unc:.4f} > TP: {tp_unc:.4f}")
        else:
            print(f">>> Model uncertainty similar on missed cases.")

    correct = sum(1 for r in results if r["correct"])
    print(f"\nAccuracy: {correct}/{len(results)} = {correct / len(results):.1%}")

    # Visualize top uncertain
    print("\n[4] Creating visualization...")
    misclass = sorted(
        [r for r in results if not r["correct"]],
        key=lambda x: x["uncertainty"],
        reverse=True,
    )[:4]

    if misclass:
        fig, axes = plt.subplots(len(misclass), 4, figsize=(12, 3 * len(misclass)))

        for row, case in enumerate(misclass):
            data = np.load(case["path"])

            for col, (ax, key) in enumerate(
                zip(axes[row], ["axial", "coronal", "sagittal"])
            ):
                ax.imshow(data[key], cmap="gray")
                ax.set_title(key.upper(), fontsize=9)
                ax.axis("off")

            axes[row, 3].text(
                0.1,
                0.7,
                f"True: {'NODULE' if case['true_label'] else 'NON'}",
                transform=axes[row, 3].transAxes,
                fontsize=9,
            )
            axes[row, 3].text(
                0.1,
                0.5,
                f"Pred: {'NODULE' if case['prediction'] else 'NON'}",
                transform=axes[row, 3].transAxes,
                fontsize=9,
            )
            axes[row, 3].text(
                0.1,
                0.3,
                f"Error: {'FN' if case['true_label'] == 1 else 'FP'}",
                transform=axes[row, 3].transAxes,
                color="red",
                fontsize=9,
            )
            axes[row, 3].text(
                0.1,
                0.1,
                f"Uncert: {case['uncertainty']:.3f}",
                transform=axes[row, 3].transAxes,
                color="blue",
                fontsize=9,
            )
            axes[row, 3].axis("off")

        plt.suptitle("Most Uncertain Misclassifications", fontsize=12)
        plt.tight_layout()
        plt.savefig("uncertainty_grid.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: uncertainty_grid.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
