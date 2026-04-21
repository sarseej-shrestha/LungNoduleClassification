#!/usr/bin/env python3
"""
Test Inference for Subset 6
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import SimpleITK as sitk
import config


class MCDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training or not self.p:
            return nn.functional.dropout(x, p=self.p, training=self.training)
        # In eval mode, use dropout with the mask
        mask = (torch.rand(x.size(0), x.size(1), device=x.device) > self.p).float() / (
            1 - self.p
        )
        return x * mask.unsqueeze(0)


HU_RESCALING_INTERCEPT = -1024
HU_LOWER = -1000
HU_UPPER = 400
CUBE_SIZE = 64


def get_backbone(name):
    if name.lower() == "resnet18":
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif name.lower() == "resnet34":
        return models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    raise ValueError(f"Unknown backbone: {name}")


class ResNet18Branch(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.backbone = get_backbone("resnet18")
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.backbone.bn1 = nn.BatchNorm2d(64)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)


class MultiViewNet(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        self.backbone = ResNet18Branch(1)
        self.dropout1 = MCDropout(p=dropout_prob)
        self.fc1 = nn.Linear(512 * 3, 256)
        self.relu = nn.ReLU()
        self.dropout2 = MCDropout(p=dropout_prob)
        self.fc2 = nn.Linear(256, 1)
        self.dropout_prob = dropout_prob

    def forward(self, axial, coronal, sagittal):
        f1 = self.backbone(axial)
        f2 = self.backbone(coronal)
        f3 = self.backbone(sagittal)
        fused = torch.cat([f1, f2, f3], dim=1)

        x = self.dropout1(fused)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x.squeeze(1)


def normalize_to_hu(volume):
    hu = volume.astype(np.float32) + HU_RESCALING_INTERCEPT
    return hu


def apply_lung_window(hu_volume):
    clipped = np.clip(hu_volume, HU_LOWER, HU_UPPER)
    return clipped


def normalize_window(volume, mean=None, std=None):
    vol = volume.astype(np.float32)
    if mean is None:
        mean = vol.mean()
    if std is None:
        std = vol.std()
    vol = (vol - mean) / (std + 1e-4)
    vol = np.clip(vol, -1, 1)
    vol = (vol + 1) / 2
    return vol


def extract_cube(volume, cx, cy, cz, size=CUBE_SIZE):
    d, h, w = volume.shape
    half = size // 2
    z1, z2 = cz - half, cz + half
    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half

    if z1 < 0 or z2 > d or y1 < 0 or y2 > h or x1 < 0 or x2 > w:
        return None

    cube = volume[z1:z2, y1:y2, x1:x2]
    if cube.shape != (size, size, size):
        return None

    return cube


def get_centered_cube(volume, cx, cy, cz, size=CUBE_SIZE):
    return extract_cube(volume, cx, cy, cz, size)


def create_multiplanar(cube):
    axial = cube[cube.shape[0] // 2, :, :]
    coronal = cube[:, cube.shape[1] // 2, :]
    sagittal = cube[:, :, cube.shape[2] // 2]

    axial = (axial - axial.min()) / (axial.max() - axial.min() + 1e-4)
    coronal = (coronal - coronal.min()) / (coronal.max() - coronal.min() + 1e-4)
    sagittal = (sagittal - sagittal.min()) / (sagittal.max() - sagittal.min() + 1e-4)

    return axial, coronal, sagittal


def run_inference_with_labels(
    model, device, data_dir, model_path, annotations_file, num_samples=50
):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded from epoch {checkpoint.get('epoch', 'N/A')}")

    annotations = pd.read_csv(annotations_file)

    mhd_files = sorted(glob.glob(os.path.join(data_dir, "*.mhd")))
    print(f"Processing {len(mhd_files)} samples...")

    results = []
    transform = transforms.ToTensor()

    for i, mhd_path in enumerate(mhd_files):
        series_uid = os.path.basename(mhd_path).replace(".mhd", "")
        raw_path = mhd_path.replace(".mhd", ".raw")

        if not os.path.exists(raw_path):
            continue

        annotation_rows = annotations[annotations["seriesuid"] == series_uid]
        true_label = 1 if len(annotation_rows) > 0 else 0

        try:
            img = sitk.ReadImage(mhd_path)
            volume = sitk.GetArrayFromImage(img)
            volume = normalize_to_hu(volume)
            volume = apply_lung_window(volume)
        except Exception as e:
            print(f"Error loading {series_uid}: {e}")
            continue

        d, h, w = volume.shape
        cx, cy, cz = w // 2, h // 2, d // 2

        cube = get_centered_cube(volume, cx, cy, cz, CUBE_SIZE)
        if cube is None:
            continue

        axial, coronal, sagittal = create_multiplanar(cube)

        axial_t = transform((axial * 255).astype(np.uint8)).unsqueeze(0).to(device)
        coronal_t = transform((coronal * 255).astype(np.uint8)).unsqueeze(0).to(device)
        sagittal_t = (
            transform((sagittal * 255).astype(np.uint8)).unsqueeze(0).to(device)
        )

        model.eval()
        outputs = []
        for _ in range(num_samples):
            with torch.no_grad():
                output = model(axial_t, coronal_t, sagittal_t)
                outputs.append(torch.sigmoid(output))

        stacked = torch.stack(outputs, dim=0)
        mean_prob = torch.mean(stacked).item()
        uncertainty = torch.std(stacked).item()

        pred_label = 1 if mean_prob > 0.5 else 0

        results.append(
            {
                "series_uid": series_uid,
                "mean_prob": mean_prob,
                "uncertainty": uncertainty,
                "prediction": pred_label,
                "true_label": true_label,
                "correct": pred_label == true_label,
            }
        )

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(mhd_files)}")

    print("\n" + "=" * 60)
    print("SUBSET 6 INFERENCE RESULTS WITH LABELS")
    print("=" * 60)
    print(f"Total samples: {len(results)}")

    pos_count = sum(1 for r in results if r["prediction"] == 1)
    neg_count = len(results) - pos_count
    print(f"Positive predictions: {pos_count}")
    print(f"Negative predictions: {neg_count}")

    actual_pos = sum(1 for r in results if r["true_label"] == 1)
    actual_neg = len(results) - actual_pos
    print(f"Actual positives: {actual_pos}")
    print(f"Actual negatives: {actual_neg}")

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    tp = sum(1 for r in results if r["true_label"] == 1 and r["prediction"] == 1)
    tn = sum(1 for r in results if r["true_label"] == 0 and r["prediction"] == 0)
    fp = sum(1 for r in results if r["true_label"] == 0 and r["prediction"] == 1)
    fn = sum(1 for r in results if r["true_label"] == 1 and r["prediction"] == 0)

    print(f"TP (True Positives):  {tp}")
    print(f"TN (True Negatives): {tn}")
    print(f"FP (False Positives): {fp}")
    print(f"FN (False Negatives): {fn}")

    accuracy = (tp + tn) / len(results) if len(results) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:   {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    print("\n" + "=" * 60)
    print("UNCERTAINTY VALIDATION")
    print("=" * 60)

    correct_results = [r for r in results if r["correct"]]
    incorrect_results = [r for r in results if not r["correct"]]

    unc_correct = (
        np.mean([r["uncertainty"] for r in correct_results]) if correct_results else 0
    )
    unc_incorrect = (
        np.mean([r["uncertainty"] for r in incorrect_results])
        if incorrect_results
        else 0
    )

    print(f"Uncertainty (Correct):   {unc_correct:.4f}")
    print(f"Uncertainty (Incorrect): {unc_incorrect:.4f}")

    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)

    if unc_incorrect > unc_correct:
        print("Uncertainty Logic is VALID and Clinically Useful")
        print(
            f"  → Incorrect predictions have higher uncertainty ({unc_incorrect:.4f} > {unc_correct:.4f})"
        )
    else:
        print("Uncertainty Logic is NOT Clinically Useful")
        print(
            f"  → Incorrect predictions have lower/equal uncertainty ({unc_incorrect:.4f} <= {unc_correct:.4f})"
        )

    misclass = sorted(incorrect_results, key=lambda x: x["uncertainty"], reverse=True)[
        :5
    ]
    if misclass:
        print("\nMisclassified with highest uncertainty:")
        for r in misclass:
            print(
                f"  {r['series_uid'][:50]}: true={r['true_label']}, pred={r['prediction']}, unc={r['uncertainty']:.3f}"
            )

    return results


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/best_uncertainty_model.pth")
    parser.add_argument("--data_dir", default="subset6_data/subset6")
    parser.add_argument("--annotations_file", default="annotations.csv")
    args = parser.parse_args()

    device = torch.device("cpu")
    model = MultiViewNet(dropout_prob=config.DROPOUT_PROB)
    run_inference_with_labels(
        model, device, args.data_dir, args.model_path, args.annotations_file
    )
