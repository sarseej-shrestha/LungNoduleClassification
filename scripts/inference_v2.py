#!/usr/bin/env python3
"""
Test Inference for Subset 6 V2.0 - Calibrated Framework
With Temperature Scaling, Brier Score, and Clinical Readiness Check
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import SimpleITK as sitk

from src.config_v2 import config_v2
from src.architecture_v2 import (
    MCDropout,
    ResNet18WithDeepDropout,
    SharedResNet_V2,
    load_model_v2,
    apply_temperature_scaling,
)

HU_RESCALING_INTERCEPT = -1024
HU_LOWER = -1000
HU_UPPER = 400
CUBE_SIZE = 64


def normalize_to_hu(volume):
    return volume.astype(np.float32) + HU_RESCALING_INTERCEPT


def apply_lung_window(hu_volume):
    return np.clip(hu_volume, HU_LOWER, HU_UPPER)


def get_centered_cube(volume, cx, cy, cz, size=CUBE_SIZE):
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


def create_multiplanar(cube):
    axial = cube[cube.shape[0] // 2, :, :]
    coronal = cube[:, cube.shape[1] // 2, :]
    sagittal = cube[:, :, cube.shape[2] // 2]

    axial = (axial - axial.min()) / (axial.max() - axial.min() + 1e-4)
    coronal = (coronal - coronal.min()) / (coronal.max() - coronal.min() + 1e-4)
    sagittal = (sagittal - sagittal.min()) / (sagittal.max() - sagittal.min() + 1e-4)

    return axial, coronal, sagittal


def calculate_brier_score(probs, labels):
    """Calculate Brier Score: mean((prob - label)^2).
    Lower is better. Range: [0, 1]."""
    probs = np.array(probs)
    labels = np.array(labels)
    return np.mean((probs - labels) ** 2)


def run_inference_v2(model_path, data_dir, annotations_file):
    """Run inference with temperature scaling and clinical readiness check."""
    device = torch.device("cpu")

    model, checkpoint = load_model_v2(model_path, strict=False)
    if model is None:
        print(f"Error: Could not load model from {model_path}")
        return

    model.to(device)
    model.eval()
    print(f"Loaded V2.0 model from epoch {checkpoint.get('epoch', 'N/A')}")
    print(f"Temperature: {config_v2.TEMPERATURE}")
    print(f"MC Samples: {config_v2.MC_SAMPLES}")

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

        # Get true label from annotations
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

        # MC Dropout inference with temperature scaling
        # probs = sigmoid(logits / temperature)
        logits_list = []
        for _ in range(config_v2.MC_SAMPLES):
            with torch.no_grad():
                logit = model.forward(axial_t, coronal_t, sagittal_t)
                logits_list.append(logit)

        logits_stack = torch.stack(logits_list, dim=0)

        # Temperature scaling
        scaled_logits = logits_stack / config_v2.TEMPERATURE
        probs = torch.sigmoid(scaled_logits)

        mean_prob = torch.mean(probs).item()
        uncertainty = torch.std(probs).item()

        # Unscaled for comparison
        unscaled_probs = torch.sigmoid(logits_stack)
        unscaled_unc = torch.std(unscaled_probs).item()

        # Raw logits for Brier Score
        raw_logit = torch.mean(logits_stack).item()

        pred_label = 1 if mean_prob > 0.5 else 0

        results.append(
            {
                "series_uid": series_uid,
                "mean_prob": mean_prob,
                "uncertainty": uncertainty,
                "prediction": pred_label,
                "true_label": true_label,
                "correct": pred_label == true_label,
                "unscaled_uncertainty": unscaled_unc,
                "raw_logit": raw_logit,
            }
        )

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(mhd_files)}")

    # ========== REPORT ==========
    print("\n" + "=" * 60)
    print("V2.0 CALIBRATED INFERENCE - SUBSET 6")
    print("=" * 60)
    print(f"Total samples: {len(results)}")
    print(f"Temperature: {config_v2.TEMPERATURE}")
    print(f"MC Samples: {config_v2.MC_SAMPLES}")

    # Confusion Matrix
    print("\n" + "-" * 40)
    print("CONFUSION MATRIX")
    print("-" * 40)

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

    # Brier Score
    print("\n" + "-" * 40)
    print("BRIER SCORE (Calibration Quality)")
    print("-" * 40)
    probs = [r["mean_prob"] for r in results]
    labels = [r["true_label"] for r in results]
    brier = calculate_brier_score(probs, labels)
    print(f"Brier Score: {brier:.4f}")
    print(f"  (0.0 = perfect, 0.25 = random, lower is better)")

    # Uncertainty Validation
    print("\n" + "-" * 40)
    print("UNCERTAINTY ANALYSIS")
    print("-" * 40)

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

    # Unscaled comparison
    unscaled_unc_correct = (
        np.mean([r["unscaled_uncertainty"] for r in correct_results])
        if correct_results
        else 0
    )
    unscaled_unc_incorrect = (
        np.mean([r["unscaled_uncertainty"] for r in incorrect_results])
        if incorrect_results
        else 0
    )

    print(f"\n[Without Temperature Scaling]")
    print(f"Uncertainty (Correct):   {unscaled_unc_correct:.4f}")
    print(f"Uncertainty (Incorrect): {unscaled_unc_incorrect:.4f}")

    # Clinical Readiness Check
    print("\n" + "=" * 60)
    print("CLINICAL READINESS CHECK")
    print("=" * 60)

    uncertainty_difference = unc_incorrect - unc_correct

    if uncertainty_difference > 0.05:
        readiness = "PASS"
        verdict = "Clinically Useful"
    else:
        readiness = "FAIL"
        verdict = "NOT Clinically Useful"

    print(f"Uncertainty Difference: {uncertainty_difference:.4f}")
    print(f"  (Incorrect - Correct, positive = model uncertain when wrong)")

    print(f"\nClinical Readiness: {readiness}")
    print(f"  → {verdict}")

    if readiness == "PASS":
        print(f"  ✓ Model is significantly more uncertain when it makes mistakes")
    else:
        print(f"  ✗ Model shows similar uncertainty for correct/incorrect predictions")

    # Show hardest cases
    print("\n" + "-" * 40)
    print("Top 5 Most Uncertain Misclassifications:")
    misclass = sorted(incorrect_results, key=lambda x: x["uncertainty"], reverse=True)[
        :5
    ]
    for r in misclass:
        print(
            f"  {r['series_uid'][:50]}: true={r['true_label']}, pred={r['prediction']}, prob={r['mean_prob']:.3f}, unc={r['uncertainty']:.3f}"
        )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/calibrated_v2.pth")
    parser.add_argument("--data_dir", default="input/subset6")
    parser.add_argument("--annotations_file", default="input/annotations.csv")
    args = parser.parse_args()

    run_inference_v2(args.model_path, args.data_dir, args.annotations_file)
