#!/usr/bin/env python3
"""
Clinical Dashboard for V2.0 Model - Memory Optimized
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config_v2
from architecture_v2 import load_model_v2

CUBE_SIZE = 64
MC_PASSES = 20  # Reduced from 50


def to_tensor(x):
    """Convert numpy array to tensor (expects 2D image, returns 4D tensor)."""
    t = torch.from_numpy(x.copy())
    t = t.float().unsqueeze(0).unsqueeze(0)  # (H,W) -> (1,1,H,W)
    return t


def normalize_to_hu(volume):
    return volume.astype(np.float32) - 1024


def apply_lung_window(hu_volume):
    return np.clip(hu_volume, -1000, 400)


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


def get_axial(cube):
    img = cube[cube.shape[0] // 2, :, :]
    return (img - img.min()) / (img.max() - img.min() + 1e-4)


def run_analysis(model_path, data_dir, annotations_file):
    os.makedirs("clinical_output", exist_ok=True)

    device = torch.device("cpu")
    model, checkpoint = load_model_v2(model_path, strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")

    annotations = pd.read_csv(annotations_file)
    mhd_files = sorted(glob.glob(os.path.join(data_dir, "*.mhd")))

    results = []
    total_mhd = len(mhd_files)

    for idx, mhd_path in enumerate(mhd_files):
        if idx % 20 == 0:
            print(f"Processing {idx}/{total_mhd}...")

        series_uid = os.path.basename(mhd_path).replace(".mhd", "")
        raw_path = mhd_path.replace(".mhd", ".raw")
        if not os.path.exists(raw_path):
            continue

        true_label = (
            1 if len(annotations[annotations["seriesuid"] == series_uid]) > 0 else 0
        )

        try:
            import SimpleITK as sitk

            img = sitk.ReadImage(mhd_path)
            volume = sitk.GetArrayFromImage(img)
            volume = apply_lung_window(normalize_to_hu(volume))
        except:
            continue

        cx, cy, cz = volume.shape[2] // 2, volume.shape[1] // 2, volume.shape[0] // 2
        cube = get_centered_cube(volume, cx, cy, cz, CUBE_SIZE)
        if cube is None:
            continue

        axial = get_axial(cube)
        img = (axial * 255).astype(np.uint8)
        axial_t = to_tensor(img).to(device)
        coronal_t = to_tensor(img).to(device)
        sagittal_t = to_tensor(img).to(device)

        # MC Dropout passes
        probs = []
        for _ in range(MC_PASSES):
            with torch.no_grad():
                try:
                    logit = model(axial_t, coronal_t, sagittal_t)
                    prob = torch.sigmoid(logit / config_v2.TEMPERATURE).item()
                    probs.append(prob)
                except Exception as e:
                    print(f"Error on {series_uid}: {e}")
                    probs.append(0.5)  # Fallback
        if not probs:
            probs = [0.5] * MC_PASSES

        probs = np.array(probs)
        mean_prob, uncertainty = np.mean(probs), np.std(probs)
        pred_label = 1 if mean_prob > 0.5 else 0

        results.append(
            {
                "series_uid": series_uid,
                "mean_prob": mean_prob,
                "uncertainty": uncertainty,
                "prediction": pred_label,
                "true_label": true_label,
                "axial": axial,
                "probs": probs,
            }
        )

    # Metrics
    total = len(results)
    tp = sum(1 for r in results if r["true_label"] == 1 and r["prediction"] == 1)
    tn = sum(1 for r in results if r["true_label"] == 0 and r["prediction"] == 0)
    fp = sum(1 for r in results if r["true_label"] == 0 and r["prediction"] == 1)
    fn = sum(1 for r in results if r["true_label"] == 1 and r["prediction"] == 0)
    accuracy = (tp + tn) / total

    # Add correct flag
    for r in results:
        r["correct"] = r["prediction"] == r["true_label"]
    correct = [r for r in results if r["correct"]]
    incorrect = [r for r in results if not r["correct"]]
    mean_unc = np.mean([r["uncertainty"] for r in results])

    unc_correct = np.mean([r["uncertainty"] for r in correct]) if correct else 0
    unc_incorrect = np.mean([r["uncertainty"] for r in incorrect]) if incorrect else 0

    high_risk = [
        r
        for r in results
        if r["true_label"] == 1 and r["prediction"] == 0 and r["uncertainty"] > mean_unc
    ]
    high_risk = sorted(high_risk, key=lambda x: x["uncertainty"], reverse=True)

    print("\n" + "=" * 60)
    print("CLINICAL DASHBOARD RESULTS")
    print("=" * 60)
    print(f"Total: {total}, TP:{tp}, TN:{tn}, FP:{fp}, FN:{fn}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean Uncertainty: {mean_unc:.4f}")
    print(f"Uncertainty Correct: {unc_correct:.4f}, Incorrect: {unc_incorrect:.4f}")
    print(f"High Risk Misses: {len(high_risk)}")

    # Generate plots
    generate_plots(results, high_risk, mean_unc)


def generate_plots(results, high_risk, mean_unc):
    print("\nGenerating plots...")

    # Sort results
    high_unc = sorted(results, key=lambda x: x["uncertainty"], reverse=True)[:10]
    low_unc = sorted(results, key=lambda x: x["uncertainty"])[:10]

    # 1. Edge Cases figure
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle("Clinical Dashboard - Edge Cases", fontsize=16, fontweight="bold")

    high_cases = high_unc[:5]
    low_cases = low_unc[:5]
    for row_idx, cases in enumerate([high_cases, low_cases]):
        for col_idx, case in enumerate(cases):
            ax = axes[row_idx, col_idx]
            ax.imshow(case["axial"], cmap="gray")
            status = "✓" if case["correct"] else "✗"
            risk = "⚠️" if case in high_risk else ""
            ax.set_title(
                f"{status} {risk}\nP:{case['mean_prob']:.2f} U:{case['uncertainty']:.3f}",
                fontsize=9,
            )
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("clinical_output/edge_cases.png", dpi=80)
    plt.close()

    # 2. MC Distributions
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("MC Dropout Distributions", fontsize=14, fontweight="bold")

    high_cases = high_unc[:5]
    low_cases = low_unc[:5]
    for row_idx, cases in enumerate([high_cases, low_cases]):
        for col_idx, case in enumerate(cases):
            ax = axes[row_idx, col_idx]
            probs = case["probs"]
            ax.hist(probs, bins=15, alpha=0.7, edgecolor="black")
            ax.axvline(case["mean_prob"], color="red", linestyle="--", linewidth=2)
            ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_title(f"Unc: {case['uncertainty']:.3f}", fontsize=10)

    plt.tight_layout()
    plt.savefig("clinical_output/mc_distributions.png", dpi=80)
    plt.close()

    # 3. Summary Gallery
    fig = plt.figure(figsize=(20, 14))

    # Metrics text
    tp = sum(1 for r in results if r["true_label"] == 1 and r["prediction"] == 1)
    tn = sum(1 for r in results if r["true_label"] == 0 and r["prediction"] == 0)
    fp = sum(1 for r in results if r["true_label"] == 0 and r["prediction"] == 1)
    fn = sum(1 for r in results if r["true_label"] == 1 and r["prediction"] == 0)
    accuracy = (tp + tn) / len(results)

    correct = [r for r in results if r["correct"]]
    incorrect = [r for r in results if not r["correct"]]
    unc_correct = np.mean([r["uncertainty"] for r in correct])
    unc_incorrect = np.mean([r["uncertainty"] for r in incorrect])

    # Title
    fig.suptitle(
        "Clinical Dashboard - V2.0 Calibrated Model\nSubset 6 Analysis",
        fontsize=16,
        fontweight="bold",
    )

    # Metrics panel
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.axis("off")
    metrics_text = f"""
    PERFORMANCE METRICS                    UNCERTAINTY ANALYSIS
    ========================             ========================
    Accuracy:   {accuracy:.4f}                   Mean Uncertainty: {mean_unc:.4f}
    Precision: {tp / (tp + fp):.4f}                   Correct:     {unc_correct:.4f}
    Recall:    {tp / (tp + fn):.4f}                   Incorrect:   {unc_incorrect:.4f}
    F1-Score:  {2 * tp / (2 * tp + fp + fn):.4f}                   Delta:       {unc_incorrect - unc_correct:.4f}
    
    TP:{tp} TN:{tn} FP:{fp} FN:{fn}           High Risk Misses: {len(high_risk)}
    """
    ax1.text(0.3, 0.5, metrics_text, fontsize=14, family="monospace", va="center")

    # High Risk section
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.axis("off")
    if high_risk:
        risk_text = "⚠️ HIGH RISK MISSES - REQUIRES RADIOLOGIST REVIEW ⚠️\n"
        risk_text += "=" * 60 + "\n"
        for i, case in enumerate(high_risk[:6]):
            risk_text += f"{i + 1}. {case['series_uid'][:45]}...\n"
            risk_text += f"   True: POS, Pred: NEG, Prob: {case['mean_prob']:.3f}, Unc: {case['uncertainty']:.3f}\n"
        ax2.text(0.02, 0.9, risk_text, fontsize=10, family="monospace", va="top")

    # Edge cases
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.axis("off")
    ax3.text(
        0.2, 0.8, "Sample Axial Slices - Edge Cases:", fontsize=12, fontweight="bold"
    )

    # Show a few samples
    for i, case in enumerate(high_unc[:3]):
        ax = fig.add_subplot(3, 5, 11 + i)
        ax.imshow(case["axial"], cmap="gray")
        status = "✓" if case["correct"] else "✗"
        ax.set_title(
            f"{status} P:{case['mean_prob']:.2f} U:{case['uncertainty']:.3f}",
            fontsize=9,
        )
        ax.axis("off")

    for i, case in enumerate(low_unc[:2]):
        ax = fig.add_subplot(3, 5, 13 + i)
        ax.imshow(case["axial"], cmap="gray")
        status = "✓" if case["correct"] else "✗"
        ax.set_title(
            f"{status} P:{case['mean_prob']:.2f} U:{case['uncertainty']:.3f}",
            fontsize=9,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("clinical_output/summary_gallery.png", dpi=100, bbox_inches="tight")
    plt.close()

    print("✓ Saved: edge_cases.png, mc_distributions.png, summary_gallery.png")


if __name__ == "__main__":
    run_analysis("models/calibrated_v2.pth", "subset6_data/subset6", "annotations.csv")
