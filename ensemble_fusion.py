#!/usr/bin/env python3
"""Ensemble Fusion - Optimized"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import SimpleITK as sitk

from architecture import load_model as load_model_v1
from architecture_v2 import load_model_v2

CUBE_SIZE = 64


def to_tensor(x):
    t = torch.from_numpy(x.copy())
    return t.float().unsqueeze(0).unsqueeze(0)


def get_cube(vol, cx, cy, cz, size=CUBE_SIZE):
    d, h, w = vol.shape
    half = size // 2
    z1, z2 = cz - half, cz + half
    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half
    if z1 < 0 or z2 > d or y1 < 0 or y2 > h or x1 < 0 or x2 > w:
        return None
    cube = vol[z1:z2, y1:y2, x1:x2]
    if cube.shape != (size, size, size):
        return None
    return cube


def get_axial(cube):
    img = cube[cube.shape[0] // 2, :, :]
    return (img - img.min()) / (img.max() - img.min() + 1e-4)


def get_prob(model, axial_t, temp=1.0):
    """Single forward pass."""
    with torch.no_grad():
        try:
            logit = model(axial_t, axial_t, axial_t)
            prob = torch.sigmoid(logit / temp).item()
            return prob
        except:
            return 0.5


def run():
    print("Loading models...")

    # Load models
    model_v1, _ = load_model_v1("models/best_uncertainty_model.pth", strict=False)
    model_v1.eval()

    model_v2, _ = load_model_v2("models/calibrated_v2.pth", strict=False)
    model_v2.eval()

    print("V1 and V2 loaded")

    # Load data
    annot = pd.read_csv("annotations.csv")
    files = sorted(glob.glob("subset6_data/subset6/*.mhd"))
    print(f"Processing {len(files)} files...")

    results = []
    for i, f in enumerate(files):
        if i % 20 == 0:
            print(f"  {i}/{len(files)}")

        uid = os.path.basename(f).replace(".mhd", "")
        true_lbl = 1 if len(annot[annot["seriesuid"] == uid]) > 0 else 0

        try:
            vol = sitk.GetArrayFromImage(sitk.ReadImage(f))
            vol = np.clip(vol.astype(np.float32) - 1024, -1000, 400)
        except:
            continue

        cx, cy, cz = vol.shape[2] // 2, vol.shape[1] // 2, vol.shape[0] // 2
        cube = get_cube(vol, cx, cy, cz, CUBE_SIZE)
        if cube is None:
            continue

        axial = get_axial(cube)
        img = (axial * 255).astype(np.uint8)
        t = to_tensor(img)

        # Get predictions
        p_v1 = get_prob(model_v1, t, temp=1.0)
        p_v2 = get_prob(model_v2, t, temp=1.5)

        # Ensemble
        p_avg = (p_v1 + p_v2) / 2
        disagreement = abs(p_v1 - p_v2)
        pred = 1 if p_avg > 0.5 else 0

        results.append(
            {
                "uid": uid,
                "p_v1": p_v1,
                "p_v2": p_v2,
                "p_avg": p_avg,
                "disc": disagreement,
                "pred": pred,
                "true": true_lbl,
                "axial": axial,
            }
        )

    # Analysis
    print("\n" + "=" * 50)
    print("ENSEMBLE RESULTS")
    print("=" * 50)

    tp = sum(1 for r in results if r["true"] == 1 and r["pred"] == 1)
    tn = sum(1 for r in results if r["true"] == 0 and r["pred"] == 0)
    fp = sum(1 for r in results if r["true"] == 0 and r["pred"] == 1)
    fn = sum(1 for r in results if r["true"] == 1 and r["pred"] == 0)

    acc = (tp + tn) / len(results)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

    # Disagreement analysis
    for r in results:
        r["correct"] = r["pred"] == r["true"]

    correct = [r for r in results if r["correct"]]
    incorrect = [r for r in results if not r["correct"]]

    unc_c = np.mean([r["disc"] for r in correct]) if correct else 0
    unc_i = np.mean([r["disc"] for r in incorrect]) if incorrect else 0

    print(f"\nDisagreement (Uncertainty):")
    print(f"  Correct:   {unc_c:.4f}")
    print(f"  Incorrect: {unc_i:.4f}")
    print(f"  Delta:    {unc_i - unc_c:.4f}")

    # Conflict cases
    conflict = [r for r in results if r["disc"] > 0.4]
    agreed_pos = [r for r in results if r["p_avg"] > 0.5 and r["disc"] < 0.1]
    agreed_neg = [r for r in results if r["p_avg"] < 0.5 and r["disc"] < 0.1]

    print(f"\nConflict (|V1-V2|>0.4): {len(conflict)}")
    print(f"Agreed Pos: {len(agreed_pos)}, Agreed Neg: {len(agreed_neg)}")

    # Plot
    print("\nGenerating gallery...")

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Ensemble Fusion - Master Gallery", fontsize=14, fontweight="bold")

    # Agreed positives
    plt.subplot(3, 4, 1)
    plt.text(0.5, 0.5, "AGREED POSITIVES\n(Low disagreement)", ha="center", va="center")
    plt.axis("off")

    for i, r in enumerate(
        sorted(agreed_pos, key=lambda x: x["p_avg"], reverse=True)[:3]
    ):
        plt.subplot(3, 4, i + 2)
        plt.imshow(r["axial"], cmap="gray")
        plt.title(
            f"P1:{r['p_v1']:.2f} P2:{r['p_v2']:.2f}\nAvg:{r['p_avg']:.2f} D:{r['disc']:.3f}",
            fontsize=8,
        )
        plt.axis("off")

    # Conflict cases
    plt.subplot(3, 4, 5)
    plt.text(
        0.5, 0.5, "CONFLICT CASES\n(|V1-V2|>0.4)", ha="center", va="center", color="red"
    )
    plt.axis("off")

    for i, r in enumerate(sorted(conflict, key=lambda x: x["disc"], reverse=True)[:3]):
        plt.subplot(3, 4, i + 6)
        plt.imshow(r["axial"], cmap="gray")
        plt.title(
            f"⚠️ P1:{r['p_v1']:.2f} P2:{r['p_v2']:.2f}\nAvg:{r['p_avg']:.2f} D:{r['disc']:.3f}",
            fontsize=8,
        )
        plt.axis("off")

    # Stats
    plt.subplot(1, 2, 2)
    plt.axis("off")
    txt = f"""
    ENSEMBLE STATISTICS
    ===================
    Accuracy:  {acc:.4f}
    Precision: {prec:.4f}
    Recall:   {rec:.4f}
    F1-Score: {f1:.4f}
    
    TP:{tp} TN:{tn} FP:{fp} FN:{fn}
    
    Disagreement:
    - Correct:   {unc_c:.4f}
    - Incorrect: {unc_i:.4f}
    - Delta:    {unc_i - unc_c:.4f}
    
    Conflict: {len(conflict)}
    """
    plt.text(
        0.1,
        0.9,
        txt,
        fontsize=10,
        family="monospace",
        va="top",
        transform=plt.gca().transAxes,
    )

    plt.tight_layout()
    plt.savefig("ensemble_master_gallery.png", dpi=80)
    plt.close()

    print("✓ Saved: ensemble_master_gallery.png")


if __name__ == "__main__":
    run()
