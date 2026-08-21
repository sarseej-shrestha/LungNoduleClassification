#!/usr/bin/env python3
"""
Evaluation suite: FROC (sensitivity at fixed false-positives/patient operating
points), patient-level bootstrapped confidence intervals, and calibration
(expected calibration error + reliability diagram data).

Defaults to a fold's own validation patients, NOT the locked test set - per the
project's standing rule, the test set is touched exactly once, at the very end
(Phase 5), not during iterative development. Pass --split test explicitly (and
mean it) to evaluate against the real held-out set.
"""

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
from src.dataset import NoduleDataset, filter_by_patients, index_samples, load_splits
from src.model import load_model

FROC_FP_PER_SCAN_POINTS = [0.125, 0.25, 0.5, 1, 2, 4, 8]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_mc_inference(model, samples, device, mc_samples=None):
    """Returns per-sample mean_prob, uncertainty, label, patient_id."""
    mc_samples = mc_samples or config.MC_SAMPLES
    loader = DataLoader(
        NoduleDataset(samples, augment=False), batch_size=1, shuffle=False
    )

    model.eval()
    results = []
    for i, (axial, coronal, sagittal, label) in enumerate(loader):
        axial, coronal, sagittal = axial.to(device), coronal.to(device), sagittal.to(device)
        mean_prob, uncertainty = model.predict_with_uncertainty(axial, coronal, sagittal)
        results.append(
            {
                "mean_prob": mean_prob,
                "uncertainty": uncertainty,
                "label": int(label.item()),
                "patient_id": samples[i]["patient_id"],
            }
        )
        if (i + 1) % 200 == 0:
            print(f"  inferred {i + 1}/{len(samples)}")
    return results


def compute_froc(results, fp_points=FROC_FP_PER_SCAN_POINTS):
    """LUNA16-style FROC: at each probability threshold, sensitivity = fraction
    of true positives detected; FP/patient = false positives / number of distinct
    patients. Returns sorted (fp_per_patient, sensitivity) points plus sensitivity
    interpolated at each of fp_points."""
    n_positives = sum(1 for r in results if r["label"] == 1)
    patient_ids = sorted(set(r["patient_id"] for r in results))
    n_patients = len(patient_ids)

    thresholds = sorted(set(r["mean_prob"] for r in results), reverse=True)
    curve = []
    for t in thresholds:
        tp = sum(1 for r in results if r["label"] == 1 and r["mean_prob"] >= t)
        fp = sum(1 for r in results if r["label"] == 0 and r["mean_prob"] >= t)
        sensitivity = tp / n_positives if n_positives else 0.0
        fp_per_patient = fp / n_patients if n_patients else 0.0
        curve.append((fp_per_patient, sensitivity))

    curve.sort()
    fps = np.array([c[0] for c in curve])
    sens = np.array([c[1] for c in curve])

    interpolated = {}
    for fp_point in fp_points:
        idx = np.searchsorted(fps, fp_point)
        if idx == 0:
            interpolated[fp_point] = float(sens[0]) if len(sens) else 0.0
        elif idx >= len(fps):
            interpolated[fp_point] = float(sens[-1]) if len(sens) else 0.0
        else:
            interpolated[fp_point] = float(sens[idx - 1])

    return {"curve": curve, "operating_points": interpolated}


def bootstrap_ci(results, statistic_fn, n_bootstrap=1000, ci=0.95, seed=None):
    """Patient-level bootstrap: resample patients WITH replacement (not
    individual samples), keeping all of a resampled patient's samples together,
    since samples from the same patient aren't independent."""
    rng = np.random.default_rng(seed if seed is not None else config.SEED)
    by_patient = {}
    for r in results:
        by_patient.setdefault(r["patient_id"], []).append(r)
    patient_ids = list(by_patient.keys())

    stats = []
    for _ in range(n_bootstrap):
        sampled_ids = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        resampled = [r for pid in sampled_ids for r in by_patient[pid]]
        stats.append(statistic_fn(resampled))

    stats = np.array(stats)
    alpha = (1 - ci) / 2
    return {
        "point": statistic_fn(results),
        "lower": float(np.percentile(stats, 100 * alpha)),
        "upper": float(np.percentile(stats, 100 * (1 - alpha))),
        "n_bootstrap": n_bootstrap,
    }


def accuracy_stat(results):
    correct = sum(1 for r in results if (r["mean_prob"] > 0.5) == bool(r["label"]))
    return correct / len(results) if results else 0.0


def sensitivity_stat(results):
    positives = [r for r in results if r["label"] == 1]
    if not positives:
        return 0.0
    return sum(1 for r in positives if r["mean_prob"] > 0.5) / len(positives)


def expected_calibration_error(results, n_bins=10):
    probs = np.array([r["mean_prob"] for r in results])
    labels = np.array([r["label"] for r in results])

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if mask.sum() == 0:
            bins.append({"range": [float(lo), float(hi)], "count": 0})
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        weight = mask.sum() / len(probs)
        ece += weight * abs(bin_conf - bin_acc)
        bins.append(
            {
                "range": [float(lo), float(hi)],
                "count": int(mask.sum()),
                "confidence": float(bin_conf),
                "accuracy": float(bin_acc),
            }
        )

    return {"ece": float(ece), "bins": bins}


def evaluate_checkpoint(ckpt_path, samples, device):
    model, checkpoint = load_model(ckpt_path, strict=False)
    model.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")

    results = run_mc_inference(model, samples, device)

    froc = compute_froc(results)
    acc_ci = bootstrap_ci(results, accuracy_stat)
    sens_ci = bootstrap_ci(results, sensitivity_stat)
    calibration = expected_calibration_error(results)

    correct = [r for r in results if (r["mean_prob"] > 0.5) == bool(r["label"])]
    incorrect = [r for r in results if (r["mean_prob"] > 0.5) != bool(r["label"])]
    unc_correct = float(np.mean([r["uncertainty"] for r in correct])) if correct else 0.0
    unc_incorrect = (
        float(np.mean([r["uncertainty"] for r in incorrect])) if incorrect else 0.0
    )

    return {
        "n_samples": len(results),
        "froc_operating_points": froc["operating_points"],
        "accuracy": acc_ci,
        "sensitivity": sens_ci,
        "calibration": calibration,
        "uncertainty_correct": unc_correct,
        "uncertainty_incorrect": unc_incorrect,
        "uncertainty_gap": unc_incorrect - unc_correct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="'val' = this fold's own validation patients (default, safe for "
        "iterative development). 'test' = the LOCKED held-out set - only use "
        "this once, at the very end.",
    )
    args = parser.parse_args()

    if args.split == "test":
        confirm = input(
            "You are about to evaluate on the LOCKED TEST SET. This should happen "
            "exactly once, at the very end of the project. Type 'yes' to proceed: "
        )
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return

    device = get_device()
    ckpt_path = args.checkpoint or f"{config.MODEL_DIR}/fold{args.fold}_best.pth"

    index = index_samples()
    splits = load_splits()

    if args.split == "test":
        samples = filter_by_patients(index, splits["test"])
    else:
        samples = filter_by_patients(index, splits["folds"][args.fold]["val"])

    print(f"Evaluating {ckpt_path} on {len(samples)} samples ({args.split})")
    report = evaluate_checkpoint(ckpt_path, samples, device)

    print(json.dumps(report, indent=2))

    out_path = f"output/eval_fold{args.fold}_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
