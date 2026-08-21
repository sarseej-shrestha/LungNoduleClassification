#!/usr/bin/env python3
"""
Patient-level stratified splitting for LIDC-IDRI.

Generates a fixed split file once: a locked held-out test set (touched only for
the final reported numbers) plus stratified K folds over the remainder for
development/model-selection. Splitting is by patient_id, never by scan or nodule,
since some patients have multiple scans - splitting below the patient level risks
leakage (per the literature review, the single most common flaw in this field).

Stratification tier per patient, based on the highest-malignancy consensus nodule
across all of that patient's scans:
  - "none"          : no annotated nodule clusters at all
  - "benign"        : highest avg malignancy < 3
  - "indeterminate" : highest avg malignancy == 3
  - "malignant"     : highest avg malignancy > 3

Run once; the output JSON is the artifact everything else consumes. Re-running
overwrites it, so treat it as an intentional action, not part of routine iteration.
"""

import json
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

import src.pylidc_compat  # noqa: F401  (must precede `import pylidc`)
import pylidc as pl

from src import config


def _malignancy_tier(avg_malignancy):
    if avg_malignancy is None:
        return "none"
    if avg_malignancy < 3:
        return "benign"
    if avg_malignancy == 3:
        return "indeterminate"
    return "malignant"


def summarize_patients():
    """One pass over every scan: cluster annotations, record per-patient the
    highest-malignancy consensus nodule and total nodule count. Cheap - just
    distance-based clustering + malignancy averaging, no volume/mask loading."""
    scans = pl.query(pl.Scan).all()
    print(f"Querying {len(scans)} scans across all patients...")

    per_patient = {}  # patient_id -> {"max_malignancy": float|None, "n_clusters": int}

    for i, scan in enumerate(scans):
        pid = scan.patient_id
        entry = per_patient.setdefault(pid, {"max_malignancy": None, "n_clusters": 0})

        clusters = scan.cluster_annotations()
        entry["n_clusters"] += len(clusters)

        for cluster in clusters:
            avg_mal = float(np.mean([a.malignancy for a in cluster]))
            if entry["max_malignancy"] is None or avg_mal > entry["max_malignancy"]:
                entry["max_malignancy"] = avg_mal

        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(scans)} scans")

    return per_patient


def build_splits(seed=None, test_fraction=None, n_folds=None):
    seed = seed if seed is not None else config.SEED
    test_fraction = test_fraction if test_fraction is not None else config.TEST_HOLDOUT_FRACTION
    n_folds = n_folds if n_folds is not None else config.N_FOLDS

    per_patient = summarize_patients()
    patient_ids = sorted(per_patient.keys())
    tiers = [_malignancy_tier(per_patient[pid]["max_malignancy"]) for pid in patient_ids]

    tier_counts = {t: tiers.count(t) for t in set(tiers)}
    print(f"\nPatient tiers: {tier_counts}")

    dev_ids, test_ids, dev_tiers, _ = train_test_split(
        patient_ids,
        tiers,
        test_size=test_fraction,
        random_state=seed,
        stratify=tiers,
    )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    dev_ids_arr = np.array(dev_ids)
    dev_tiers_arr = np.array(dev_tiers)
    for train_idx, val_idx in skf.split(dev_ids_arr, dev_tiers_arr):
        folds.append(
            {
                "train": sorted(dev_ids_arr[train_idx].tolist()),
                "val": sorted(dev_ids_arr[val_idx].tolist()),
            }
        )

    result = {
        "seed": seed,
        "test_fraction": test_fraction,
        "n_folds": n_folds,
        "tier_counts": tier_counts,
        "test": sorted(test_ids),
        "folds": folds,
        "patient_summary": per_patient,
    }

    os.makedirs(os.path.dirname(config.SPLITS_FILE), exist_ok=True)
    with open(config.SPLITS_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote splits to {config.SPLITS_FILE}")
    print(f"  Held-out test: {len(test_ids)} patients")
    print(f"  Dev set: {len(dev_ids)} patients across {n_folds} folds")
    for i, fold in enumerate(folds):
        print(f"    fold {i}: train={len(fold['train'])}, val={len(fold['val'])}")

    return result


if __name__ == "__main__":
    build_splits()
