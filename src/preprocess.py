#!/usr/bin/env python3
"""
LIDC-IDRI preprocessing: extract positive (consensus nodule) and hard-negative
(lung parenchyma, away from any annotation) samples per scan.

Split-agnostic by design - every sample is tagged with patient_id/series_uid so
train/val/test membership is resolved at load time against
input/splits/patient_splits.json (see src/splits.py), not baked in here. This
means changing fold structure never requires re-running preprocessing.

Resumable: every output file is checked with os.path.exists before extraction,
so an interrupted run can simply be restarted.
"""

import argparse
import hashlib
import os
import shutil

import cv2
import numpy as np

import src.pylidc_compat  # noqa: F401  (must precede `import pylidc`)
import pylidc as pl
from pylidc.utils import consensus

from src import config

HU_LOWER = -1000
HU_UPPER = 400
CUBE_SIZE = config.IMG_SIZE


def apply_lung_window(volume):
    return np.clip(volume.astype(np.float32), HU_LOWER, HU_UPPER)


def rescale_to_unit(windowed):
    return (windowed - HU_LOWER) / (HU_UPPER - HU_LOWER)


def extract_cube(volume, center, size=CUBE_SIZE):
    c0, c1, c2 = center
    half = size // 2
    d0, d1, d2 = volume.shape

    lo0, hi0 = max(0, c0 - half), min(d0, c0 + half)
    lo1, hi1 = max(0, c1 - half), min(d1, c1 + half)
    lo2, hi2 = max(0, c2 - half), min(d2, c2 + half)

    cube = volume[lo0:hi0, lo1:hi1, lo2:hi2]

    pad0 = (max(0, half - c0), max(0, c0 + half - d0))
    pad1 = (max(0, half - c1), max(0, c1 + half - d1))
    pad2 = (max(0, half - c2), max(0, c2 + half - d2))

    if any(p[0] or p[1] for p in (pad0, pad1, pad2)):
        cube = np.pad(cube, (pad0, pad1, pad2), mode="constant", constant_values=0)

    return cube


def apply_clahe(slice_2d, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply((slice_2d * 255).astype(np.uint8)) / 255.0


def orthogonal_slices(cube):
    """Three orthogonal mid-planes through the cube. Axis0/1 are the DICOM
    row/col (in-plane) axes, axis2 is the true CT slice axis - "axial" here
    means the plane through axis2, matching the scanner's native slice plane;
    "coronal"/"sagittal" are the other two in-plane cuts by convention, not
    verified against per-scan patient orientation metadata."""
    m0, m1, m2 = (s // 2 for s in cube.shape)
    axial = cube[:, :, m2]
    coronal = cube[m0, :, :]
    sagittal = cube[:, m1, :]
    return {
        "axial": axial,
        "coronal": coronal,
        "sagittal": sagittal,
        "axial_clahe": apply_clahe(axial),
        "coronal_clahe": apply_clahe(coronal),
        "sagittal_clahe": apply_clahe(sagittal),
    }


def save_sample(path, cube, slices, **meta):
    if os.path.exists(path):
        return False
    np.savez_compressed(
        path,
        cube=cube,
        axial=slices["axial_clahe"],
        coronal=slices["coronal_clahe"],
        sagittal=slices["sagittal_clahe"],
        axial_raw=slices["axial"],
        coronal_raw=slices["coronal"],
        sagittal_raw=slices["sagittal"],
        **meta,
    )
    return True


def series_tag(series_uid):
    """Short unique tag for filenames. Every LIDC-IDRI series_uid shares the same
    ~12-char organizational OID prefix ("1.3.6.1.4.1."), so a naive prefix slice
    collides across different scans of the same patient - use a hash instead."""
    return hashlib.md5(series_uid.encode()).hexdigest()[:10]


def cluster_centroid(cluster):
    _, bbox, _ = consensus(cluster, clevel=0.5)
    return tuple(int((b.start + b.stop) // 2) for b in bbox)


def safe_center_range(dim, half):
    """Sampling range for a negative's center along one axis; degrades
    gracefully for scans thinner than CUBE_SIZE along a given axis."""
    if dim > 2 * half:
        return half, dim - half
    mid = dim // 2
    return mid, mid + 1


def process_scan(scan, output_dir, negatives_per_positive, rng):
    volume = scan.to_volume(verbose=False)
    windowed = rescale_to_unit(apply_lung_window(volume))
    d0, d1, d2 = volume.shape
    half = CUBE_SIZE // 2

    clusters = scan.cluster_annotations()
    centroids = []
    pos_count = 0

    for i, cluster in enumerate(clusters):
        centroid = cluster_centroid(cluster)
        centroids.append(centroid)

        out_path = os.path.join(
            output_dir, f"{scan.patient_id}_{series_tag(scan.series_instance_uid)}_c{i}_pos.npz"
        )
        if os.path.exists(out_path):
            pos_count += 1
            continue

        cube = extract_cube(windowed, centroid)
        slices = orthogonal_slices(cube)

        diameter = float(np.mean([a.diameter for a in cluster]))
        characteristics = {
            f"char_{name}": np.array([int(getattr(a, name)) for a in cluster])
            for name in (
                "subtlety",
                "margin",
                "lobulation",
                "spiculation",
                "texture",
                "calcification",
            )
        }

        saved = save_sample(
            out_path,
            cube,
            slices,
            label=1,
            patient_id=scan.patient_id,
            series_uid=scan.series_instance_uid,
            centroid_voxel=np.array(centroid),
            reader_count=len(cluster),
            malignancy_scores=np.array([int(a.malignancy) for a in cluster]),
            diameter_mm=diameter,
            **characteristics,
        )
        if saved:
            pos_count += 1

    n_negatives = negatives_per_positive * max(len(clusters), 1)
    neg_count = 0
    for neg_idx in range(n_negatives):
        out_path = os.path.join(
            output_dir, f"{scan.patient_id}_{series_tag(scan.series_instance_uid)}_n{neg_idx}_neg.npz"
        )
        if os.path.exists(out_path):
            neg_count += 1
            continue

        lo0, hi0 = safe_center_range(d0, half)
        lo1, hi1 = safe_center_range(d1, half)
        lo2, hi2 = safe_center_range(d2, half)

        for _ in range(50):
            c0 = int(rng.integers(lo0, hi0))
            c1 = int(rng.integers(lo1, hi1))
            c2 = int(rng.integers(lo2, hi2))

            # Two axis-aligned CUBE_SIZE cubes overlap iff every per-axis center
            # delta is < CUBE_SIZE (an AABB/Chebyshev test). A Euclidean centroid
            # distance threshold is a tighter (stricter-to-fail) test along
            # diagonals and silently under-rejects real overlaps there.
            too_close = any(
                abs(c0 - e0) < CUBE_SIZE and abs(c1 - e1) < CUBE_SIZE and abs(c2 - e2) < CUBE_SIZE
                for e0, e1, e2 in centroids
            )
            if too_close:
                continue

            cube = extract_cube(windowed, (c0, c1, c2))
            slices = orthogonal_slices(cube)
            saved = save_sample(
                out_path,
                cube,
                slices,
                label=0,
                patient_id=scan.patient_id,
                series_uid=scan.series_instance_uid,
                centroid_voxel=np.array((c0, c1, c2)),
            )
            if saved:
                neg_count += 1
            break
        else:
            print(
                f"  WARNING: {scan.patient_id} {scan.series_instance_uid[:20]}... "
                f"neg {neg_idx}: no valid placement found in 50 attempts, skipped"
            )

    return pos_count, neg_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only first N scans (testing)"
    )
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument(
        "--negatives-per-positive", type=int, default=config.NEGATIVES_PER_POSITIVE
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--delete-raw-after",
        action="store_true",
        help="Delete a patient's raw DICOM folder once fully processed (default: keep)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    scans = pl.query(pl.Scan).all()
    if args.limit:
        scans = scans[: args.limit]

    total_pos, total_neg, failed = 0, 0, 0
    for i, scan in enumerate(scans):
        try:
            pos, neg = process_scan(scan, args.output_dir, args.negatives_per_positive, rng)
            total_pos += pos
            total_neg += neg
            if args.delete_raw_after:
                patient_dir = os.path.join(config.RAW_DICOM_DIR, scan.patient_id)
                if os.path.isdir(patient_dir):
                    shutil.rmtree(patient_dir)
        except Exception as e:
            failed += 1
            print(f"FAILED {scan.patient_id}: {type(e).__name__}: {e}")

        if (i + 1) % 25 == 0:
            print(f"[{i + 1}/{len(scans)}] pos={total_pos} neg={total_neg} failed={failed}")

    print(f"\nDone. Positives: {total_pos}, Negatives: {total_neg}, Failed scans: {failed}")


if __name__ == "__main__":
    main()
