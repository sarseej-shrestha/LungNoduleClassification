#!/usr/bin/env python3
"""
Lung Nodule CAD Preprocessing Pipeline
- Multi-Subset Support: subset0.zip through subset5.zip
- Hard Negative Mining: 3 negatives per positive (chest wall + random lung)
- Target: ~2000 total samples
"""

import zipfile
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
import tempfile
import random


HU_RESCALING_INTERCEPT = -1024
HU_LOWER = -1000
HU_UPPER = 400
CUBE_SIZE = 64
OUTPUT_DIR = "preprocessed"
NEGATIVES_PER_POSITIVE = 3


class NoduleExtractor:
    def __init__(self, subset_zip_path: str):
        self.subset_zip_path = subset_zip_path
        self.subset_zip = zipfile.ZipFile(subset_zip_path, "r")
        self.temp_dir = tempfile.mkdtemp()
        self.extracted_files = {}
        self._extract_and_load_headers()

    def _extract_and_load_headers(self):
        for name in self.subset_zip.namelist():
            if name.endswith(".mhd"):
                series_uid = os.path.basename(name).replace(".mhd", "")
                mhd_content = self.subset_zip.read(name)
                raw_name = name.replace(".mhd", ".raw")
                raw_content = self.subset_zip.read(raw_name)

                mhd_path = os.path.join(self.temp_dir, os.path.basename(name))
                raw_path = os.path.join(self.temp_dir, os.path.basename(raw_name))

                with open(mhd_path, "wb") as f:
                    f.write(mhd_content)
                with open(raw_path, "wb") as f:
                    f.write(raw_content)

                self.extracted_files[series_uid] = (mhd_path, raw_path)

    def world_to_voxel(
        self, series_uid: str, world_coords: Tuple[float, float, float]
    ) -> Tuple[int, int, int]:
        mhd_path = self.extracted_files[series_uid][0]
        img = sitk.ReadImage(mhd_path)
        itk_index = img.TransformPhysicalPointToIndex(world_coords)
        x, y, z = itk_index
        return (z, y, x)

    def load_raw_volume(self, series_uid: str) -> np.ndarray:
        mhd_path = self.extracted_files[series_uid][0]
        img = sitk.ReadImage(mhd_path)
        arr = sitk.GetArrayFromImage(img)
        return arr

    def normalize_to_hu(self, volume: np.ndarray) -> np.ndarray:
        hu = volume.astype(np.float32) + HU_RESCALING_INTERCEPT
        return hu

    def apply_lung_window(self, hu_volume: np.ndarray) -> np.ndarray:
        clipped = np.clip(hu_volume, HU_LOWER, HU_UPPER)
        return clipped

    def rescale_to_unit(self, windowed_volume: np.ndarray) -> np.ndarray:
        normalized = (windowed_volume - HU_LOWER) / (HU_UPPER - HU_LOWER)
        return normalized

    def extract_cube(
        self,
        volume: np.ndarray,
        center_voxel: Tuple[int, int, int],
        cube_size: int = CUBE_SIZE,
    ) -> np.ndarray:
        cz, cy, cx = center_voxel
        half = cube_size // 2

        d, h, w = volume.shape

        pad_z_low = max(0, cz - half)
        pad_z_high = min(d, cz + half)
        pad_y_low = max(0, cy - half)
        pad_y_high = min(h, cy + half)
        pad_x_low = max(0, cx - half)
        pad_x_high = min(w, cx + half)

        cube = volume[pad_z_low:pad_z_high, pad_y_low:pad_y_high, pad_x_low:pad_x_high]

        pad_z0 = max(0, half - cz)
        pad_z1 = max(0, cz + half - d)
        pad_y0 = max(0, half - cy)
        pad_y1 = max(0, cy + half - h)
        pad_x0 = max(0, half - cx)
        pad_x1 = max(0, cx + half - w)

        if pad_z0 or pad_z1 or pad_y0 or pad_y1 or pad_x0 or pad_x1:
            cube = np.pad(
                cube,
                ((pad_z0, pad_z1), (pad_y0, pad_y1), (pad_x0, pad_x1)),
                mode="constant",
                constant_values=0,
            )

        return cube

    def apply_clahe(
        self,
        slice_2d: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply((slice_2d * 255).astype(np.uint8)) / 255.0

    def extractOrthogonalSlices(self, cube: np.ndarray) -> Dict[str, np.ndarray]:
        d, h, w = cube.shape
        mid_z = d // 2
        mid_y = h // 2
        mid_x = w // 2

        axial = cube[mid_z, :, :]
        coronal = cube[:, mid_y, :]
        sagittal = cube[:, :, mid_x]

        axial_clahe = self.apply_clahe(axial)
        coronal_clahe = self.apply_clahe(coronal)
        sagittal_clahe = self.apply_clahe(sagittal)

        return {
            "axial": axial,
            "coronal": coronal,
            "sagittal": sagittal,
            "axial_clahe": axial_clahe,
            "coronal_clahe": coronal_clahe,
            "sagittal_clahe": sagittal_clahe,
        }

    def generate_negatives(
        self,
        series_uid: str,
        existing_voxels: List[Tuple[int, int, int]],
        num_negatives: int = NEGATIVES_PER_POSITIVE,
    ) -> List[Optional[Tuple[int, int, int]]]:
        volume = self.load_raw_volume(series_uid)
        d, h, w = volume.shape

        half = CUBE_SIZE // 2
        negatives = []

        for neg_type in range(num_negatives):
            for _ in range(50):
                if neg_type == 0:
                    cx = np.random.randint(half, w // 4)
                    cy = np.random.randint(half, h - half)
                    cz = np.random.randint(half, d - half)
                else:
                    cx = np.random.randint(half, w - half)
                    cy = np.random.randint(half, h - half)
                    cz = np.random.randint(half, d - half)

                is_too_close = False
                for ex_vox in existing_voxels:
                    dist = np.sqrt(
                        (cx - ex_vox[2]) ** 2
                        + (cy - ex_vox[1]) ** 2
                        + (cz - ex_vox[0]) ** 2
                    )
                    if dist < CUBE_SIZE:
                        is_too_close = True
                        break

                if not is_too_close:
                    negatives.append((cz, cy, cx))
                    break
            else:
                negatives.append(None)

        return negatives

    def process_nodule(
        self,
        series_uid: str,
        world_coords: Tuple[float, float, float],
        diameter_mm: float,
        label: int = 1,
    ) -> Optional[Dict]:
        if series_uid not in self.extracted_files:
            return None

        voxel_coords = self.world_to_voxel(series_uid, world_coords)

        raw_volume = self.load_raw_volume(series_uid)
        hu_volume = self.normalize_to_hu(raw_volume)
        windowed_volume = self.apply_lung_window(hu_volume)
        normalized_volume = self.rescale_to_unit(windowed_volume)

        cube = self.extract_cube(normalized_volume, voxel_coords)
        slices = self.extractOrthogonalSlices(cube)

        return {
            "series_uid": series_uid,
            "world_coords": world_coords,
            "voxel_coords": voxel_coords,
            "diameter_mm": diameter_mm,
            "label": label,
            "cube": cube,
            "slices": slices,
        }

    def close(self):
        import shutil

        self.subset_zip.close()
        shutil.rmtree(self.temp_dir)


def load_annotations(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def process_subset(
    subset_zip_path: str, annotations_df: pd.DataFrame, subset_idx: int
) -> Tuple[int, int, int, int]:
    """Process a single subset with hard negative mining."""

    print(f"\nProcessing {subset_zip_path}...")
    extractor = NoduleExtractor(subset_zip_path)
    print(f"Loaded {len(extractor.extracted_files)} scans")

    annot_series = set(annotations_df["seriesuid"].unique())
    zip_series = set(extractor.extracted_files.keys())
    common_series = annot_series & zip_series
    print(f"Matching series: {len(common_series)}")

    positive_count = 0
    negative_count = 0
    failed_count = 0
    skipped_count = 0

    for series_uid in common_series:
        series_annots = annotations_df[annotations_df["seriesuid"] == series_uid]

        existing_voxels = []

        for idx, row in series_annots.iterrows():
            output_path = os.path.join(
                OUTPUT_DIR, f"subset{subset_idx}_{series_uid}_{idx}_label_1.npz"
            )

            if os.path.exists(output_path):
                skipped_count += 1
                continue

            world_coords = (row["coordX"], row["coordY"], row["coordZ"])
            diameter = row["diameter_mm"]

            result = extractor.process_nodule(
                series_uid, world_coords, diameter, label=1
            )

            if result is not None:
                slices = result["slices"]
                np.savez_compressed(
                    output_path,
                    cube=result["cube"],
                    axial=slices["axial_clahe"],
                    coronal=slices["coronal_clahe"],
                    sagittal=slices["sagittal_clahe"],
                    axial_raw=slices["axial"],
                    coronal_raw=slices["coronal"],
                    sagittal_raw=slices["sagittal"],
                    series_uid=series_uid,
                    voxel_coords=np.array(result["voxel_coords"]),
                    diameter_mm=diameter,
                    label=1,
                    subset_idx=subset_idx,
                )
                positive_count += 1
                existing_voxels.append(result["voxel_coords"])
            else:
                failed_count += 1

        negatives = extractor.generate_negatives(
            series_uid, existing_voxels, NEGATIVES_PER_POSITIVE
        )

        for neg_idx, neg_voxel in enumerate(negatives):
            if neg_voxel is None:
                continue

            neg_output_path = os.path.join(
                OUTPUT_DIR,
                f"subset{subset_idx}_{series_uid}_{idx}_label_0_{neg_idx}.npz",
            )

            if os.path.exists(neg_output_path):
                continue

            raw_volume = extractor.load_raw_volume(series_uid)
            hu_volume = extractor.normalize_to_hu(raw_volume)
            windowed_volume = extractor.apply_lung_window(hu_volume)
            normalized_volume = extractor.rescale_to_unit(windowed_volume)

            cube = extractor.extract_cube(normalized_volume, neg_voxel)
            slices = extractor.extractOrthogonalSlices(cube)

            np.savez_compressed(
                neg_output_path,
                cube=cube,
                axial=slices["axial_clahe"],
                coronal=slices["coronal_clahe"],
                sagittal=slices["sagittal_clahe"],
                axial_raw=slices["axial"],
                coronal_raw=slices["coronal"],
                sagittal_raw=slices["sagittal"],
                series_uid=series_uid,
                voxel_coords=np.array(neg_voxel),
                diameter_mm=0.0,
                label=0,
                subset_idx=subset_idx,
            )
            negative_count += 1
            existing_voxels.append(neg_voxel)

    extractor.close()
    print(
        f"Subset {subset_idx}: Positive: {positive_count}, Negative: {negative_count}, Failed: {failed_count}, Skipped: {skipped_count}"
    )
    return positive_count, negative_count, failed_count, skipped_count


def main():
    subset_zips = [f"subset{i}.zip" for i in range(6)]

    print("Loading annotations...")
    annotations = load_annotations("annotations.csv")
    print(f"Found {len(annotations)} annotations")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_positive = 0
    total_negative = 0
    total_failed = 0
    total_skipped = 0

    for idx, subset_zip in enumerate(subset_zips):
        if not os.path.exists(subset_zip):
            print(f"Warning: {subset_zip} not found, skipping...")
            continue

        pos, neg, fail, skip = process_subset(subset_zip, annotations, idx)
        total_positive += pos
        total_negative += neg
        total_failed += fail
        total_skipped += skip

    print(
        f"\n=== Total: Positive: {total_positive}, Negative: {total_negative}, Failed: {total_failed}, Skipped: {total_skipped} ==="
    )
    print(
        f"Target: ~{total_positive * 4} total samples (1:{NEGATIVES_PER_POSITIVE} ratio)"
    )


if __name__ == "__main__":
    main()
