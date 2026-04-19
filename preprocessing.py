#!/usr/bin/env python3
"""
Lung Nodule CAD Preprocessing Pipeline
- Coordinate Mapping: SimpleITK TransformPhysicalPointToIndex
- Axis Ordering: SimpleITK (x,y,z) -> NumPy (z,y,x)
- CLAHE Enhancement: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
- Windowing: -1000 to 400 HU
- Visualization: With crosshair verification
"""

import zipfile
import io
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


HU_RESCALING_INTERCEPT = -1024
HU_LOWER = -1000
HU_UPPER = 400
CUBE_SIZE = 64
OUTPUT_DIR = "preprocessed"


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
            "center_voxel": (mid_x, mid_y, mid_z),
        }

    def process_nodule(
        self,
        series_uid: str,
        world_coords: Tuple[float, float, float],
        diameter_mm: float,
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
            "cube": cube,
            "slices": slices,
        }

    def close(self):
        import shutil

        self.subset_zip.close()
        shutil.rmtree(self.temp_dir)


def visualize_nodule(
    result: Dict, fig_size: Tuple[int, int] = (12, 8), save_path: str = None
):
    slices = result["slices"]
    center = slices["center_voxel"]
    h, w = slices["axial"].shape

    fig, axes = plt.subplots(2, 3, figsize=fig_size)

    titles = ["Axial", "Coronal", "Sagittal"]
    raw_slices = [slices["axial"], slices["coronal"], slices["sagittal"]]
    clahe_slices = [
        slices["axial_clahe"],
        slices["coronal_clahe"],
        slices["sagittal_clahe"],
    ]

    for row_idx, (slice_list, label) in enumerate(
        [(raw_slices, "Before CLAHE"), (clahe_slices, "After CLAHE")]
    ):
        for ax_idx, (ax, img, title) in enumerate(
            zip(axes[row_idx], slice_list, titles)
        ):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)

            cx, cy = w // 2, h // 2
            ax.axvline(x=cx, color="red", linewidth=1, linestyle="--", alpha=0.7)
            ax.axhline(y=cy, color="red", linewidth=1, linestyle="--", alpha=0.7)

            ax.set_title(f"{label}: {title}")
            ax.axis("off")

    series_uid = result["series_uid"][:30]
    fig.suptitle(
        f"Nodule: {series_uid}...\nVoxel: {result['voxel_coords']}, Diameter: {result['diameter_mm']:.1f}mm"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)
        plt.close()
    return fig


def load_annotations(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def test_single_nodule(
    annotations_path: str = "annotations.csv", subset_zip_path: str = "subset0.zip"
):
    print("Loading annotations...")
    df = load_annotations(annotations_path)
    print(f"Found {len(df)} annotations")

    print("Initializing extractor...")
    extractor = NoduleExtractor(subset_zip_path)
    print(f"Loaded {len(extractor.extracted_files)} scans")

    annot_series = set(df["seriesuid"].unique())
    zip_series = set(extractor.extracted_files.keys())
    common_series = annot_series & zip_series
    print(f"Matching series: {len(common_series)}")

    if not common_series:
        print("No matching series found!")
        return

    first_series = list(common_series)[0]
    first_row = df[df["seriesuid"] == first_series].iloc[0]
    world_coords = (first_row["coordX"], first_row["coordY"], first_row["coordZ"])
    diameter = first_row["diameter_mm"]

    print(f"Processing test nodule: {first_series[:50]}...")
    print(f"World coords: {world_coords}")
    result = extractor.process_nodule(first_series, world_coords, diameter)

    if result:
        print(f"ITK Voxel (z,y,x): {result['voxel_coords']}")
        print(f"Cube shape: {result['cube'].shape}")

        visualize_nodule(result, save_path="sample_nodule.png")
        print("Saved sample_nodule.png")

    extractor.close()
    return result


def main():
    print("Loading annotations...")
    annotations = load_annotations("annotations.csv")
    print(f"Found {len(annotations)} annotations")

    print("Initializing extractor...")
    extractor = NoduleExtractor("subset0.zip")
    print(f"Loaded {len(extractor.extracted_files)} scans")

    annot_series = set(annotations["seriesuid"].unique())
    zip_series = set(extractor.extracted_files.keys())
    common_series = annot_series & zip_series
    print(f"Matching series: {len(common_series)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    processed_count = 0
    failed_count = 0

    for series_uid in common_series:
        series_annots = annotations[annotations["seriesuid"] == series_uid]

        for idx, row in series_annots.iterrows():
            world_coords = (row["coordX"], row["coordY"], row["coordZ"])
            diameter = row["diameter_mm"]

            result = extractor.process_nodule(series_uid, world_coords, diameter)

            if result is not None:
                slices = result["slices"]
                output_path = os.path.join(OUTPUT_DIR, f"{series_uid}_{idx}.npz")
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
                )
                processed_count += 1
            else:
                failed_count += 1

    extractor.close()
    print(f"Done! Processed: {processed_count}, Failed: {failed_count}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_single_nodule()
    else:
        main()
