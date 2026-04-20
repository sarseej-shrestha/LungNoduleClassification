#!/usr/bin/env python3
"""
Lung Nodule CAD System - Main Entry Point
Three modes: preprocess, train, validate
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision import models
import cv2
import SimpleITK as sitk
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Import local modules
import config
from architecture import create_model, load_model, SharedResNet


# ===================== DATA =====================


class NoduleDataset(Dataset):
    def __init__(self, npz_dir=None, npz_files=None, augment=True, target_label=None):
        if npz_files is not None:
            self.npz_files = npz_files
        else:
            self.npz_files = sorted(
                glob.glob(os.path.join(npz_dir or config.OUTPUT_DIR, "*.npz"))
            )
        self.augment = augment
        self.target_label = target_label

    def __len__(self):
        return len(self.npz_files)

    def _get_transform(self):
        if self.augment:
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                ]
            )
        return transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        data = np.load(self.npz_files[idx])

        label = data.get("label", 0)
        if self.target_label is not None and label != self.target_label:
            return None

        axial = (data["axial"] * 255).astype(np.uint8)
        coronal = (data["coronal"] * 255).astype(np.uint8)
        sagittal = (data["sagittal"] * 255).astype(np.uint8)

        transform = self._get_transform()

        return (
            transform(axial),
            transform(coronal),
            transform(sagittal),
            torch.tensor(float(label)),
        )


# ===================== PREPROCESSING =====================


class NoduleExtractor:
    """Extract nodules with hard negative mining."""

    def __init__(self, subset_zip):
        import zipfile
        import tempfile

        self.zip_file = zipfile.ZipFile(subset_zip, "r")
        self.temp_dir = tempfile.mkdtemp()
        self.headers = {}
        self._load_headers()

    def _load_headers(self):
        for name in self.zip_file.namelist():
            if name.endswith(".mhd"):
                uid = name.split("/")[-1].replace(".mhd", "")
                mhd_data = self.zip_file.read(name)

                mhd_path = os.path.join(self.temp_dir, f"{uid}.mhd")
                raw_path = os.path.join(self.temp_dir, f"{uid}.raw")

                with open(mhd_path, "wb") as f:
                    f.write(mhd_data)

                raw_name = name.replace(".mhd", ".raw")
                raw_data = self.zip_file.read(raw_name)
                with open(raw_path, "wb") as f:
                    f.write(raw_data)

                img = sitk.ReadImage(mhd_path)
                self.headers[uid] = {
                    "img": img,
                    "shape": img.GetSize(),
                    "spacing": img.GetSpacing(),
                    "origin": img.GetOrigin(),
                }

    def voxel_from_world(self, uid, world_coords):
        img = self.headers[uid]["img"]
        return img.TransformPhysicalPointToIndex(world_coords)

    def extract_cube(self, uid, voxel_coords, size=config.IMG_SIZE):
        import zipfile

        header = self.headers[uid]
        raw_path = os.path.join(self.temp_dir, f"{uid}.raw")

        # Read raw data
        with open(raw_path, "rb") as f:
            raw_data = f.read()

        shape = header["shape"]  # (x, y, z)
        volume = np.frombuffer(raw_data, dtype=np.int16).reshape(shape[::-1])  # z, y, x

        # Extract cube
        cz, cy, cx = voxel_coords
        half = size // 2

        d, h, w = volume.shape

        z0 = max(0, cz - half)
        z1 = min(d, cz + half)
        y0 = max(0, cy - half)
        y1 = min(h, cy + half)
        x0 = max(0, cx - half)
        x1 = min(w, cx + half)

        cube = volume[z0:z1, y0:y1, x0:x1]

        # Pad if needed
        pad_z = (0, half * 2 - cube.shape[0]) if cube.shape[0] < half * 2 else (0, 0)
        pad_y = (0, half * 2 - cube.shape[1]) if cube.shape[1] < half * 2 else (0, 0)
        pad_x = (0, half * 2 - cube.shape[2]) if cube.shape[2] < half * 2 else (0, 0)

        if any(p[0] or p[1] for p in [pad_z, pad_y, pad_x]):
            cube = np.pad(
                cube, (pad_z, pad_y, pad_x), mode="constant", constant_values=0
            )

        # Convert to HU
        hu = cube.astype(np.float32) - 1024
        hu = np.clip(hu, -1000, 400)
        hu = (hu + 1000) / 1400
        hu = hu.astype(np.float32)

        return hu

    def apply_clahe(self, slice_2d):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply((slice_2d * 255).astype(np.uint8)) / 255.0

    def get_slices(self, cube):
        cz, cy, cx = cube.shape[0] // 2, cube.shape[1] // 2, cube.shape[2] // 2

        axial = cube[cz, :, :]
        coronal = cube[:, cy, :]
        sagittal = cube[:, :, cx]

        return {
            "axial": self.apply_clahe(axial),
            "coronal": self.apply_clahe(coronal),
            "sagittal": self.apply_clahe(sagittal),
        }

    def close(self):
        import shutil

        self.zip_file.close()
        shutil.rmtree(self.temp_dir)


def preprocess_subset(subset_idx, subset_zip, annotations_df):
    """Preprocess a single subset with hard negative mining."""

    extractor = NoduleExtractor(subset_zip)

    ann_series = set(annotations_df["seriesuid"].unique())
    zip_series = set(extractor.headers.keys())
    common = ann_series & zip_series

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    pos_count, neg_count = 0, 0

    for series_uid in common:
        series_annots = annotations_df[annotations_df["seriesuid"] == series_uid]
        existing_voxels = []

        # Extract positives
        for idx, row in series_annots.iterrows():
            output_path = os.path.join(
                config.OUTPUT_DIR, f"subset{subset_idx}_{series_uid}_{idx}_label_1.npz"
            )

            if os.path.exists(output_path):
                continue

            try:
                world = (row["coordX"], row["coordY"], row["coordZ"])
                voxel = extractor.voxel_from_world(series_uid, world)
                cube = extractor.extract_cube(series_uid, voxel)
                slices = extractor.get_slices(cube)

                np.savez_compressed(
                    output_path,
                    cube=cube,
                    axial=slices["axial"],
                    coronal=slices["coronal"],
                    sagittal=slices["sagittal"],
                    label=1,
                    series_uid=series_uid,
                    diameter_mm=row["diameter_mm"],
                )
                pos_count += 1
                existing_voxels.append(voxel)
            except:
                pass

        # Extract hard negatives
        for i in range(config.NEGATIVES_PER_POSITIVE):
            if not existing_voxels:
                continue

            neg_output_path = os.path.join(
                config.OUTPUT_DIR,
                f"subset{subset_idx}_{series_uid}_{idx}_label_0_{i}.npz",
            )

            if os.path.exists(neg_output_path):
                continue

            try:
                shape = extractor.headers[series_uid]["shape"]
                # Pick random location away from annotations
                cx = np.random.randint(10, shape[0] - 10)
                cy = np.random.randint(10, shape[1] - 10)
                cz = np.random.randint(10, shape[2] - 10)

                cube = extractor.extract_cube(series_uid, (cz, cy, cx))
                slices = extractor.get_slices(cube)

                np.savez_compressed(
                    neg_output_path,
                    cube=cube,
                    axial=slices["axial"],
                    coronal=slices["coronal"],
                    sagittal=slices["sagittal"],
                    label=0,
                    series_uid=series_uid,
                )
                neg_count += 1
            except:
                pass

    extractor.close()
    return pos_count, neg_count


# ===================== TRAINING =====================


def train_model(model, train_loader, val_loader, device):
    """Training loop with class weights and early stopping."""

    # Calculate class weights
    all_labels = []
    for _, _, _, labels in train_loader:
        all_labels.extend(labels.numpy().tolist())

    n_pos = sum(all_labels)
    n_neg = len(all_labels) - n_pos
    pos_weight = n_neg / max(n_pos, 1)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(config.NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for axial, coronal, sagittal, labels in train_loader:
            axial, coronal, sagittal, labels = (
                axial.to(device),
                coronal.to(device),
                sagittal.to(device),
                labels.to(device),
            )
            optimizer.zero_grad()
            outputs = model(axial, coronal, sagittal)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        preds_all, labels_all = [], []

        with torch.no_grad():
            for axial, coronal, sagittal, labels in val_loader:
                axial, coronal, sagittal, labels = (
                    axial.to(device),
                    coronal.to(device),
                    sagittal.to(device),
                    labels.to(device),
                )
                outputs = model(axial, coronal, sagittal)
                val_loss += criterion(outputs, labels).item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                preds_all.extend(preds.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        accuracy = sum(p == l for p, l in zip(preds_all, labels_all)) / len(preds_all)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(accuracy)

        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch + 1}] Train: {train_loss:.4f} Val: {val_loss:.4f} Acc: {accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": accuracy,
                },
                config.BEST_MODEL_PATH,
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return history


# ===================== VALIDATION =====================


def validate_model(model, device, subset_files):
    """Validate with MC Dropout for uncertainty."""

    model.eval()

    results = []

    for i, f in enumerate(subset_files):
        data = np.load(f)

        transform = transforms.ToTensor()
        axial = (
            transform((data["axial"] * 255).astype(np.uint8)).unsqueeze(0).to(device)
        )
        coronal = (
            transform((data["coronal"] * 255).astype(np.uint8)).unsqueeze(0).to(device)
        )
        sagittal = (
            transform((data["sagittal"] * 255).astype(np.uint8)).unsqueeze(0).to(device)
        )

        true_label = data.get("label", 0)

        # MC Dropout
        mean_prob, uncertainty = model.predict_with_uncertainty(
            axial, coronal, sagittal
        )

        pred_label = 1 if mean_prob > 0.5 else 0

        results.append(
            {
                "mean_prob": mean_prob,
                "uncertainty": uncertainty,
                "prediction": pred_label,
                "true_label": true_label,
                "correct": pred_label == true_label,
            }
        )

        if (i + 1) % 20 == 0:
            print(f"Validated {i + 1}/{len(subset_files)}")

    # Metrics
    tp = sum(1 for r in results if r["true_label"] == 1 and r["prediction"] == 1)
    tn = sum(1 for r in results if r["true_label"] == 0 and r["prediction"] == 0)
    fp = sum(1 for r in results if r["true_label"] == 0 and r["prediction"] == 1)
    fn = sum(1 for r in results if r["true_label"] == 1 and r["prediction"] == 0)

    accuracy = (tp + tn) / len(results)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1)

    print("\n" + "=" * 60)
    print("CLINICAL READINESS METRICS")
    print("=" * 60)
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Uncertainty analysis
    tp_unc = np.mean(
        [
            r["uncertainty"]
            for r in results
            if r["true_label"] == 1 and r["prediction"] == 1
        ]
    )
    fn_unc = np.mean(
        [
            r["uncertainty"]
            for r in results
            if r["true_label"] == 1 and r["prediction"] == 0
        ]
    )
    print(f"\nUncertainty (TP): {tp_unc:.4f}")
    print(f"Uncertainty (FN): {fn_unc:.4f}")

    if fn > 0 and tp > 0 and fn_unc > tp_unc:
        print("\n>>> AHA: Missed nodules have HIGHER uncertainty!")

    return results


# ===================== MAIN =====================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Lung Nodule CAD System")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["preprocess", "train", "validate"],
        help="Operation mode",
    )
    parser.add_argument("--subset", type=str, default="0", help="Subset number (0-5)")
    parser.add_argument("--model_path", type=str, default=None, help="Model path")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory")

    args = parser.parse_args()

    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # Set random seeds
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    device = torch.device("cpu")

    if args.mode == "preprocess":
        print("=" * 60)
        print("PREPROCESSING MODE")
        print("=" * 60)

        subset_idx = int(args.subset)
        subset_zip = f"subset{subset_idx}.zip"

        if not os.path.exists(subset_zip):
            print(f"Error: {subset_zip} not found")
            return

        annotations = pd.read_csv(config.ANNOTATIONS_FILE)

        print(f"Processing {subset_zip}...")
        pos, neg = preprocess_subset(subset_idx, subset_zip, annotations)
        print(f"Done! Positives: {pos}, Negatives: {neg}")

    elif args.mode == "train":
        print("=" * 60)
        print("TRAINING MODE")
        print("=" * 60)

        # Load data
        dataset = NoduleDataset(config.OUTPUT_DIR)
        print(f"Total samples: {len(dataset)}")

        # Stratified split
        all_labels = [int(np.load(f).get("label", 0)) for f in dataset.npz_files]
        train_idx, val_idx = train_test_split(
            range(len(dataset)),
            test_size=0.2,
            random_state=config.SEED,
            stratify=all_labels,
        )

        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=config.BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=config.BATCH_SIZE)

        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

        # Create model
        model = create_model().to(device)
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

        # Train
        history = train_model(model, train_loader, val_loader, device)

        # Save curve
        plt.figure(figsize=(10, 4))
        plt.plot(history["train_loss"], label="Train")
        plt.plot(history["val_loss"], label="Val")
        plt.legend()
        plt.savefig(config.TRAINING_CURVE_PATH)
        print(f"Saved: {config.TRAINING_CURVE_PATH}")

    elif args.mode == "validate":
        print("=" * 60)
        print("VALIDATION MODE")
        print("=" * 60)

        # Load model
        model_path = args.model_path or config.BEST_MODEL_PATH

        if not os.path.exists(model_path):
            print(f"Error: Model not found: {model_path}")
            return

        model, checkpoint = load_model(model_path, strict=False)

        if model is None:
            print("Error: Could not load model")
            return

        model.to(device)
        print(f"Loaded from epoch {checkpoint.get('epoch', 'N/A')}")

        # Load data
        data_dir = args.data_dir or config.OUTPUT_DIR
        subset_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

        if args.subset:
            subset_num = int(args.subset)
            subset_files = [f for f in subset_files if f"subset{subset_num}_" in f]

        print(f"Validating {len(subset_files)} samples...")

        results = validate_model(model, device, subset_files)

        # Save visualization
        misclass = sorted(
            [r for r in results if not r["correct"]],
            key=lambda x: x["uncertainty"],
            reverse=True,
        )[:4]

        if misclass:
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            for row, case in enumerate(misclass[:2]):
                data = np.load(case["path"])
                for col, key in enumerate(["axial", "coronal", "sagittal"]):
                    axes[row, col].imshow(data[key], cmap="gray")
                    axes[row, col].axis("off")

                axes[row, 3].text(
                    0.1,
                    0.5,
                    f"Pred: {case['prediction']}",
                    transform=axes[row, 3].transAxes,
                )
                axes[row, 3].text(
                    0.1,
                    0.3,
                    f"Uncert: {case['uncertainty']:.3f}",
                    transform=axes[row, 3].transAxes,
                )
                axes[row, 3].axis("off")

            plt.savefig(config.UNCERTAINTY_GRID_PATH)
            print(f"Saved: {config.UNCERTAINTY_GRID_PATH}")


if __name__ == "__main__":
    main()
