#!/usr/bin/env python3
"""Dataset indexing/loading for the preprocessed LIDC-IDRI samples. Filename-based
indexing (patient_id is always the first "_"-delimited token, and patient_id itself
never contains underscores) avoids opening all ~11k .npz files just to build an index.
"""

import glob
import json
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from src import config


def index_samples(data_dir=None):
    data_dir = data_dir or config.OUTPUT_DIR
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    index = []
    for f in files:
        base = os.path.basename(f)
        patient_id = base.split("_")[0]
        label = 1 if base.endswith("_pos.npz") else 0
        index.append({"path": f, "patient_id": patient_id, "label": label})
    return index


def load_splits(splits_file=None):
    splits_file = splits_file or config.SPLITS_FILE
    with open(splits_file) as f:
        return json.load(f)


def filter_by_patients(index, patient_ids):
    pid_set = set(patient_ids)
    return [s for s in index if s["patient_id"] in pid_set]


class NoduleDataset(Dataset):
    def __init__(self, samples, augment=True):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _transform(self):
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
        s = self.samples[idx]
        d = np.load(s["path"])

        axial = (d["axial"] * 255).astype(np.uint8)
        coronal = (d["coronal"] * 255).astype(np.uint8)
        sagittal = (d["sagittal"] * 255).astype(np.uint8)

        t = self._transform()
        return (
            t(axial),
            t(coronal),
            t(sagittal),
            torch.tensor(float(s["label"])),
        )
