#!/usr/bin/env python3
"""
Central Configuration for Lung Nodule CAD System
"""

# Data Configuration
SUBSET_LIST = [
    "subset0.zip",
    "subset1.zip",
    "subset2.zip",
    "subset3.zip",
    "subset4.zip",
    "subset5.zip",
]
IMG_SIZE = 64
OUTPUT_DIR = "preprocessed"
ANNOTATIONS_FILE = "annotations.csv"

# Model Configuration
BACKBONE = "resnet18"  # or "resnet34"
SHARED_BACKBONE = True
DROPOUT_PROB = 0.5

# Training Configuration
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
PATIENCE = 10
NUM_WORKERS = 0

# Class Balancing
NEGATIVES_PER_POSITIVE = 3  # Hard negatives per positive

# Uncertainty Configuration
UNCERTAINTY_PASSES = 50  # MC Dropout passes for inference
UNCERTAINTY_THRESHOLD = 0.3  # High uncertainty threshold

# Paths
MODEL_DIR = "models"
BEST_MODEL_PATH = f"{MODEL_DIR}/best_model.pth"
TRAINING_CURVE_PATH = "training_curve.png"
UNCERTAINTY_GRID_PATH = "uncertainty_grid.png"

# Random Seed
SEED = 42
