#!/usr/bin/env python3
"""
Central Configuration for Lung Nodule CAD System
"""

# Data Configuration
SUBSET_LIST = [
    "input/subset0.zip",
    "input/subset1.zip",
    "input/subset2.zip",
    "input/subset3.zip",
    "input/subset4.zip",
    "input/subset5.zip",
]
IMG_SIZE = 64
OUTPUT_DIR = "input/preprocessed"
ANNOTATIONS_FILE = "input/annotations.csv"

# Model Configuration
BACKBONE = "resnet18"
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
NEGATIVES_PER_POSITIVE = 3

# Uncertainty Configuration
UNCERTAINTY_PASSES = 50
UNCERTAINTY_THRESHOLD = 0.3

# Paths
MODEL_DIR = "models"
BEST_MODEL_PATH = f"{MODEL_DIR}/best_uncertainty_model.pth"
TRAINING_CURVE_PATH = "output/training_curve.png"
UNCERTAINTY_GRID_PATH = "output/uncertainty_grid.png"

# Random Seed
SEED = 42
