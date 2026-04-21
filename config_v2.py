#!/usr/bin/env python3
"""
Central Configuration for Lung Nodule CAD System V2.0 - Calibrated Framework
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
DROPOUT_PROB = 0.5
INTERNAL_DROPOUT_PROB = 0.2  # Dropout injected in ResNet blocks

# Temperature Scaling for calibrated uncertainty
TEMPERATURE = 1.5  # Logit softening factor

# MC Dropout for uncertainty estimation
MC_SAMPLES = 50  # High-fidelity variance passes

# Training Configuration
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
PATIENCE = 10
NUM_WORKERS = 0

# Class Balancing
NEGATIVES_PER_POSITIVE = 3  # Hard negatives per positive

# Uncertainty Configuration
UNCERTAINTY_THRESHOLD = 0.3  # High uncertainty threshold

# Paths
MODEL_DIR = "models"
BEST_MODEL_PATH = f"{MODEL_DIR}/calibrated_v2.pth"
TRAINING_CURVE_PATH = "output/training_curve_v2.png"
UNCERTAINTY_GRID_PATH = "output/uncertainty_grid_v2.png"

# Random Seed
SEED = 42
