#!/usr/bin/env python3
"""
Central Configuration for Lung Nodule CAD System - LIDC-IDRI Rebuild
"""

# Data locations (raw DICOM lives outside the repo, never git-tracked)
RAW_DICOM_DIR = "/Users/userselu/Downloads/LIDC-IDRI-raw/LIDC-IDRI"
OUTPUT_DIR = "input/preprocessed"
SPLITS_FILE = "input/splits/patient_splits.json"

IMG_SIZE = 64

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

# Patient-level splitting (see src/splits.py)
TEST_HOLDOUT_FRACTION = 0.175  # ~15-20% of patients, locked away, touched once
N_FOLDS = 5

# Paths
MODEL_DIR = "models"
BEST_MODEL_PATH = f"{MODEL_DIR}/calibrated_v2.pth"
TRAINING_CURVE_PATH = "output/training_curve_v2.png"
UNCERTAINTY_GRID_PATH = "output/uncertainty_grid_v2.png"

# Random Seed
SEED = 42
