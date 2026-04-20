#!/usr/bin/env python3
"""
Lung Nodule CAD - Modular Architecture
Supports ResNet-18 and ResNet-34 backbones with shared weight option
"""

import torch
import torch.nn as nn
from torchvision import models
import config


def get_backbone(backbone_name="resnet18"):
    """Factory function to get the specified backbone."""

    if backbone_name.lower() == "resnet18":
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif backbone_name.lower() == "resnet34":
        return models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")


class ResNetBackbone(nn.Module):
    """Single ResNet backbone for feature extraction."""

    def __init__(self, backbone_name="resnet18", in_channels=1):
        super().__init__()
        self.backbone = get_backbone(backbone_name)

        # Modify first conv layer for grayscale input
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Initialize with 3-channel weights
        with torch.no_grad():
            self.backbone.conv1.weight = nn.Parameter(
                original_conv.weight.repeat(1, 3, 1, 1) / 3
            )

        # Remove classification head
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)


class SharedResNet(nn.Module):
    """Multi-view ResNet with shared or separate backbone."""

    def __init__(self, backbone_name=None, dropout_prob=None, shared=True):
        """Initialize the multi-view network.

        Args:
            backbone_name: "resnet18" or "resnet34"
            dropout_prob: Dropout probability
            shared: Use single shared backbone (True) or separate (False)
        """
        # Use config defaults
        if backbone_name is None:
            backbone_name = config.BACKBONE
        if dropout_prob is None:
            dropout_prob = config.DROPOUT_PROB

        self.shared = shared

        if shared:
            # Single backbone processes all 3 views
            self.backbone = ResNetBackbone(backbone_name)
            feature_dim = 512  # Output features from single backbone
        else:
            # 3 separate backbones
            self.axial_branch = ResNetBackbone(backbone_name)
            self.coronal_branch = ResNetBackbone(backbone_name)
            self.sagittal_branch = ResNetBackbone(backbone_name)
            feature_dim = 512 * 3

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(256, 1),
        )

    def forward(self, axial, coronal, sagittal):
        if self.shared:
            feat_axial = self.backbone(axial)
            feat_coronal = self.backbone(coronal)
            feat_sagittal = self.backbone(sagittal)
        else:
            feat_axial = self.axial_branch(axial)
            feat_coronal = self.coronal_branch(coronal)
            feat_sagittal = self.sagittal_branch(sagittal)

        fused = torch.cat([feat_axial, feat_coronal, feat_sagittal], dim=1)
        return self.classifier(fused).squeeze(1)

    def predict_with_uncertainty(self, axial, coronal, sagittal, num_samples=None):
        """MC Dropout for uncertainty estimation.

        Args:
            axial, coronal, sagittal: Input tensors
            num_samples: Number of forward passes

        Returns:
            mean_prob: Average prediction probability
            uncertainty: Standard deviation (uncertainty score)
        """
        if num_samples is None:
            num_samples = config.UNCERTAINTY_PASSES

        self.eval()

        outputs = []
        for _ in range(num_samples):
            with torch.no_grad():
                # Forward in eval mode but with dropout active
                output = self.forward(axial, coronal, sagittal)
                probs = torch.sigmoid(output)
                outputs.append(probs)

        stacked = torch.stack(outputs, dim=0)
        mean_prob = torch.mean(stacked, dim=0)
        uncertainty = torch.std(stacked, dim=0)

        return mean_prob.item(), uncertainty.item()


def create_model():
    """Factory function to create model from config."""
    return SharedResNet(
        backbone_name=config.BACKBONE,
        dropout_prob=config.DROPOUT_PROB,
        shared=config.SHARED_BACKBONE,
    )


# Convenience function for loading
def load_model(path, strict=True):
    """Load model with proper handling."""
    import os

    os.makedirs(config.MODEL_DIR, exist_ok=True)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    # Try to load checkpoint
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Create model and load state
        model = create_model()

        # Try loading with different strictness
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        except RuntimeError:
            # Try non-strict loading as fallback
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        return model, checkpoint

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
