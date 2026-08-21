#!/usr/bin/env python3
"""
Lung Nodule CAD - 3-branch ResNet-18 with Deep Dropout + MC Dropout + Temperature Scaling
"""

import torch
import torch.nn as nn
from torchvision import models
from src import config


class MCDropout(nn.Module):
    """Custom dropout that stays active during eval() mode for MC Dropout."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training or not self.p:
            return nn.functional.dropout(x, p=self.p, training=self.training)
        # In eval mode, generate random dropout mask
        # Handle both 1D and 2D tensors
        if x.dim() == 2:
            mask = (
                torch.rand(x.size(0), x.size(1), device=x.device) > self.p
            ).float() / (1 - self.p)
        else:
            mask = (torch.rand_like(x) > self.p).float() / (1 - self.p)
        return x * mask


class DropoutBasicBlock(nn.Module):
    """ResNet BasicBlock with dropout injected after the second convolution."""

    def __init__(self, original_block, dropout_prob=0.2):
        super().__init__()
        self.conv1 = original_block.conv1
        self.bn1 = original_block.bn1
        self.relu = original_block.relu
        self.conv2 = original_block.conv2
        self.bn2 = original_block.bn2
        self.downsample = original_block.downsample
        self.stride = original_block.stride

        # Inject dropout after second convolution
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply dropout after second conv (before residual addition)
        out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def inject_dropout_into_resnet(resnet, dropout_prob=0.2):
    """Inject dropout into all BasicBlocks of a ResNet."""
    for i, block in enumerate(resnet.layer1):
        resnet.layer1[i] = DropoutBasicBlock(block, dropout_prob)
    for i, block in enumerate(resnet.layer2):
        resnet.layer2[i] = DropoutBasicBlock(block, dropout_prob)
    for i, block in enumerate(resnet.layer3):
        resnet.layer3[i] = DropoutBasicBlock(block, dropout_prob)
    for i, block in enumerate(resnet.layer4):
        resnet.layer4[i] = DropoutBasicBlock(block, dropout_prob)
    return resnet


class ResNet18WithDeepDropout(nn.Module):
    """ResNet-18 with dropout injected into all BasicBlocks."""

    def __init__(self, in_channels=1, internal_dropout_prob=0.2):
        super().__init__()

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Replace first conv for grayscale input
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.backbone.bn1 = nn.BatchNorm2d(64)

        self.layer1 = nn.Sequential(
            *[
                DropoutBasicBlock(block, internal_dropout_prob)
                for block in self.backbone.layer1
            ]
        )
        self.layer2 = nn.Sequential(
            *[
                DropoutBasicBlock(block, internal_dropout_prob)
                for block in self.backbone.layer2
            ]
        )
        self.layer3 = nn.Sequential(
            *[
                DropoutBasicBlock(block, internal_dropout_prob)
                for block in self.backbone.layer3
            ]
        )
        self.layer4 = nn.Sequential(
            *[
                DropoutBasicBlock(block, internal_dropout_prob)
                for block in self.backbone.layer4
            ]
        )

        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        self.avgpool = self.backbone.avgpool
        self.fc = nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class NoduleClassifier(nn.Module):
    """3-branch (axial/coronal/sagittal) MultiView Network with Deep Dropout,
    MC Dropout, and Temperature Scaling."""

    def __init__(self, dropout_prob=None, internal_dropout_prob=None):
        dropout_prob = dropout_prob or config.DROPOUT_PROB
        internal_dropout_prob = internal_dropout_prob or config.INTERNAL_DROPOUT_PROB

        super().__init__()

        self.backbone = ResNet18WithDeepDropout(
            in_channels=1, internal_dropout_prob=internal_dropout_prob
        )

        self.mc_dropout = MCDropout(p=dropout_prob)
        self.fc1 = nn.Linear(512 * 3, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)

        self.temperature = config.TEMPERATURE
        self.mc_samples = config.MC_SAMPLES

    def forward(self, axial, coronal, sagittal):
        f1 = self.backbone(axial)
        f2 = self.backbone(coronal)
        f3 = self.backbone(sagittal)

        fused = torch.cat([f1, f2, f3], dim=1)

        x = self.mc_dropout(fused)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.mc_dropout(x)
        x = self.fc2(x)

        return x.squeeze(1)

    def predict_with_uncertainty(self, axial, coronal, sagittal):
        """MC Dropout inference with temperature scaling."""
        self.eval()

        logits_list = []
        for _ in range(self.mc_samples):
            with torch.no_grad():
                logit = self.forward(axial, coronal, sagittal)
                logits_list.append(logit)

        logits_stack = torch.stack(logits_list, dim=0)

        scaled_logits = logits_stack / self.temperature
        probs = torch.sigmoid(scaled_logits)

        mean_prob = torch.mean(probs).item()
        uncertainty = torch.std(probs).item()

        return mean_prob, uncertainty

    def predict_with_temperature(self, axial, coronal, sagittal, temperature=None):
        """Predict with custom temperature."""
        temperature = temperature or self.temperature

        self.eval()
        with torch.no_grad():
            logit = self.forward(axial, coronal, sagittal)
            prob = torch.sigmoid(logit / temperature).item()

        return prob


def create_model():
    return NoduleClassifier(
        dropout_prob=config.DROPOUT_PROB,
        internal_dropout_prob=config.INTERNAL_DROPOUT_PROB,
    )


def load_model(path, strict=False):
    import os

    os.makedirs(config.MODEL_DIR, exist_ok=True)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = create_model()
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        return model, checkpoint
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def apply_temperature_scaling(logits, temperature=None):
    """Apply temperature scaling to logits."""
    temperature = temperature or config.TEMPERATURE
    return torch.sigmoid(logits / temperature)
