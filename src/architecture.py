#!/usr/bin/env python3
"""
Lung Nodule CAD - Modular Architecture
"""

import torch
import torch.nn as nn
from torchvision import models
from src import config_v1 as config


class MCDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training or not self.p:
            return nn.functional.dropout(x, p=self.p, training=self.training)
        mask = (torch.rand(x.size(0), x.size(1), device=x.device) > self.p).float() / (
            1 - self.p
        )
        return x * mask.unsqueeze(0)


def get_backbone(name):
    if name.lower() == "resnet18":
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif name.lower() == "resnet34":
        return models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    raise ValueError(f"Unknown backbone: {name}")


class ResNet18Branch(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.backbone.bn1 = nn.BatchNorm2d(64)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)


class MultiViewNet(nn.Module):
    def __init__(self, dropout_prob=None):
        dropout_prob = dropout_prob or config.DROPOUT_PROB
        super().__init__()
        self.backbone = ResNet18Branch(1)
        self.dropout1 = MCDropout(p=dropout_prob)
        self.fc1 = nn.Linear(512 * 3, 256)
        self.relu = nn.ReLU()
        self.dropout2 = MCDropout(p=dropout_prob)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, axial, coronal, sagittal):
        f1 = self.backbone(axial)
        f2 = self.backbone(coronal)
        f3 = self.backbone(sagittal)
        fused = torch.cat([f1, f2, f3], dim=1)

        x = self.dropout1(fused)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x.squeeze(1)

    def predict_with_uncertainty(self, axial, coronal, sagittal, num_samples=None):
        if num_samples is None:
            num_samples = config.UNCERTAINTY_PASSES

        self.eval()
        outputs = []

        for _ in range(num_samples):
            with torch.no_grad():
                output = self.forward(axial, coronal, sagittal)
                outputs.append(torch.sigmoid(output))

        stacked = torch.stack(outputs, dim=0)
        return torch.mean(stacked).item(), torch.std(stacked).item()


def create_model():
    return MultiViewNet(dropout_prob=config.DROPOUT_PROB)


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


def predict_with_uncertainty(model, axial, coronal, sagittal, device, num_samples=None):
    if num_samples is None:
        num_samples = config.UNCERTAINTY_PASSES

    model.eval()
    outputs = []

    for _ in range(num_samples):
        with torch.no_grad():
            output = model(axial, coronal, sagittal)
            outputs.append(torch.sigmoid(output))

    stacked = torch.stack(outputs, dim=0)
    return torch.mean(stacked).item(), torch.std(stacked).item()
