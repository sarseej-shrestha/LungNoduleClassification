#!/usr/bin/env python3
"""
Grad-CAM saliency sanity check: verifies the model attends to the actual nodule
region, not background, per Zhang et al. (2022) - a good-accuracy classifier can
still be "right for the wrong reasons" if it learns background shortcuts instead
of the nodule itself. See the literature review in the project vault.

Positive samples are cube-centered on the consensus nodule centroid by
construction (src/preprocess.py), so for positives we have a cheap quantitative
proxy even without a stored segmentation mask: how far is the Grad-CAM peak from
the cube center? Saves qualitative overlay images too, for manual inspection.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import config
from src.dataset import NoduleDataset, filter_by_patients, index_samples, load_splits
from src.model import load_model


class GradCAM:
    """Hooks the shared backbone's layer4. The backbone is called 3x per forward
    pass (axial, coronal, sagittal, in that order - see NoduleClassifier.forward).
    Forward hooks fire in call order, which is an unambiguous PyTorch guarantee.
    Backward-hook firing order for a module reused multiple times in one forward
    pass is NOT part of the public API contract (it depends on autograd engine
    scheduling internals) - relying on "reverse of forward order" is folklore, not
    a guarantee, and was verified to actually be unreliable for this exact setup
    before being replaced with this approach: each view's gradient is obtained via
    a separate torch.autograd.grad(..., inputs=activation) call, which asks
    autograd for the gradient w.r.t. a *specific* captured tensor - unambiguous
    regardless of internal scheduling."""

    VIEWS = ("axial", "coronal", "sagittal")

    def __init__(self, model):
        self.model = model
        self.activations = []
        model.backbone.layer4.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inp, output):
        self.activations.append(output)

    def generate(self, axial, coronal, sagittal):
        self.activations.clear()

        logit = self.model(axial, coronal, sagittal)
        # forward call order in NoduleClassifier.forward is axial, coronal,
        # sagittal - so self.activations[0,1,2] correspond to those views.
        assert len(self.activations) == 3, f"expected 3 activations, got {len(self.activations)}"

        cams = {}
        input_size = axial.shape[-1]
        for i, view in enumerate(self.VIEWS):
            act = self.activations[i]  # (1, C, H, W), retains graph
            (grad,) = torch.autograd.grad(logit, act, retain_graph=True)
            act = act[0].detach()  # (C, H, W)
            grad = grad[0]  # (C, H, W)
            weights = grad.mean(dim=(1, 2))
            cam = torch.relu((weights[:, None, None] * act).sum(0))
            cam = cam / (cam.max() + 1e-8)
            cam_np = cam.cpu().numpy()
            cams[view] = _resize_cam(cam_np, input_size)

        return cams, torch.sigmoid(logit).item()


def _resize_cam(cam, size):
    import cv2

    return cv2.resize(cam, (size, size))


def cam_peak_offset_from_center(cam):
    """Voxel distance from the CAM's peak-activation pixel to the array center."""
    h, w = cam.shape
    center = np.array([h / 2, w / 2])
    peak = np.array(np.unravel_index(np.argmax(cam), cam.shape))
    return float(np.linalg.norm(peak - center))


def run_sanity_check(ckpt_path, samples, device, n_samples=20, output_dir="output/gradcam"):
    os.makedirs(output_dir, exist_ok=True)
    model, checkpoint = load_model(ckpt_path, strict=False)
    model.to(device)
    model.eval()

    grad_cam = GradCAM(model)

    rng = np.random.default_rng(config.SEED)
    positives = [s for s in samples if s["label"] == 1]
    chosen = rng.choice(positives, size=min(n_samples, len(positives)), replace=False)

    offsets = []
    dataset = NoduleDataset(list(chosen), augment=False)

    for i, s in enumerate(chosen):
        axial, coronal, sagittal, label = dataset[i]
        axial = axial.unsqueeze(0).to(device).requires_grad_(False)
        coronal = coronal.unsqueeze(0).to(device)
        sagittal = sagittal.unsqueeze(0).to(device)

        cams, prob = grad_cam.generate(axial, coronal, sagittal)
        offset = cam_peak_offset_from_center(cams["axial"])
        offsets.append(offset)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, view in zip(axes, GradCAM.VIEWS):
            img = {"axial": axial, "coronal": coronal, "sagittal": sagittal}[view][
                0, 0
            ].cpu().numpy()
            ax.imshow(img, cmap="gray")
            ax.imshow(cams[view], cmap="jet", alpha=0.4)
            ax.set_title(view)
            ax.axis("off")
        fig.suptitle(f"{s['patient_id']} pred={prob:.2f} (label=1)")
        fig.savefig(os.path.join(output_dir, f"{s['patient_id']}_{i}.png"), dpi=80)
        plt.close(fig)

    offsets = np.array(offsets)
    print(f"\nAxial CAM peak offset from cube center (voxels), n={len(offsets)}:")
    print(f"  mean={offsets.mean():.1f}  median={np.median(offsets):.1f}  max={offsets.max():.1f}")
    print(f"  (cube is 64x64; center-quadrant radius ~16, half-cube radius ~32)")
    print(f"Saved overlay images to {output_dir}/")

    return offsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt_path = args.checkpoint or f"{config.MODEL_DIR}/fold{args.fold}_best.pth"

    index = index_samples()
    splits = load_splits()
    samples = filter_by_patients(index, splits["folds"][args.fold]["val"])

    run_sanity_check(ckpt_path, samples, device, n_samples=args.n_samples)


if __name__ == "__main__":
    main()
