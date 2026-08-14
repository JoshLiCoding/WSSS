"""Turning cached DINOv2 similarities into soft pseudo-labels, and colouring them."""

import numpy as np
import torch

from utils.dataset import cmap


def soft_pseudolabels(sim, present, temperature):
    """Same chain as ~/wsss: temperature softmax over the classes an image contains,
    per-channel min-max scaling over space, then renormalize every pixel onto the
    probability simplex.

    Args:
        sim: (B, C, H, W) raw patch cosine similarities scattered into class space, at the
            resolution the loss is computed at. Channels of absent classes are ignored.
        present: (B, C) bool mask of the classes each image contains (channel 0 = background).
        temperature: softmax temperature.

    Returns:
        (B, C, H, W) soft pseudo-labels; channels of absent classes are exactly zero.
    """
    mask = present[:, :, None, None]
    logits = torch.where(mask, sim / temperature, torch.full_like(sim, float('-inf')))
    probs = logits.softmax(dim=1)

    # Min-max per channel over space. Absent channels are all-zero, so they stay zero.
    # low = probs.amin(dim=(2, 3), keepdim=True)
    # high = probs.amax(dim=(2, 3), keepdim=True)
    # probs = (probs - low) / (high - low + 1e-8)
    # probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)

    # Min-max has nothing to scale when only background is present; fall back to a hard label.
    single = present.sum(dim=1) == 1
    if single.any():
        probs[single] = mask[single].to(probs.dtype).expand_as(probs[single])
    return probs


def soft_probability_vis(probs):
    """probs: (C, H, W) probabilities. Blends the VOC colour of each class by its
    probability, giving a soft-label image."""
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    colors = cmap(normalized=True)[:probs.shape[0]]  # (C, 3)
    vis = np.tensordot(probs, colors, axes=([0], [0]))  # (H, W, 3)
    return np.clip(vis * 255, 0, 255).astype(np.uint8)
