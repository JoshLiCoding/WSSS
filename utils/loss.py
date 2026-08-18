import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy import ndimage


def load_class_weights(cfg, num_classes):
    """Per-class Potts weights: uniform, or the table in loss.class_weights_file when
    loss.per_class_weights is on."""
    if not cfg.loss.per_class_weights:
        return torch.full((num_classes,), float(cfg.loss.potts_weight))

    weights = OmegaConf.load(cfg.loss.class_weights_file).weights
    assert len(weights) == num_classes, \
        f"{cfg.loss.class_weights_file} holds {len(weights)} weights, expected {num_classes}"
    return torch.tensor(weights, dtype=torch.float32)


def calculate_pairwise_affinity(contour, dilation_size):
    """Dilate the contours by a `dilation_size`-pixel max filter and invert them into affinities."""
    contour_np = contour.cpu().numpy().astype(np.float32)
    dilated = np.stack([ndimage.maximum_filter(c, size=dilation_size) for c in contour_np])
    return torch.from_numpy(1.0 - dilated).to(device=contour.device, dtype=torch.float32)


def CollisionCrossEntropyLoss(logits, target_probs):
    """
    See "Soft Self-labeling and Potts Relaxations for Weakly-Supervised Segmentation" paper.
    CCE loss is robust to pseudo-label uncertainty without requiring hard labels.

    Args:
        logits: (B, C, H, W) tensor of logits from model
        target_probs: (B, C, H, W) tensor of target probabilities (pseudolabels)
    """
    # CCE: -ln(sum_k(σ_i^k * y_i^k)) for each pixel
    sum_probs_target = torch.sum(torch.softmax(logits, dim=1) * target_probs, dim=1)  # (B, H, W)
    return -torch.log(sum_probs_target + 1e-8).mean()


def CrossEntropyLoss(logits, target_probs, soft_targets=False):
    log_pred = F.log_softmax(logits, dim=1)
    if soft_targets:
        # CE = -sum_k target_k * log(pred_k)
        per_element = -torch.sum(target_probs * log_pred, dim=1)
    else:
        # Hard: one-hot from argmax, then CE = -log(pred[true_class])
        per_element = F.nll_loss(log_pred, target_probs.argmax(dim=1), reduction='none')
    return per_element.mean()


def KLDivergenceLoss(logits, target_probs, eps=1e-8):
    """
    Zero-avoiding KL divergence loss: KL(target || pred), with gradients w.r.t. logits.

    Args:
        logits: (B, C, ...) tensor of logits from the model (will be trained).
        target_probs: (B, C, ...) tensor of target probabilities (same shape as logits).
        eps: minimum value for target_probs to avoid log(0). Clamped values are
            renormalized so target remains a valid distribution over the class dim.
    """
    log_pred = F.log_softmax(logits, dim=1)

    # Zero-avoiding: clamp target so log(target) is finite, then renormalize to a distribution
    target_safe = target_probs.clamp(min=eps)
    target_safe = target_safe / target_safe.sum(dim=1, keepdim=True)

    kl = target_safe * (torch.log(target_safe) - log_pred)
    return kl.sum(dim=1).mean()


def get_unary_loss(unary_method):
    """Resolve loss.unary_method into the unary loss function it names."""
    losses = {
        'cce': CollisionCrossEntropyLoss,
        'kldiv': KLDivergenceLoss,
    }
    assert unary_method in losses, \
        f"unknown loss.unary_method '{unary_method}', expected one of {sorted(losses)}"
    return losses[unary_method]


def PottsLoss(logits, contours_x, contours_y, class_weights, dilation_size, use_color_diff=False):
    """
    Potts loss (quadratic relaxation), penalizing neighbouring pixels that disagree
    away from a contour.

    Args:
        logits: (B, C, H, W) tensor of logits
        contours_x: (B, H, W-1) tensor of horizontal contours
        contours_y: (B, H-1, W) tensor of vertical contours
        class_weights: (C,) per-class weights
        dilation_size: max-filter size used to widen the contours before inverting them
        use_color_diff: If True, contours are already affinities, so use them directly
    """
    w_x = contours_x if use_color_diff else calculate_pairwise_affinity(contours_x, dilation_size)
    w_y = contours_y if use_color_diff else calculate_pairwise_affinity(contours_y, dilation_size)

    prob = torch.softmax(logits, dim=1)
    weights = class_weights.view(1, -1, 1, 1)
    diff_x = prob - torch.roll(prob, -1, dims=3)
    diff_y = prob - torch.roll(prob, -1, dims=2)

    loss_x = torch.sum(0.5 * diff_x**2 * weights, dim=1)
    loss_y = torch.sum(0.5 * diff_y**2 * weights, dim=1)

    return (loss_x[:, :, :-1] * w_x).mean() + (loss_y[:, :-1, :] * w_y).mean()
