import torch
import torch.nn.functional as F
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

def calculate_pairwise_affinity(contour):
    device = contour.device
    contour_cpu = contour.cpu().numpy().astype(np.float32)

    dilated_contour = np.zeros_like(contour_cpu)
    for i in range(contour_cpu.shape[0]):
        dilated_contour[i] = ndimage.maximum_filter(contour_cpu[i], size=5)
    w_np = 1.0 - dilated_contour

    w = torch.from_numpy(w_np).to(device=device, dtype=torch.float32)
    return w

def CollisionCrossEntropyLoss(logits, target_probs):
    """
    See "Soft Self-labeling and Potts Relaxations for Weakly-Supervised Segmentation" paper.
    CCE loss is robust to pseudo-label uncertainty without requiring hard labels.

    Args:
        logits: (B, C, H, W) tensor of logits from model
        target_probs: (B, C, H, W) tensor of target probabilities (pseudolabels)
    """
    probs = torch.softmax(logits, dim=1)

    # Compute sum_k(σ_i^k * y_i^k) for each pixel
    sum_probs_target = torch.sum(probs * target_probs, dim=1)  # (B, H, W)

    # Standard CCE: -ln(sum_k(σ_i^k * y_i^k))
    per_pixel_loss = -torch.log(sum_probs_target + 1e-8)  # (B, H, W)

    return per_pixel_loss.mean()


def CrossEntropyLoss(logits, target_probs, soft_targets=False):
    log_pred = F.log_softmax(logits, dim=1)
    if soft_targets:
        # CE = -sum_k target_k * log(pred_k)
        per_element = -torch.sum(target_probs * log_pred, dim=1)
    else:
        # Hard: one-hot from argmax, then CE = -log(pred[true_class])
        target_class = target_probs.argmax(dim=1)  # (B, ...)
        per_element = F.nll_loss(log_pred, target_class, reduction='none')
    return per_element.mean()

def KLDivergenceLoss(logits, target_probs, eps=1e-8):
    """
    Zero-avoiding KL divergence loss: KL(target || pred), with gradients w.r.t. logits.

    Args:
        logits: (B, C, ...) tensor of logits from the model (will be trained).
        target_probs: (B, C, ...) tensor of target probabilities (same shape as logits).
        eps: minimum value for target_probs to avoid log(0). Clamped values are
            renormalized so target remains a valid distribution over the class dim.

    Returns:
        Scalar loss: mean over batch and spatial dimensions of sum over classes
        of target_k * (log(target_k) - log(pred_k)).
    """
    # Numerically stable: log(pred) from logits without forming pred explicitly
    log_pred = F.log_softmax(logits, dim=1)

    # Zero-avoiding: clamp target so log(target) is always finite
    target_safe = target_probs.clamp(min=eps)
    # Renormalize over class dim so we still have a distribution
    target_safe = target_safe / target_safe.sum(dim=1, keepdim=True)

    # KL(target || pred) = sum_k target_k * (log(target_k) - log(pred_k))
    kl = target_safe * (torch.log(target_safe) - log_pred)
    loss = kl.sum(dim=1).mean()
    return loss


def PottsLoss(logits, contours_x, contours_y):
    """
    Quadratic Potts loss with contour-based weighting.

    Args:
        logits: (B, C, H, W) tensor of logits
        contours_x: (B, H, W-1) tensor of horizontal contours
        contours_y: (B, H-1, W) tensor of vertical contours
        affinity: 'dilate' (default) = soft gaussian dilation; 'maximum' = hard max-filter dilation
    """
    w_x = calculate_pairwise_affinity(contours_x)
    w_y = calculate_pairwise_affinity(contours_y)

    prob = torch.softmax(logits, dim=1)
    num_classes = prob.shape[1]

    device = prob.device
    class_weights = torch.full((num_classes,), 500.0, device=device, dtype=prob.dtype)

    # voc weights from kl divergence
    class_weights[0]  = 0.0  # background (max mIoU: 0.8929)
    # class_weights[1]  = 1000.0 # aeroplane (max mIoU: 0.8335)
    # class_weights[2]  = 0.0    # bicycle (max mIoU: 0.4772)
    # class_weights[3]  = 700.0  # bird (max mIoU: 0.8787)
    # class_weights[4]  = 300.0  # boat (max mIoU: 0.7342)
    # class_weights[5]  = 100.0  # bottle (max mIoU: 0.5836)
    # class_weights[6]  = 500.0  # bus (max mIoU: 0.9396)
    # class_weights[7]  = 400.0  # car (max mIoU: 0.8419)
    # class_weights[8]  = 300.0  # cat (max mIoU: 0.9360)
    # class_weights[9]  = 200.0  # chair (max mIoU: 0.4530)
    # class_weights[10] = 200.0  # cow (max mIoU: 0.9249)
    # class_weights[11] = 0.0    # diningtable (max mIoU: 0.4489)
    # class_weights[12] = 800.0  # dog (max mIoU: 0.9039)
    # class_weights[13] = 700.0  # horse (max mIoU: 0.8704)
    # class_weights[14] = 400.0  # motorbike (max mIoU: 0.8093)
    # class_weights[15] = 300.0  # person (max mIoU: 0.8482)
    # class_weights[16] = 100.0  # potted plant (max mIoU: 0.5221)
    # class_weights[17] = 400.0  # sheep (max mIoU: 0.9096)
    # class_weights[18] = 100.0  # sofa (max mIoU: 0.5234)
    # class_weights[19] = 700.0  # train (max mIoU: 0.8674)
    # class_weights[20] = 100.0  # tv/monitor (max mIoU: 0.5426)

    class_weights = class_weights.view(1, num_classes, 1, 1)  # (1, C, 1, 1) for broadcasting

    prob_x = torch.roll(prob, -1, dims=3)
    # Compute per-class loss and apply class weights before summing
    loss_x_per_class = 0.5 * (prob - prob_x)**2  # (B, C, H, W)
    loss_x_weighted = loss_x_per_class * class_weights  # (B, C, H, W)
    loss_x = torch.sum(loss_x_weighted, dim=1)  # (B, H, W)
    loss_x = loss_x[:, :, :-1] * w_x

    prob_y = torch.roll(prob, -1, dims=2)
    # Compute per-class loss and apply class weights before summing
    loss_y_per_class = 0.5 * (prob - prob_y)**2  # (B, C, H, W)
    loss_y_weighted = loss_y_per_class * class_weights  # (B, C, H, W)
    loss_y = torch.sum(loss_y_weighted, dim=1)  # (B, H, W)
    loss_y = loss_y[:, :-1, :] * w_y

    return loss_x.mean() + loss_y.mean()