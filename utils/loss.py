import torch
import torch.nn.functional as F
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

def calculate_pairwise_affinity(sam_contour, transform_type):
    device = sam_contour.device

    # dilate by 15
    sam_contour_cpu = sam_contour.cpu().numpy()
    dilated_contour = np.zeros_like(sam_contour_cpu)
    for i in range(sam_contour_cpu.shape[0]):
        dilated_contour[i] = ndimage.maximum_filter(sam_contour_cpu[i], size=17)
    sam_contour = torch.from_numpy(dilated_contour).to(device)

    if transform_type is None:
        w = (~sam_contour.bool()).to(torch.float32)
    else:
        sam_contour_cpu = sam_contour.cpu().numpy()
        w = np.zeros(sam_contour_cpu.shape)
        for i in range(sam_contour_cpu.shape[0]):
            if transform_type == 'euclidean':
                w[i] = ndimage.distance_transform_edt(~sam_contour_cpu[i].astype(bool))
            elif transform_type == 'exponential':
                # T = 20
                w[i] = np.exp(ndimage.distance_transform_edt(~sam_contour_cpu[i].astype(bool)) / 20.0) * 20.0
            elif transform_type == 'gaussian_blur':
                blur = ndimage.gaussian_filter(sam_contour_cpu[i], sigma=5)
                w[i] = 1-blur
            
        w = torch.from_numpy(w).to(dtype=torch.float32, device=device)
    return w

def CollisionCrossEntropyLoss(logits, target_probs):
    """
    See "Soft Self-labeling and Potts Relaxations for Weakly-Supervised Segmentation" paper.
    CCE loss is robust to pseudo-label uncertainty without requiring hard labels.
    
    Args:
        logits: (B, C, H, W) tensor of logits from model
        target_probs: (B, C, H, W) tensor of target probabilities (pseudolabels)
        class_weights: (C,) tensor of class weights for class balancing (optional)
    """
    probs = torch.softmax(logits, dim=1)
    
    # Compute pixel-wise log probabilities
    log_sum = torch.log(torch.sum(probs * target_probs, dim=1) + 1e-8)  # (B, H, W)
    per_pixel_loss = -log_sum  # (B, H, W)
    
    loss = per_pixel_loss.mean()
    
    return loss

def PottsLoss(type, logits, sam_contours_x, sam_contours_y, distance_transform):
    w_x = calculate_pairwise_affinity(sam_contours_x, distance_transform)
    w_y = calculate_pairwise_affinity(sam_contours_y, distance_transform)
    
    if type == 'bilinear':
        weighting = 200.0
        
        prob = torch.softmax(logits, dim=1)
        
        prob_x = torch.roll(prob, -1, dims=3)
        loss_x = 1 - torch.sum(prob*prob_x, dim=1)
        loss_x = loss_x[:, :, :-1] * w_x

        prob_y = torch.roll(prob, -1, dims=2)
        loss_y = 1 - torch.sum(prob*prob_y, dim=1)
        loss_y = loss_y[:, :-1, :] * w_y
    elif type == 'quadratic':
        prob = torch.softmax(logits, dim=1)
        num_classes = prob.shape[1]
        
        device = prob.device
        class_weights = torch.full((num_classes,), 20000.0, device=device, dtype=prob.dtype)

        # List A:
        # class_weights[5] = 500.0    # bottle
        # class_weights[11] = 500.0   # diningtable
        # class_weights[16] = 500.0   # potted plant

        # List B:
        # class_weights[2] = 1300.0   # bicycle
        # class_weights[4] = 1300.0   # boat
        # class_weights[20] = 1300.0  # tv/monitor
        # class_weight[0] = 1300.0  # background

        # List C: aeroplane, bus, car, cat, cow, dog, motorbike, sheep
        # class_weights[1] = 3000.0   # aeroplane
        # class_weights[6] = 3000.0   # bus
        # class_weights[7] = 3000.0   # car
        # class_weights[8] = 3000.0   # cat
        # class_weights[10] = 3000.0   # cow
        # class_weights[12] = 3000.0   # dog
        # class_weights[14] = 3000.0   # motorbike
        # class_weights[17] = 3000.0   # sheep

        # List A:
        # class_weights[5] = 300.0    # bottle
        # class_weights[11] = 300.0   # diningtable
        # class_weights[16] = 0.0   # potted plant
        # class_weights[20] = 300.0  # tv/monitor

        # # List B:
        # class_weights[2] = 2500.0   # bicycle
        # class_weights[4] = 2500.0   # boat
        # class_weights[8] = 2500.0   # cat

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
    elif type == 'log_quadratic':
        prob = torch.softmax(logits, dim=1)
        num_classes = prob.shape[1]

        device = prob.device
        class_weights = torch.full((num_classes,), 30000.0, device=device, dtype=prob.dtype)

        class_weights = class_weights.view(1, num_classes, 1, 1)  # (1, C, 1, 1) for broadcasting

        diff_x = prob - torch.roll(prob, -1, dims=3)
        diff_y = prob - torch.roll(prob, -1, dims=2)

        sq_norm_x = torch.sum((diff_x ** 2) * class_weights, dim=1)
        sq_norm_y = torch.sum((diff_y ** 2) * class_weights, dim=1)

        eps = 1e-6

        inside_x = torch.clamp(1.0 - 0.5 * sq_norm_x[:, :, :-1], min=eps)
        inside_y = torch.clamp(1.0 - 0.5 * sq_norm_y[:, :-1, :], min=eps)

        loss_x = -torch.log(inside_x) * w_x
        loss_y = -torch.log(inside_y) * w_y

    loss = loss_x.mean() + loss_y.mean()
    return loss