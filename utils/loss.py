import torch
import torch.nn.functional as F
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

def calculate_pairwise_affinity(sam_contour, transform_type):
    device = sam_contour.device

    if transform_type is None:
        w = (~sam_contour.bool()).to(torch.float32)
    else:
        sam_contour_cpu = sam_contour.cpu().numpy()
        w = np.zeros(sam_contour_cpu.shape)
        for i in range(sam_contour_cpu.shape[0]):
            if transform_type == 'euclidean':
                w[i] = ndimage.distance_transform_edt(~sam_contour_cpu[i].bool())
            elif transform_type == 'exponential':
                # T = 20
                w[i] = np.exp(ndimage.distance_transform_edt(~sam_contour_cpu[i].bool()) / 20.0) * 20.0
            elif transform_type == 'gaussian_blur':
                blur = ndimage.gaussian_filter(sam_contour_cpu[i], sigma=5)
                w[i] = 1-blur
            
        w = torch.from_numpy(w).to(dtype=torch.float32, device=device)
    return w

def CrossEntropyLoss(logits, target_logits):
    """
    For each pixel, sets the largest value in the C channel of target_logits to 1 (others to 0),
    then computes the cross-entropy between logits and this one-hot mask.
    """
    max_idx = torch.argmax(target_logits, dim=1, keepdim=True)  # (B, 1, H, W)
    target_one_hot = torch.zeros_like(target_logits)
    target_one_hot.scatter_(1, max_idx, 1)
    log_prob = F.log_softmax(logits, dim=1)
    loss = -(target_one_hot * log_prob).sum(dim=1).mean()
    return loss

def CollisionCrossEntropyLoss(logits, target_logits):
    """
    See "Soft Self-labeling and Potts Relaxations for Weakly-Supervised Segmentation" paper.
    CCE loss is robust to pseudo-label uncertainty without requiring hard labels.
    """
    t = 0.01 # temperature
    target_log_prob = F.log_softmax(target_logits / t, dim=1)
    log_prob = F.log_softmax(logits, dim=1)
    loss = (-torch.logsumexp(log_prob + target_log_prob, dim=1)).mean()
    
    return loss

def PottsLoss(type, logits, sam_contours_x, sam_contours_y, distance_transform):
    w_x = calculate_pairwise_affinity(sam_contours_x, distance_transform)
    w_y = calculate_pairwise_affinity(sam_contours_y, distance_transform)
    
    if type == 'bilinear':
        if distance_transform is None:
            weighting = 200.0
        
        prob = torch.softmax(logits, dim=1)
        
        prob_x = torch.roll(prob, -1, dims=3)
        loss_x = 1 - torch.sum(prob*prob_x, dim=1)
        loss_x = loss_x[:, :, :-1] * w_x

        prob_y = torch.roll(prob, -1, dims=2)
        loss_y = 1 - torch.sum(prob*prob_y, dim=1)
        loss_y = loss_y[:, :-1, :] * w_y
    elif type == 'quadratic':
        if distance_transform is None:
            weighting = 100.0
        
        prob = torch.softmax(logits, dim=1)
        
        prob_x = torch.roll(prob, -1, dims=3)
        loss_x = 0.5 * torch.sum((prob - prob_x)**2, dim=1)
        loss_x = loss_x[:, :, :-1] * w_x

        prob_y = torch.roll(prob, -1, dims=2)
        loss_y = 0.5 * torch.sum((prob - prob_y)**2, dim=1)
        loss_y = loss_y[:, :-1, :] * w_y

    loss = loss_x.mean() + loss_y.mean()
    loss *= weighting
    return loss