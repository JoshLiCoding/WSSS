import torch
import torch.nn.functional as F
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

def calculate_pairwise_affinity(sam_contour, transform_type):
    device = sam_contour.device
    if transform_type is None:
        w = (~sam_contour).to(torch.float32)
    else:
        sam_contour_cpu = sam_contour.cpu().numpy()
        w = np.zeros(sam_contour_cpu.shape)
        for i in range(sam_contour_cpu.shape[0]):
            if transform_type == 'euclidean':
                w[i] = ndimage.distance_transform_edt(~sam_contour_cpu[i])
            elif transform_type == 'manhattan':
                w[i] = ndimage.distance_transform_cdt(~sam_contour_cpu[i], metric='taxicab')
        # Convert back to tensor and move to original device
        w = torch.from_numpy(w).to(dtype=torch.float32, device=device)
    w[:, :, -1] = 0
    w[:, -1, :] = 0
    return w

def CollisionCrossEntropyLoss(logits, target_logits):
    target_log_prob = F.log_softmax(target_logits, dim=1)
    log_prob = F.log_softmax(logits, dim=1)
    loss = (-torch.logsumexp(log_prob + target_log_prob, dim=1)).mean()
    
    return loss

# Collision Divergence
def CDPottsLoss(logits, sam_contour, distance_transform='euclidean', weighting=1.0):
    w = calculate_pairwise_affinity(sam_contour, distance_transform)

    log_prob = F.log_softmax(logits, dim=1)
    norm = torch.norm(F.softmax(logits, dim=1), p=2, dim=1) #L2 norm
    
    logits_roll_x = torch.roll(logits, 1, dims=3)
    log_prob_x = F.log_softmax(logits_roll_x, dim=1)
    norm_x = torch.norm(F.softmax(logits_roll_x, dim=1), p=2, dim=1)
    
    loss_x = torch.logsumexp(log_prob + log_prob_x, dim=1) - (torch.log(norm) + torch.log(norm_x))
    loss_x = loss_x * w
    
    logits_roll_y = torch.roll(logits, 1, dims=2)
    log_prob_y = F.log_softmax(logits_roll_y, dim=1)
    norm_y = torch.norm(F.softmax(logits_roll_y, dim=1), p=2, dim=1)
    
    loss_y = torch.logsumexp(log_prob + log_prob_y, dim=1) - (torch.log(norm) + torch.log(norm_y))
    loss_y = loss_y * w

    loss = (-(loss_x + loss_y) * weighting).mean()
    return loss

# Quadratic
def QuadPottsLoss(logits, sam_contour, distance_transform='euclidean', weighting=100.0):
    w = calculate_pairwise_affinity(sam_contour, distance_transform)

    prob = torch.softmax(logits, dim=1)
    
    prob_x = torch.roll(prob, -1, dims=3)
    loss_x = torch.square(torch.norm(prob-prob_x, p=2, dim=1)) / 2.0
    loss_x = loss_x * w
    
    prob_y = torch.roll(prob, -1, dims=2)
    loss_y = torch.square(torch.norm(prob-prob_y, p=2, dim=1)) / 2.0
    loss_y = loss_y * w

    loss = ((loss_x+loss_y) * weighting).mean()
    return loss

# Bi-linear
def BLPottsLoss(logits, sam_contour, distance_transform='euclidean', weighting=1.0):
    w = calculate_pairwise_affinity(sam_contour, distance_transform)
    
    prob = torch.softmax(logits, dim=1)
    
    prob_x = torch.roll(prob, -1, dims=3)
    loss_x = 1 - torch.sum(prob*prob_x, dim=1)
    loss_x = loss_x * w
    
    prob_y = torch.roll(prob, -1, dims=2)
    loss_y = 1 - torch.sum(prob*prob_y, dim=1)
    loss_y = loss_y * w

    loss = ((loss_x+loss_y) * weighting).mean()
    return loss

def EntropyLoss(logits):
    a = F.softmax(logits, dim=1)*F.log_softmax(logits, dim=1)
    b = a.sum(dim=1)
    return -b.mean()