import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
from skimage.segmentation import slic as _skimage_slic

def generate_gt_contours_batch(targets, device):
    """
    Generate contours from ground truth segmentation masks.
    Edges are detected when classes switch between neighboring pixels.
    
    Args:
        targets: torch.Tensor of shape [B, H, W] with class indices (int64)
        device: torch device
    
    Returns:
        contours_x_batch: torch.Tensor of shape [B, H, W-1] with float dtype
        contours_y_batch: torch.Tensor of shape [B, H-1, W] with float dtype
    """
    B, H, W = targets.shape
    
    # Resize targets using nearest-neighbor interpolation to preserve class indices
    targets_downsampled = targets.unsqueeze(1).float()  # [B, 1, H, W]
    H_downsampled = H // 4
    W_downsampled = W // 4
    targets_downsampled = F.interpolate(
        targets_downsampled,
        size=(H_downsampled, W_downsampled),
        mode='nearest'
    ).squeeze(1).long()  # [B, H_downsampled, W_downsampled]
    
    # Convert to numpy for efficient comparison
    targets_np = targets_downsampled.cpu().numpy()
    
    contours_x_list = []
    contours_y_list = []
    
    for b in range(B):
        target = targets_np[b]  # [H_downsampled, W_downsampled]
        
        # Detect horizontal edges (contours_x): compare each pixel with its right neighbor
        contours_x = (target[:, :-1] != target[:, 1:]).astype(np.float32)  # [H, W-1]
        
        # Detect vertical edges (contours_y): compare each pixel with its bottom neighbor
        contours_y = (target[:-1, :] != target[1:, :]).astype(np.float32)  # [H-1, W]
        
        contours_x_tensor = torch.from_numpy(contours_x).to(device)  # [H, W-1]
        contours_y_tensor = torch.from_numpy(contours_y).to(device)  # [H-1, W]
        contours_x_list.append(contours_x_tensor)
        contours_y_list.append(contours_y_tensor)
    
    # Stack into batch tensors
    contours_x_batch = torch.stack(contours_x_list)  # [B, H, W-1]
    contours_y_batch = torch.stack(contours_y_list)  # [B, H-1, W]
    
    return contours_x_batch, contours_y_batch

def generate_fastsam_contours_batch(fastsam_model, images, device):
    """
    Generate FastSAM contours for a batch of images.
    
    Args:
        fastsam_model: FastSAM model instance
        images: List of PIL Images
        device: torch device
    
    Returns:
        contours_x_batch: torch.Tensor of shape [B, H, W-1] with float dtype
        contours_y_batch: torch.Tensor of shape [B, H-1, W] with float dtype
    """
    contours_x_list = []
    contours_y_list = []
    
    for image in images:
        # Resize to 4x downsampled
        image = image.resize((image.size[1] // 4, image.size[0] // 4), Image.Resampling.BILINEAR)
        
        # Save image temporarily for FastSAM (it expects a file path or numpy array)
        image_np = np.array(image)
        
        # Generate masks using FastSAM everything mode
        everything_results = fastsam_model(
            image_np, 
            device=device, 
            retina_masks=True, 
            imgsz=1024,
            conf=0.1,
            iou=0.7,
            verbose=False  # Disable logging
        )
        
        H, W = image_np.shape[:2]
        contours_x = np.zeros((H, W - 1), dtype=bool)
        contours_y = np.zeros((H - 1, W), dtype=bool)
        
        # Extract segmentation masks from FastSAM results
        # FastSAM results are YOLO-style, masks are in results[0].masks.data
        if everything_results[0].masks is not None:
            masks = everything_results[0].masks.data.cpu().numpy()  # [num_masks, H, W]
            for mask in masks:
                # Convert to boolean and ensure correct shape
                mask_bool = mask.astype(bool)
                if mask_bool.shape[0] == H and mask_bool.shape[1] == W:
                    contours_x |= np.logical_xor(mask_bool[:, :-1], mask_bool[:, 1:])  # shape: (H, W-1)
                    contours_y |= np.logical_xor(mask_bool[:-1, :], mask_bool[1:, :])  # shape: (H-1, W)
        
        # Convert to tensors
        contours_x_tensor = torch.from_numpy(contours_x).float()  # [H, W-1]
        contours_y_tensor = torch.from_numpy(contours_y).float()  # [H-1, W]
        contours_x_list.append(contours_x_tensor)
        contours_y_list.append(contours_y_tensor)
    
    # Stack into batch tensors
    contours_x_batch = torch.stack(contours_x_list)  # [B, H, W-1]
    contours_y_batch = torch.stack(contours_y_list)  # [B, H-1, W]
    
    return contours_x_batch, contours_y_batch

def generate_sam_contours_batch(sam_mask_generator, images, device):
    """xf
    Generate SAM (Segment Anything Model) contours for a batch of images using the automatic mask generator.
    
    Args:
        sam_mask_generator: SamAutomaticMaskGenerator instance
        images: List of PIL Images
        device: torch device
    
    Returns:
        contours_x_batch: torch.Tensor of shape [B, H, W-1] with float dtype
        contours_y_batch: torch.Tensor of shape [B, H-1, W] with float dtype
    """
    contours_x_list = []
    contours_y_list = []
    
    for image in images:
        H_orig, W_orig = image.size[1], image.size[0]
        H_downsampled = H_orig // 4
        W_downsampled = W_orig // 4
        image_resized = image.resize((W_downsampled, H_downsampled), Image.Resampling.BILINEAR)
        
        # Convert to numpy array for SAM
        image_np = np.array(image_resized)  # [H, W, 3] in RGB
        
        H, W = image_np.shape[:2]
        contours_x = np.zeros((H, W - 1), dtype=bool)
        contours_y = np.zeros((H - 1, W), dtype=bool)
        
        # Generate masks using SAM automatic mask generator
        masks = sam_mask_generator.generate(image_np)
        
        # Extract contours from all masks
        for mask in masks:
            segmentation = mask['segmentation']  # Boolean array [H, W]
            contours_x |= np.logical_xor(segmentation[:, :-1], segmentation[:, 1:])  # shape: (H, W-1)
            contours_y |= np.logical_xor(segmentation[:-1, :], segmentation[1:, :])  # shape: (H-1, W)
        
        # Convert to tensors
        contours_x_tensor = torch.from_numpy(contours_x).float()  # [H, W-1]
        contours_y_tensor = torch.from_numpy(contours_y).float()  # [H-1, W]
        contours_x_list.append(contours_x_tensor)
        contours_y_list.append(contours_y_tensor)
    
    # Stack into batch tensors
    contours_x_batch = torch.stack(contours_x_list)  # [B, H, W-1]
    contours_y_batch = torch.stack(contours_y_list)  # [B, H-1, W]
    
    return contours_x_batch, contours_y_batch

def generate_slic_contours_batch(
    images,
    device,
    n_segments=100,
    compactness=10.0,
    sigma=0.0,
):
    """
    Generate SLIC-superpixel contours for a batch of images.

    Args:
        images: List of PIL Images (RGB)
        device: torch device (kept for API parity with other generators)
        n_segments: approximate number of superpixels
        compactness: SLIC compactness parameter
        sigma: Gaussian smoothing applied prior to segmentation (in pixels)

    Returns:
        contours_x_batch: torch.Tensor of shape [B, H, W-1] with float dtype
        contours_y_batch: torch.Tensor of shape [B, H-1, W] with float dtype
    """

    contours_x_list = []
    contours_y_list = []

    for image in images:
        H_orig, W_orig = image.size[1], image.size[0]
        H_downsampled = H_orig // 4
        W_downsampled = W_orig // 4
        image_resized = image.resize((W_downsampled, H_downsampled), Image.Resampling.BILINEAR)

        image_np = np.array(image_resized).astype(np.float32) / 255.0  # [H, W, 3]
        labels = _skimage_slic(
            image_np,
            n_segments=int(n_segments),
            compactness=float(compactness),
            sigma=float(sigma),
            start_label=0,
            channel_axis=-1,
        )  # [H, W] int

        contours_x = (labels[:, :-1] != labels[:, 1:]).astype(np.float32)  # [H, W-1]
        contours_y = (labels[:-1, :] != labels[1:, :]).astype(np.float32)  # [H-1, W]

        contours_x_list.append(torch.from_numpy(contours_x))
        contours_y_list.append(torch.from_numpy(contours_y))

    contours_x_batch = torch.stack(contours_x_list)  # [B, H, W-1]
    contours_y_batch = torch.stack(contours_y_list)  # [B, H-1, W]

    return contours_x_batch, contours_y_batch