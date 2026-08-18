import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.segmentation import slic as _skimage_slic


def generate_contours_batch(method, images, targets, denormalize, size, device,
                            sam_mask_generator=None, fastsam_model=None):
    """
    Contours for a batch, from whichever source `method` names. Everything is produced at
    `size`, the resolution the segmentation logits come out at.

    Args:
        method: 'sam', 'fastsam', 'slic', 'gt' or 'color_diff'
        images: (B, C, H, W) normalized image batch
        targets: (B, H, W) ground truth masks, only used by 'gt'
        denormalize: undoes the image normalization in place
        size: (H, W) resolution to produce contours at
        device: torch device

    Returns:
        contours_x: (B, H, W-1) and contours_y: (B, H-1, W) float tensors
    """
    if method == 'gt':
        return generate_gt_contours_batch(targets, size, device)

    denormed = torch.stack([denormalize(img.clone()) for img in images])

    if method == 'color_diff':
        resized = F.interpolate(denormed.to(device), size=size, mode='bilinear', align_corners=False)
        return generate_color_diff_contours_batch(resized, device)

    images_pil = [Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                  for img in denormed]
    if method == 'sam':
        contours = generate_sam_contours_batch(sam_mask_generator, images_pil, size)
    elif method == 'fastsam':
        contours = generate_fastsam_contours_batch(fastsam_model, images_pil, size, device)
    elif method == 'slic':
        contours = generate_slic_contours_batch(images_pil, size)
    else:
        raise ValueError(f"Unknown contour_method: {method}")
    return contours[0].to(device), contours[1].to(device)


def _contours_from_masks(masks, shape):
    """OR together the boundaries of every binary mask. Returns (H, W-1) and (H-1, W) floats."""
    H, W = shape
    contours_x = np.zeros((H, W - 1), dtype=bool)
    contours_y = np.zeros((H - 1, W), dtype=bool)
    for mask in masks:
        mask = mask.astype(bool)
        assert mask.shape == (H, W), f"mask shape {mask.shape} != image shape {(H, W)}"
        contours_x |= np.logical_xor(mask[:, :-1], mask[:, 1:])
        contours_y |= np.logical_xor(mask[:-1, :], mask[1:, :])
    return torch.from_numpy(contours_x).float(), torch.from_numpy(contours_y).float()


def generate_gt_contours_batch(targets, size, device):
    """
    Generate contours from ground truth segmentation masks.
    Edges are detected when classes switch between neighboring pixels.

    Args:
        targets: torch.Tensor of shape [B, H, W] with class indices (int64)
        size: (H, W) resolution to produce contours at
        device: torch device

    Returns:
        contours_x_batch: torch.Tensor of shape [B, H, W-1] with float dtype
        contours_y_batch: torch.Tensor of shape [B, H-1, W] with float dtype
    """
    # Nearest-neighbour resize preserves the class indices
    targets = F.interpolate(targets.unsqueeze(1).float(), size=size, mode='nearest').squeeze(1)

    contours_x = (targets[:, :, :-1] != targets[:, :, 1:]).float()  # [B, H, W-1]
    contours_y = (targets[:, :-1, :] != targets[:, 1:, :]).float()  # [B, H-1, W]
    return contours_x.to(device), contours_y.to(device)


def generate_fastsam_contours_batch(fastsam_model, images, size, device):
    """
    Generate FastSAM contours for a batch of images.

    Args:
        fastsam_model: FastSAM model instance
        images: List of PIL Images
        size: (H, W) resolution to produce contours at
        device: torch device

    Returns:
        contours_x_batch: torch.Tensor of shape [B, H, W-1] with float dtype
        contours_y_batch: torch.Tensor of shape [B, H-1, W] with float dtype
    """
    contours_x_list, contours_y_list = [], []

    for image in images:
        image_np = np.array(image.resize(size[::-1], Image.Resampling.BILINEAR))

        # FastSAM "everything" mode
        results = fastsam_model(
            image_np,
            device=device,
            retina_masks=True,
            imgsz=1024,
            conf=0.1,
            iou=0.7,
            verbose=False,
        )
        masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else []

        contours_x, contours_y = _contours_from_masks(masks, image_np.shape[:2])
        contours_x_list.append(contours_x)
        contours_y_list.append(contours_y)

    return torch.stack(contours_x_list), torch.stack(contours_y_list)


def generate_sam_contours_batch(sam_mask_generator, images, size):
    """
    Generate SAM (Segment Anything Model) contours for a batch of images using the automatic
    mask generator.

    Args:
        sam_mask_generator: SamAutomaticMaskGenerator instance
        images: List of PIL Images
        size: (H, W) resolution to produce contours at

    Returns:
        contours_x_batch: torch.Tensor of shape [B, H, W-1] with float dtype
        contours_y_batch: torch.Tensor of shape [B, H-1, W] with float dtype
    """
    contours_x_list, contours_y_list = [], []

    for image in images:
        image_np = np.array(image.resize(size[::-1], Image.Resampling.BILINEAR))  # [H, W, 3] RGB
        masks = [m['segmentation'] for m in sam_mask_generator.generate(image_np)]

        contours_x, contours_y = _contours_from_masks(masks, image_np.shape[:2])
        contours_x_list.append(contours_x)
        contours_y_list.append(contours_y)

    return torch.stack(contours_x_list), torch.stack(contours_y_list)


def generate_slic_contours_batch(images, size, n_segments=100, compactness=10.0, sigma=0.0):
    """
    Generate SLIC-superpixel contours for a batch of images.

    Args:
        images: List of PIL Images (RGB)
        size: (H, W) resolution to produce contours at
        n_segments: approximate number of superpixels
        compactness: SLIC compactness parameter
        sigma: Gaussian smoothing applied prior to segmentation (in pixels)

    Returns:
        contours_x_batch: torch.Tensor of shape [B, H, W-1] with float dtype
        contours_y_batch: torch.Tensor of shape [B, H-1, W] with float dtype
    """
    contours_x_list, contours_y_list = [], []

    for image in images:
        image_np = np.array(image.resize(size[::-1], Image.Resampling.BILINEAR)).astype(np.float32) / 255.0
        labels = _skimage_slic(
            image_np,
            n_segments=int(n_segments),
            compactness=float(compactness),
            sigma=float(sigma),
            start_label=0,
            channel_axis=-1,
        )  # [H, W] int

        contours_x_list.append(torch.from_numpy((labels[:, :-1] != labels[:, 1:]).astype(np.float32)))
        contours_y_list.append(torch.from_numpy((labels[:-1, :] != labels[1:, :]).astype(np.float32)))

    return torch.stack(contours_x_list), torch.stack(contours_y_list)


def generate_color_diff_contours_batch(images, device, sigma=0.2):
    """
    Generate color-difference based affinities for a batch of images. Unlike the other
    methods these are already weights, so they are used without negation.

    Args:
        images: torch.Tensor of shape [B, C, H, W] with images normalized to [0, 1]
        device: torch device
        sigma: Standard deviation parameter for the Gaussian weighting

    Returns:
        contours_x_batch: torch.Tensor of shape [B, H, W-1] with float dtype
        contours_y_batch: torch.Tensor of shape [B, H-1, W] with float dtype
    """
    diff_x = torch.sum((images[:, :, :, :-1] - images[:, :, :, 1:]) ** 2, dim=1)  # (B, H, W-1)
    diff_y = torch.sum((images[:, :, :-1, :] - images[:, :, 1:, :]) ** 2, dim=1)  # (B, H-1, W)

    contours_x_batch = torch.exp(-diff_x / (2 * sigma ** 2))
    contours_y_batch = torch.exp(-diff_y / (2 * sigma ** 2))
    return contours_x_batch.to(device), contours_y_batch.to(device)
