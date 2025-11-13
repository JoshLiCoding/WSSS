"""Visualize COCO images with ground truth masks and predicted pseudolabels"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

from utils.dataset import COCOSegmentation

# COCO class names (80 classes + background)
COCO_CLASS_NAMES = {
    0: "background",
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 12: "stop sign", 13: "parking meter", 14: "bench",
    15: "bird", 16: "cat", 17: "dog", 18: "horse", 19: "sheep",
    20: "cow", 21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe",
    25: "backpack", 26: "umbrella", 27: "handbag", 28: "tie", 29: "suitcase",
    30: "frisbee", 31: "skis", 32: "snowboard", 33: "sports ball", 34: "kite",
    35: "baseball bat", 36: "baseball glove", 37: "skateboard", 38: "surfboard",
    39: "tennis racket", 40: "bottle", 41: "wine glass", 42: "cup", 43: "fork",
    44: "knife", 45: "spoon", 46: "bowl", 47: "banana", 48: "apple",
    49: "sandwich", 50: "orange", 51: "broccoli", 52: "carrot", 53: "hot dog",
    54: "pizza", 55: "donut", 56: "cake", 57: "chair", 58: "couch",
    59: "potted plant", 60: "bed", 61: "dining table", 62: "toilet", 63: "tv",
    64: "laptop", 65: "mouse", 66: "remote", 67: "keyboard", 68: "cell phone",
    69: "microwave", 70: "oven", 71: "toaster", 72: "sink", 73: "refrigerator",
    74: "book", 75: "clock", 76: "vase", 77: "scissors", 78: "teddy bear",
    79: "hair drier", 80: "toothbrush", 255: "ignore"
}

def visualize_coco_sample(
    coco_dataset,
    index,
    pseudolabels_dir,
    output_path=None,
    canonical_size=(448, 448)
):
    """
    Visualize COCO image with ground truth mask and soft predicted pseudolabels.
    
    Shows probability-weighted color blending where each pixel's color is a weighted
    combination of all class colors, where weights are the class probabilities.
    
    Args:
        coco_dataset: COCOSegmentation dataset instance
        index: Index of the sample to visualize
        pseudolabels_dir: Directory containing pseudolabel files
        output_path: Path to save the visualization (optional)
        canonical_size: Size used for pseudolabels (H, W)
    """
    # Load image and ground truth
    img, gt_mask = coco_dataset[index]
    img_array = np.array(img)
    gt_mask_array = np.array(gt_mask)
    
    # Load pseudolabels
    pseudolabels_path = os.path.join(pseudolabels_dir, f'pseudolabels_{index}.npy')
    class_indices_path = os.path.join(pseudolabels_dir, f'class_indices_{index}.npy')
    
    if not os.path.exists(pseudolabels_path):
        raise FileNotFoundError(f"Pseudolabels not found: {pseudolabels_path}")
    if not os.path.exists(class_indices_path):
        raise FileNotFoundError(f"Class indices not found: {class_indices_path}")
    
    # Load pseudolabels and class indices
    pseudolabels = np.load(pseudolabels_path)  # (H, W, C) - logits or probabilities
    class_indices = np.load(class_indices_path)  # (C-1,) - class indices (excluding background)
    
    # Resize pseudolabels to match image size if needed
    H, W = img_array.shape[:2]
    if pseudolabels.shape[:2] != (H, W):
        pseudolabels_tensor = torch.from_numpy(pseudolabels).permute(2, 0, 1).unsqueeze(0).float()
        pseudolabels_tensor = F.interpolate(
            pseudolabels_tensor,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        pseudolabels = pseudolabels_tensor[0].permute(1, 2, 0).numpy()
    
    pseudolabels_probs = pseudolabels / 0.01
    pseudolabels_probs = torch.from_numpy(pseudolabels_probs).float().permute(2, 0, 1)
    pseudolabels_probs = pseudolabels_probs.softmax(dim=0)
    pseudolabels_probs = pseudolabels_probs.permute(1, 2, 0).numpy()
    
    num_fg_classes = len(class_indices)
    num_classes_total = pseudolabels_probs.shape[2]  # num_fg_classes + 1 (background)
    
    # Get class names for legend
    present_classes = np.unique(gt_mask_array)
    present_classes = present_classes[present_classes != 255]  # Remove ignore
    gt_class_names = [COCO_CLASS_NAMES.get(cls, f"class_{cls}") for cls in present_classes]
    
    # Get pseudolabel class names
    pseudolabel_class_names = [COCO_CLASS_NAMES.get(cls, f"class_{cls}") for cls in class_indices]
    
    # Create probability-weighted color blend
    # For each pixel, blend all class colors weighted by their probabilities
    soft_mask_colored = np.zeros((H, W, 3), dtype=np.float32)
    
    # Get colormap
    cmap = COCOSegmentation.cmap
    
    # Blend background color
    bg_probs = pseudolabels_probs[:, :, num_fg_classes]  # (H, W)
    bg_color = cmap[0]  # Background color
    soft_mask_colored += bg_probs[:, :, np.newaxis] * bg_color
    
    # Blend foreground class colors
    for i, class_idx in enumerate(class_indices):
        class_probs = pseudolabels_probs[:, :, i]  # (H, W)
        class_color = cmap[class_idx]  # (3,)
        soft_mask_colored += class_probs[:, :, np.newaxis] * class_color
    
    # Convert to uint8
    soft_mask_colored = np.clip(soft_mask_colored, 0, 255).astype(np.uint8)
    
    # Also compute max probability for confidence visualization
    max_probs = np.max(pseudolabels_probs, axis=2)  # (H, W)
    max_class_idx = np.argmax(pseudolabels_probs, axis=2)  # (H, W)
    
    # Map max class indices to actual COCO class indices for overlay
    max_class_mapped = np.zeros_like(max_class_idx, dtype=np.uint8)
    for i in range(num_fg_classes):
        max_class_mapped[max_class_idx == i] = class_indices[i]
    max_class_mapped[max_class_idx == num_fg_classes] = 0  # background
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 0: Original image, GT mask, GT overlay
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title(f'Sample {index} - Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    gt_colored = COCOSegmentation.decode_target(gt_mask_array)
    axes[0, 1].imshow(gt_colored)
    axes[0, 1].set_title('Ground Truth Mask', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    gt_overlay = img_array.copy().astype(float)
    non_bg_gt = gt_mask_array > 0
    gt_overlay[non_bg_gt] = gt_overlay[non_bg_gt] * 0.5 + gt_colored[non_bg_gt] * 0.5
    axes[0, 2].imshow(gt_overlay.astype(np.uint8))
    axes[0, 2].set_title('Image + GT Overlay', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 1: Soft pseudolabel visualization
    axes[1, 0].imshow(img_array)
    axes[1, 0].set_title('Original Image', fontsize=12)
    axes[1, 0].axis('off')
    
    # Show probability-weighted color blend (soft pseudolabel)
    axes[1, 1].imshow(soft_mask_colored)
    axes[1, 1].set_title('Soft Pseudolabel (Probability-Weighted Colors)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Show overlay with soft pseudolabel
    soft_overlay = img_array.copy().astype(float)
    # Use max probability as alpha for overlay blending
    alpha = max_probs[:, :, np.newaxis] * 0.6  # 60% opacity based on confidence
    soft_overlay = soft_overlay * (1 - alpha) + soft_mask_colored.astype(float) * alpha
    axes[1, 2].imshow(soft_overlay.astype(np.uint8))
    axes[1, 2].set_title('Image + Soft Pseudolabel Overlay', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Add text information
    info_text = f"GT classes: {', '.join(gt_class_names[:3])}"
    if len(gt_class_names) > 3:
        info_text += f" ... (+{len(gt_class_names)-3} more)"
    info_text += f" | Pseudolabel classes: {', '.join(pseudolabel_class_names[:3])}"
    if len(pseudolabel_class_names) > 3:
        info_text += f" ... (+{len(pseudolabel_class_names)-3} more)"
    fig.suptitle(f'COCO Sample {index} - Soft Pseudolabels\n{info_text}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_path}")
    else:
        plt.savefig(f'coco_pseudolabel_vis_{index}.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to coco_pseudolabel_vis_{index}.png")
    
    plt.close()
    
    # Also print some statistics
    print(f"\nSample {index} Statistics:")
    print(f"  Image size: {img_array.shape}")
    print(f"  GT mask size: {gt_mask_array.shape}")
    print(f"  Pseudolabels shape: {pseudolabels.shape}")
    print(f"  Pseudolabels probabilities shape: {pseudolabels_probs.shape}")
    print(f"  Classes in GT: {present_classes.tolist()}")
    print(f"  Classes in pseudolabel: {class_indices.tolist()}")
    print(f"  Max probability range: [{max_probs.min():.3f}, {max_probs.max():.3f}]")
    print(f"  Mean max probability: {max_probs.mean():.3f}")
    print(f"  Predicted classes (from argmax): {np.unique(max_class_mapped).tolist()}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize COCO images with GT and pseudolabels')
    parser.add_argument('--coco_root', type=str, default='coco',
                        help='Root directory of COCO dataset')
    parser.add_argument('--pseudolabels_dir', type=str, default='pseudolabels_coco',
                        help='Directory containing pseudolabel files')
    parser.add_argument('--index', type=int, default=0,
                        help='Index of sample to visualize')
    parser.add_argument('--image_set', type=str, default='train', choices=['train', 'val'],
                        help='Dataset split to use')
    parser.add_argument('--year', type=str, default='2014', choices=['2014', '2017'],
                        help='COCO dataset year')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for visualization (optional)')
    parser.add_argument('--n_samples', type=int, default=1,
                        help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Load COCO dataset
    print(f"Loading COCO {args.year} {args.image_set} dataset...")
    coco_dataset = COCOSegmentation(
        root=args.coco_root,
        year=args.year,
        image_set=args.image_set,
        n_images=-1
    )
    print(f"✓ Loaded {len(coco_dataset)} images")
    
    # Visualize samples
    pseudolabels_dir = Path(args.pseudolabels_dir)
    if not pseudolabels_dir.exists():
        raise FileNotFoundError(f"Pseudolabels directory not found: {pseudolabels_dir}")
    
    start_idx = args.index
    end_idx = min(start_idx + args.n_samples, len(coco_dataset))
    
    for idx in range(start_idx, end_idx):
        output_path = args.output
        if output_path is None:
            output_path = f'coco_pseudolabel_vis_{args.image_set}_{idx}.png'
        elif args.n_samples > 1:
            output_path = f'{args.output}_{idx}.png'
        
        try:
            visualize_coco_sample(
                coco_dataset,
                idx,
                str(pseudolabels_dir),
                output_path=output_path
            )
        except FileNotFoundError as e:
            print(f"⚠ Skipping sample {idx}: {e}")
            continue
        except Exception as e:
            print(f"❌ Error visualizing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ Visualization complete!")


def visualize_sample_simple(coco_root='coco', pseudolabels_dir='pseudolabels_coco', index=0, image_set='train', year='2014'):
    """
    Simple function to visualize a single sample without command line arguments.
    
    Example usage:
        visualize_sample_simple(coco_root='coco', pseudolabels_dir='pseudolabels_coco', index=0)
    """
    # Load COCO dataset
    print(f"Loading COCO {year} {image_set} dataset...")
    coco_dataset = COCOSegmentation(
        root=coco_root,
        year=year,
        image_set=image_set,
        n_images=-1
    )
    print(f"✓ Loaded {len(coco_dataset)} images")
    
    # Visualize
    visualize_coco_sample(
        coco_dataset,
        index,
        pseudolabels_dir,
        output_path=f'coco_pseudolabel_vis_{image_set}_{index}.png'
    )


if __name__ == '__main__':
    main()

