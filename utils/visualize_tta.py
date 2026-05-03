"""
Visualize TTA (test-time augmentation) results on Pascal VOC or COCO val images at a fixed interval.
Shows: original image, ground truth, soft segmentation output (at val size), and hard (argmax) output.
"""
import argparse
import os
import sys
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    _CRF_AVAILABLE = True
except ImportError:
    _CRF_AVAILABLE = False

# Add parent directory to path to allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clip

from model.clip import ClipWSSS
from utils.dataset import VOCSegmentation, COCOSegmentation, CustomSegmentationValTTA, MEAN, STD
from utils.metrics import test_time_augmentation_inference
from utils.vis import visualize_soft_probabilities

BASE_SIZE = (224, 224)
DEFAULT_TTA_SCALES = [1.0]


def dense_crf(probs, image, sxy=80, srgb=13, compat=10, n_iters=10):
    """CRF post-process: probs (C,H,W), image (3,H,W) tensor normalized; returns (1,C,H,W) probs."""
    C, H, W = probs.shape
    d = dcrf.DenseCRF2D(W, H, C)
    unary = unary_from_softmax(probs.astype(np.float32))
    unary = np.ascontiguousarray(unary)
    d.setUnaryEnergy(unary)
    img_uint8 = (image.permute(1, 2, 0).numpy() * np.array(STD) + np.array(MEAN)).clip(0, 1)
    img_uint8 = np.ascontiguousarray((img_uint8 * 255).astype(np.uint8))
    d.addPairwiseBilateral(sxy=sxy, srgb=srgb, rgbim=img_uint8, compat=compat)
    d.addPairwiseGaussian(sxy=3, compat=3)
    Q = d.inference(n_iters)
    out = np.array(Q).reshape((C, H, W))
    return torch.from_numpy(out).float().unsqueeze(0)


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize TTA results on validation images at a fixed interval'
    )
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_tta_vis',
        help='Directory to save visualizations (default: output_tta_vis)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=50,
        help='Visualize every N-th image (default: 30)'
    )
    parser.add_argument(
        '--scales',
        type=float,
        nargs='+',
        default=DEFAULT_TTA_SCALES,
        help=f'TTA scale factors (default: {DEFAULT_TTA_SCALES})'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (default: cuda if available, else cpu)'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to visualize (default: no limit)'
    )
    parser.add_argument('--crf', action='store_true', help='Post-process with dense CRF (pydensecrf)')
    args = parser.parse_args()

    if args.crf and not _CRF_AVAILABLE:
        raise ImportError('--crf requires pydensecrf; install with: pip install pydensecrf')

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device(
        args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    config = load_config(args.config)

    num_classes = config['model']['num_classes']
    dataset_name = config['dataset']['dataset_name']

    # Build original validation dataset (for original image and GT)
    if dataset_name == 'voc':
        original_val_dataset = VOCSegmentation(
            config['dataset']['root'],
            image_set=config['dataset']['val_image_set'],
            download=config['dataset']['download']
        )
    elif dataset_name == 'coco':
        original_val_dataset = COCOSegmentation(
            os.path.join(config['dataset']['root'], 'coco'),
            image_set=config['dataset']['val_image_set'],
            download=config['dataset']['download']
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    val_dataset = CustomSegmentationValTTA(original_val_dataset)

    clip_model, _ = clip.load(config["model"]["clip_model_name"], device=device, jit=False)
    clip_model = clip_model.float()
    model = ClipWSSS(
        clip_model,
        num_transformer_blocks=config["model"]["num_transformer_blocks"],
        num_conv_blocks=config["model"]["num_conv_blocks"],
        out_channels=config["model"]["out_channels"],
        use_bottleneck=config["model"]["use_bottleneck"],
        use_transpose_conv=config["model"]["use_transpose_conv"],
    ).to(device)

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    # Indices to visualize: every args.interval, optionally capped by args.max_images
    indices = list(range(0, len(val_dataset), args.interval))
    if args.max_images is not None:
        indices = indices[: args.max_images]

    print(f"Visualizing {len(indices)} images (interval={args.interval}){' + CRF' if args.crf else ''}...")
    with torch.no_grad():
        for idx in tqdm(indices, desc="Visualizing"):
            # Original image and GT from original dataset
            img, gt_mask = original_val_dataset[idx]
            gt_mask_decoded = original_val_dataset.decode_target(gt_mask)

            # Transformed image and target (same spatial size as original for TTA)
            val_image, val_target = val_dataset[idx]
            val_image = val_image.to(device)
            target_size = val_target.shape[-2:]

            # TTA inference: output is [1, num_classes, H, W] at validation image size
            segmentation = test_time_augmentation_inference(
                model, val_image, target_size,
                scales=args.scales, base_size=BASE_SIZE, device=device
            )
            if args.crf:
                probs = torch.softmax(segmentation, dim=1)[0].cpu().numpy()
                segmentation = dense_crf(probs, val_image.cpu()).to(device)
            seg = segmentation[0].cpu()  # [C, H, W]

            # Soft visualization (probability blend). Use softmax=False when CRF was applied (already probs).
            soft_vis = visualize_soft_probabilities(seg, softmax=not args.crf)

            # Hard (argmax) and decode for colored mask
            hard_np = seg.argmax(0).numpy().astype(np.uint8)
            hard_vis = Image.fromarray(hard_np)
            hard_vis = original_val_dataset.decode_target(hard_vis)

            # Figure: 1 row of 4 panels (original, GT, soft TTA, hard TTA)
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[1].imshow(gt_mask_decoded)
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')

            axes[2].imshow(soft_vis)
            axes[2].set_title('Soft TTA Output')
            axes[2].axis('off')

            axes[3].imshow(hard_vis)
            axes[3].set_title('Hard TTA Output (argmax)')
            axes[3].axis('off')

            plt.tight_layout()
            save_path = os.path.join(args.output_dir, f'tta_vis_{idx:05d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

    print(f"Saved {len(indices)} visualizations to {args.output_dir}")


if __name__ == '__main__':
    main()
