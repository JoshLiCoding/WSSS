"""
Evaluate a trained checkpoint on the validation set using test-time augmentation (TTA).
Computes mIoU with multi-scale inference and horizontal flip augmentation.
"""
import argparse
import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    _CRF_AVAILABLE = True
except ImportError:
    _CRF_AVAILABLE = False

# Add parent directory to path to allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dino import DinoWSSS
from utils.dataset import CustomSegmentationValTTA, build_dataset, MEAN, STD
from utils.metrics import update_miou, test_time_augmentation_inference

DINOV3_LOCATION = '/u501/j234li/reg_loss/model/dinov3'
sys.path.append(DINOV3_LOCATION) 

VOC_CLASS_NAMES = {
    0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat",
    5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow",
    11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person",
    16: "potted plant", 17: "sheep", 18: "sofa", 19: "train", 20: "tv/monitor",
    255: "ignore"
}

COCO_CLASS_NAMES = {
    0: "background", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle",
    5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat",
    10: "traffic light", 11: "fire hydrant", 12: "stop sign", 13: "parking meter",
    14: "bench", 15: "bird", 16: "cat", 17: "dog", 18: "horse", 19: "sheep",
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

BASE_SIZE = (448, 448)
DEFAULT_TTA_SCALES = [4.0]


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
    """Load configuration from YAML file"""
    return OmegaConf.load(config_path)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate checkpoint with test-time augmentation'
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

    # Config parameters
    num_classes = config['model']['num_classes']
    ignore_index = config['training']['ignore_index']
    dataset_name = config['dataset']['dataset_name']
    class_names = VOC_CLASS_NAMES if dataset_name == 'voc' else COCO_CLASS_NAMES

    original_val_dataset = build_dataset(config, 'val')

    val_dataset = CustomSegmentationValTTA(original_val_dataset)

    # Initialize model
    model = DinoWSSS(
        backbone_name=config['model']['backbone_name'],
        num_transformer_blocks=config['model']['num_transformer_blocks'],
        num_conv_blocks=config['model']['num_conv_blocks'],
        out_channels=config['model']['out_channels'],
        use_bottleneck=config['model']['use_bottleneck'],
        use_transpose_conv=config['model']['use_transpose_conv']
    ).to(device)

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Evaluate with TTA
    intersection_counts = np.zeros(num_classes)
    union_counts = np.zeros(num_classes)

    print(f"Evaluating with TTA (scales: {args.scales}){' + CRF' if args.crf else ''}...")
    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset)), desc="Evaluating"):
            val_image, val_target = val_dataset[idx]
            val_image = val_image.to(device)
            val_target = val_target.to(device)
            target_size = val_target.shape[-2:]

            segmentation = test_time_augmentation_inference(
                model, val_image, target_size,
                scales=args.scales, base_size=BASE_SIZE, device=device
            )
            if args.crf:
                probs = torch.softmax(segmentation, dim=1)[0].cpu().numpy()
                segmentation = dense_crf(probs, val_image.cpu()).to(device)

            update_miou(
                segmentation, val_target.unsqueeze(0),
                intersection_counts, union_counts,
                num_classes, ignore_index
            )

    # Compute per-class IoU and mIoU
    ious = []
    print("\nPer-class IoU:")
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        if union_counts[cls] == 0:
            continue
        iou = intersection_counts[cls] / union_counts[cls]
        ious.append(iou)
        print(f"  {class_names.get(cls, f'class_{cls}')}: {iou:.4f}")

    avg_miou = np.mean(ious) if ious else 0.0
    print(f"\nmIoU (with TTA): {avg_miou:.4f}")


if __name__ == "__main__":
    main()
