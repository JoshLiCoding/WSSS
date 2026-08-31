"""Evaluate a checkpoint on the VOC/COCO val set, with optional test-time augmentation."""
import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dino import DinoWSSS
from utils.dataset import CustomSegmentationVal, build_dataset
from utils.metrics import update_miou

VOC_CLASS_NAMES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"]

COCO_CLASS_NAMES = ["background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', help='path to the model checkpoint (.pt)')
    parser.add_argument('--dataset', choices=('voc', 'coco'), default='voc')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--flip', action='store_true',
                        help='average the segmentation of the image and of its horizontal flip')
    parser.add_argument('--resize', type=int, default=None,
                        help='single input resolution, overriding dataset.resize_size')
    parser.add_argument('--device', default=None)
    return parser.parse_args()


def predict(model, image, flip):
    """Segmentation logits for a [1, 3, H, W] batch, optionally flip-averaged."""
    logits = model(image)['seg']
    if flip:
        flipped = model(torch.flip(image, dims=[-1]))['seg']
        logits = (logits + torch.flip(flipped, dims=[-1])) / 2
    return logits


def main():
    args = parse_args()
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    cfg = OmegaConf.load(args.config)
    cfg.dataset.dataset_name = args.dataset
    class_names = VOC_CLASS_NAMES if args.dataset == 'voc' else COCO_CLASS_NAMES
    num_classes = len(class_names)
    resize_size = args.resize or cfg.dataset.resize_size

    val_dataset = CustomSegmentationVal(build_dataset(cfg, 'val'), resize_size=resize_size)

    model = DinoWSSS(
        backbone_name=cfg.model.backbone_name,
        num_transformer_blocks=cfg.model.num_transformer_blocks,
        num_conv_blocks=cfg.model.num_conv_blocks,
        out_channels=num_classes,
        use_bottleneck=cfg.model.use_bottleneck,
        use_transpose_conv=cfg.model.use_transpose_conv,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    print(f"Evaluating {args.dataset} val at {resize_size}px"
          f"{', flip-averaged' if args.flip else ''} ({len(val_dataset)} images)")

    intersection_counts = np.zeros(num_classes)
    union_counts = np.zeros(num_classes)
    with torch.no_grad():
        for image, target in tqdm(val_dataset, desc="Evaluating"):
            segmentation = predict(model, image.unsqueeze(0).to(device), args.flip)
            update_miou(segmentation, target.unsqueeze(0).to(device),
                        intersection_counts, union_counts, num_classes, cfg.training.ignore_index)

    ious = []
    for cls in range(num_classes):
        if union_counts[cls] == 0:
            continue
        iou = intersection_counts[cls] / union_counts[cls]
        ious.append(iou)
        print(f"Class {class_names[cls]} IoU: {iou:.4f}")
    print(f"\nmIoU: {np.mean(ious):.4f}")


if __name__ == "__main__":
    main()
