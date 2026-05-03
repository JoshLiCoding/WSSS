"""
Run TTA inference on VOC 2012 test set and save predictions as PNGs for server submission.
Uses the same loading and transform as evaluate_tta (CustomSegmentationValTTA), but on test images only.
Saves one indexed PNG per image: pixel values 0–20 (21 classes), VOC colormap embedded in PNG palette metadata.
"""
import argparse
import os
import sys
import torch
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clip

from model.clip import ClipWSSS
from utils.dataset import (
    val_transform_tta,
    cmap,
    DATASET_YEAR_DICT,
)
from utils.metrics import test_time_augmentation_inference

class VOCTestImageDataset(torch.utils.data.Dataset):
    """VOC test split: images only (no masks). Returns (PIL_image, (H, W), image_name)."""

    def __init__(self, root, year='2012', image_set='test'):
        self.root = os.path.expanduser(root)
        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        split_f = os.path.join(voc_root, 'ImageSets', 'Segmentation', image_set.rstrip('\n') + '.txt')
        with open(split_f) as f:
            self.names = [x.strip() for x in f.readlines()]
        self.images = [os.path.join(image_dir, n + '.jpg') for n in self.names]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        w, h = img.size
        return img, (h, w), self.names[idx]


class TestTTADataset(torch.utils.data.Dataset):
    """Wraps VOCTestImageDataset with val_transform_tta. Returns (tensor, (H, W), name)."""

    def __init__(self, test_dataset):
        self.dataset = test_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, size, name = self.dataset[idx]
        x = val_transform_tta(image)
        return x, size, name


def load_config(config_path='config.yaml'):
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='TTA predictions on VOC test set for submission')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint (.pt)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--out-dir', type=str, default='voc_test_preds', help='Output directory for PNGs')
    parser.add_argument('--scales', type=float, nargs='+', default=[1.0], help='TTA scale factors')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    config = load_config(args.config)
    num_classes = config['model']['num_classes']

    # VOC test dataset (images only)
    root = config['dataset']['root']
    year = '2012'
    test_dataset = VOCTestImageDataset(root=root, year=year, image_set='test')
    dataset = TestTTADataset(test_dataset)

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
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    # VOC colormap for palette (256 entries, R,G,B per entry → 768 values for PIL)
    voc_cmap = cmap(256, normalized=False).astype(np.uint8)
    palette = voc_cmap.flatten().tolist()

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc='Predicting'):
            image, (h, w), name = dataset[idx]
            image = image.to(device)
            target_size = (h, w)

            seg_logits = test_time_augmentation_inference(
                model, image, target_size, scales=args.scales, device=device
            )
            pred = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy()
            pred = np.clip(pred, 0, 20).astype(np.uint8)

            # Indexed image with colormap embedded in PNG metadata (like MATLAB imwrite(im, cmap, path))
            img_p = Image.fromarray(pred, mode='P')
            img_p.putpalette(palette)
            img_p.save(os.path.join(args.out_dir, name + '.png'))

    print(f"Saved {len(dataset)} predictions to {args.out_dir}")


if __name__ == '__main__':
    main()
