"""Mask-boundary generation for the Potts pairwise loss.

Boundaries are cached once per image, at the image's own resolution, as a two-channel
edge map: channel 0 marks a boundary between pixel (i, j) and (i, j+1), channel 1 between
(i, j) and (i+1, j). The training dataset crops/flips/downsamples that map alongside the
image so the pairwise loss sees the boundaries of the augmented view.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def _edges_from_labels(labels):
    """Edge map of a label image: a boundary wherever neighbouring labels differ."""
    edges = np.zeros((2,) + labels.shape, dtype=bool)
    edges[0, :, :-1] = labels[:, :-1] != labels[:, 1:]
    edges[1, :-1, :] = labels[:-1, :] != labels[1:, :]
    return edges


def _edges_from_masks(masks, shape):
    """Union of the outlines of a set of (possibly overlapping) binary masks."""
    edges = np.zeros((2,) + shape, dtype=bool)
    for mask in masks:
        mask = np.asarray(mask).astype(bool)
        if mask.shape != shape:  # e.g. a letterboxed detector output
            resized = Image.fromarray(mask.astype(np.uint8)).resize(
                (shape[1], shape[0]), Image.Resampling.NEAREST
            )
            mask = np.array(resized).astype(bool)
        edges[0, :, :-1] |= mask[:, :-1] != mask[:, 1:]
        edges[1, :-1, :] |= mask[:-1, :] != mask[1:, :]
    return edges


class BoundaryGenerator:
    """Computes a boundary map for one image with the configured method."""

    def __init__(self, method, device='cuda', sam_checkpoint='sam_checkpoint/sam_vit_b_01ec64.pth'):
        self.method = method
        self.device = device
        self.model = None

        if method == 'fastsam':
            from ultralytics import FastSAM
            self.model = FastSAM('FastSAM-x.pt')
        elif method == 'sam':
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
            sam = sam_model_registry['vit_b'](checkpoint=sam_checkpoint).to(device)
            self.model = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=16,
                points_per_batch=256,
                pred_iou_thresh=0.8,
                stability_score_thresh=0.8,
            )
        elif method not in ('slic', 'gt'):
            raise ValueError(f"Unknown contour method: {method}")

    def __call__(self, image, target):
        """image: PIL RGB image; target: PIL segmentation mask. Returns (2, H, W) bool."""
        image_np = np.array(image)
        shape = image_np.shape[:2]

        if self.method == 'gt':
            return _edges_from_labels(np.array(target))

        if self.method == 'slic':
            from skimage.segmentation import slic
            labels = slic(
                image_np.astype(np.float32) / 255.0,
                n_segments=100, compactness=10.0, sigma=0.0, start_label=0, channel_axis=-1,
            )
            return _edges_from_labels(labels)

        if self.method == 'fastsam':
            results = self.model(
                image_np, device=self.device, retina_masks=True, imgsz=1024,
                conf=0.1, iou=0.7, verbose=False,
            )
            masks = results[0].masks
            masks = [] if masks is None else masks.data.cpu().numpy()
            return _edges_from_masks(masks, shape)

        masks = [mask['segmentation'] for mask in self.model.generate(image_np)]
        return _edges_from_masks(masks, shape)


def ensure_boundaries(cfg, dataset, out_dir, device):
    """Generate the boundary maps missing from out_dir, then leave them cached there
    as one `{image name}.npz` per image holding the `edges` map."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i, name in enumerate(dataset.names) if not (out_dir / f'{name}.npz').exists()]
    if not todo:
        print(f"Using cached {cfg.loss.contour_method} boundaries for {len(dataset)} images in {out_dir}")
        return
    print(f"Generating {cfg.loss.contour_method} boundaries for {len(todo)}/{len(dataset)} images into {out_dir}")

    generator = BoundaryGenerator(cfg.loss.contour_method, device=device)
    for index in tqdm(todo, desc=f"{cfg.loss.contour_method} boundaries"):
        image, target = dataset[index]
        edges = generator(image, target)
        np.savez_compressed(out_dir / f'{dataset.names[index]}.npz', edges=edges)
