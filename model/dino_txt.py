"""dino.txt pseudo-label generation.

Ported from ~/wsss/model/dino_txt_full_img.py. For every image we cosine-match the dino.txt
patch tokens against prompt-ensembled text embeddings of the foreground classes the image
contains plus a fixed bank of background class names, and condense the background names with
a max. What gets cached is the raw similarity grid at the ViT patch resolution; the softmax /
min-max chain that turns it into a soft label runs later, at the resolution the loss is
computed at (see utils/pseudolabels.py).
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


PROMPT_TEMPLATES = (
    "a photo of {}", "an image of {}", "a photograph of {}", "a picture of {}",
    "a photo of a {}", "an image of a {}", "a photo of the {}", "an image of the {}",
    "a close-up photo of {}", "a cropped image featuring {}",
)

# Foreground names are indexed by class id - 1; background names are condensed into one channel.
VOC_FG_NAMES = [
    "airplane", "bicycle frame and seat", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "table", "dog", "horse", "motorcycle", "people", "potted plant", "sheep",
    "sofa", "train", "television receiver",
]
VOC_BG_NAMES = [
    "ground", "land", "grass", "tree", "building", "wall", "sky", "lake", "water", "river",
    "sea", "railway", "railroad", "keyboard", "helmet", "cloud", "house", "mountain", "ocean",
    "road", "rock", "street", "valley", "bridge", "sign",
]

COCO_FG_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
COCO_BG_NAMES = [
    "ground", "land", "grass", "tree", "building", "wall", "sky", "lake", "water", "river",
    "sea", "railway", "railroad", "helmet", "cloud", "house", "mountain", "ocean", "road",
    "rock", "street", "valley", "bridge",
]


def class_names(dataset_name):
    """(foreground names, background names) for a dataset."""
    if dataset_name == 'voc':
        return VOC_FG_NAMES, VOC_BG_NAMES
    if dataset_name == 'coco':
        return COCO_FG_NAMES, COCO_BG_NAMES
    raise ValueError(f"Unknown dataset: {dataset_name}")


def prepare_model(dinov3_location, device):
    """Load the dino.txt ViT-L/16 vision + text encoder from a local dinov3 checkout. The hub
    entry point is imported directly rather than through torch.hub.load, whose hubconf pulls
    in the segmentation models (and their extra dependencies) that we have no use for."""
    if dinov3_location not in sys.path:
        sys.path.append(dinov3_location)
    from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l

    weights_dir = os.path.join(dinov3_location, 'weights')
    model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(
        weights=os.path.join(weights_dir, 'dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth'),
        backbone_weights=os.path.join(weights_dir, 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'),
    )
    return model.to(device).eval(), tokenizer


@torch.no_grad()
def build_text_embeddings(model, tokenizer, names, device):
    """Prompt-ensembled, L2-normalized text embeddings [len(names), D]. Only the second half
    of the joint embedding is used: that is the part aligned with the patch tokens."""
    prompts = [template.format(name) for name in names for template in PROMPT_TEMPLATES]
    tokens = tokenizer.tokenize(prompts).to(device)
    embs = model.encode_text(tokens)[:, 1024:]
    embs = embs.view(len(names), len(PROMPT_TEMPLATES), -1).mean(dim=1)
    return F.normalize(embs, p=2, dim=1)


@torch.no_grad()
def patch_similarities(model, images, text_emb, num_fg):
    """Cosine similarity between dino.txt patch tokens and each class embedding.

    Args:
        images: (B, 3, H, W) dino.txt-preprocessed images.
        text_emb: (num_fg + num_bg, D) normalized text embeddings, foreground classes first.
        num_fg: number of foreground classes.

    Returns:
        (B, num_fg + 1, p, p) similarities on the ViT patch grid, background classes
        condensed into the last channel by a max.
    """
    _, patch_tokens, _ = model.encode_image_with_patch_tokens(images)
    B, P, _ = patch_tokens.shape
    p = int(math.sqrt(P))
    assert p * p == P, "non-square patch grid"

    sim = F.normalize(patch_tokens, p=2, dim=2) @ text_emb.t()  # (B, P, C)
    sim = sim.permute(0, 2, 1).reshape(B, text_emb.shape[0], p, p)
    return torch.cat([sim[:, :num_fg], sim[:, num_fg:].amax(dim=1, keepdim=True)], dim=1)


class _PreprocessedImages(Dataset):
    """Yields dino.txt-preprocessed images alongside their image-level tags."""

    def __init__(self, dataset, preprocess, num_classes):
        self.dataset = dataset
        self.preprocess = preprocess
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        ids = self.dataset.labels(idx)
        present = torch.zeros(self.num_classes, dtype=torch.bool)
        present[torch.from_numpy(ids).long()] = True
        return self.preprocess(image), present, idx


def ensure_pseudolabels(cfg, dataset, out_dir, device):
    """Generate the dino.txt pseudo-labels missing from out_dir, then leave them cached there.

    One `{image name}.npz` per image, holding the raw similarity grid `sim` [K, p, p] (the
    classes present, in ascending order, then background) and the foreground class ids
    `classes` [K-1].
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [
        i for i, name in enumerate(dataset.names)
        if not (out_dir / f'{name}.npz').exists()
    ]
    if not todo:
        print(f"Using cached dino.txt pseudo-labels for {len(dataset)} images in {out_dir}")
        return
    print(f"Generating dino.txt pseudo-labels for {len(todo)}/{len(dataset)} images into {out_dir}")

    num_classes = cfg.model.num_classes
    model, tokenizer = prepare_model(cfg.pseudolabel.dinov3_location, device)

    from dinov3.data.transforms import make_eval_transform  # needs the dinov3 checkout on sys.path
    preprocess = make_eval_transform(
        crop_size=None, resize_square=True, resize_size=cfg.pseudolabel.image_size
    )

    fg_names, bg_names = class_names(cfg.dataset.dataset_name)
    text_emb = build_text_embeddings(model, tokenizer, list(fg_names) + list(bg_names), device)

    loader = DataLoader(
        Subset(_PreprocessedImages(dataset, preprocess, num_classes), todo),
        batch_size=cfg.pseudolabel.batch_size,
        num_workers=cfg.training.num_workers,
    )
    for images, present, indices in tqdm(loader, desc="dino.txt pseudo-labels"):
        sims = patch_similarities(model, images.to(device), text_emb, len(fg_names)).cpu()
        for b, index in enumerate(indices.tolist()):
            classes = present[b].nonzero().flatten()
            sim = torch.cat([sims[b, classes - 1], sims[b, -1:]], dim=0).numpy()
            name = dataset.names[index]
            np.savez(out_dir / f'{name}.npz', sim=sim, classes=classes.numpy().astype(np.int16))
