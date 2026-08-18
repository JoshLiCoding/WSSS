# Please see https://github.com/facebookresearch/dinov2/issues/530, credits go to @jbdel

import math

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Prompts                                                                      #
# --------------------------------------------------------------------------- #

PROMPT_TEMPLATES = (
    "a photo of {}", "an image of {}", "a photograph of {}", "a picture of {}",
    "a photo of a {}", "an image of a {}", "a photo of the {}", "an image of the {}",
    "a close-up photo of {}", "a cropped image featuring {}",
)

VOC_FG_CLASS_NAMES = [
    "airplane", "bicycle frame and seat", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "table", "dog", "horse", "motorcycle", "people", "potted plant", "sheep",
    "sofa", "train", "television receiver",
]

COCO_FG_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

BG_CLASS_NAMES = [
    "ground", "land", "grass", "tree", "building", "wall", "sky", "lake", "water", "river",
    "sea", "railway", "railroad", "helmet", "cloud", "house", "mountain", "ocean", "road",
    "rock", "street", "valley", "bridge",
]

VOC_BG_CLASS_NAMES = BG_CLASS_NAMES + ["keyboard", "sign"]


def get_class_names(dataset_name):
    """Foreground and background prompt names for a dataset."""
    if dataset_name == 'voc':
        return VOC_FG_CLASS_NAMES, VOC_BG_CLASS_NAMES
    if dataset_name == 'coco':
        return COCO_FG_CLASS_NAMES, BG_CLASS_NAMES
    raise ValueError(f"Unknown dataset: {dataset_name}")


# ----------------------------- text embeddings ----------------------------- #
@torch.no_grad()
def build_text_embeddings(model, tokenizer, class_names, device) -> torch.Tensor:
    """Prompt-ensembled, L2-normalized text embeddings [C, D] for the given class names."""
    prompts = [tpl.format(name) for name in class_names for tpl in PROMPT_TEMPLATES]
    embs = model.encode_text(tokenizer.tokenize(prompts).to(device))[:, 1024:]  # [C * T, D]
    embs = embs.view(len(class_names), len(PROMPT_TEMPLATES), -1).mean(dim=1)   # [C, D]
    return F.normalize(embs, p=2, dim=1)


# ------------------------------- pseudolabels ------------------------------- #
@torch.no_grad()
def generate_pseudolabels_batch(patch_tokens, present, text_emb_all, num_fg, size,
                                temperature, min_max):
    """Soft pseudo-labels from dino.txt patch tokens.

    Patch tokens are scored against the class prompts by cosine similarity and laid out in
    class space (channel 0 is background, condensed as the max over the background prompts),
    upsampled to `size`, then turned into a distribution over the classes the image is
    tagged with.

    Args:
        patch_tokens: (B, P, D) dino.txt patch tokens.
        present: (B, C) bool mask of the classes each image is tagged with.
        text_emb_all: (num_fg + num_bg, D) normalized text embeddings, foreground first.
        num_fg: number of foreground prompts.
        size: (H, W) resolution to produce the pseudo-labels at.
        temperature: softmax temperature.
        min_max: per-channel min-max scaling over space before renormalizing onto the simplex.

    Returns:
        (B, C, H, W) pseudo-label probabilities; channels of absent classes are exactly zero.
    """
    B, P, _ = patch_tokens.shape
    p = int(math.sqrt(P))
    assert p * p == P, "non-square patch grid"

    sim = F.normalize(patch_tokens, p=2, dim=2) @ text_emb_all.t()  # [B, P, num_fg + num_bg]
    sim = sim.permute(0, 2, 1).view(B, -1, p, p)
    sim = torch.cat([sim[:, num_fg:].amax(dim=1, keepdim=True), sim[:, :num_fg]], dim=1)
    sim = F.interpolate(sim, size=size, mode='bilinear', align_corners=False)

    mask = present[:, :, None, None]
    probs = torch.where(mask, sim / temperature, torch.full_like(sim, float('-inf'))).softmax(dim=1)

    if min_max:
        low = probs.amin(dim=(2, 3), keepdim=True)
        high = probs.amax(dim=(2, 3), keepdim=True)
        probs = (probs - low) / (high - low + 1e-8)
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
        # Min-max has nothing to scale when only background is present; fall back to a hard label.
        single = present.sum(dim=1) == 1
        probs[single] = mask[single].to(probs.dtype).expand_as(probs[single])

    return probs
