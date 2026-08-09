from typing import Sequence

import clip
import numpy as np
import torch
import torch.nn.functional as F

from model.clip_wsss import embed_patches, imagenet_to_clip, interpolate_pos_embed

PROMPT_TEMPLATE = "a clean origami {}."

FG_CLASS_NAMES_VOC: Sequence[str] = [
    "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "table", "dog", "horse", "motorcycle", "people", "potted plant", "sheep", "sofa", "train",
    "television receiver",
]
BG_CLASS_NAMES_VOC: Sequence[str] = [
    "ground", "land", "grass", "tree", "building", "wall", "sky", "lake", "water", "river", "sea",
    "railway", "railroad", "keyboard", "helmet", "cloud", "house", "mountain", "ocean", "road",
    "rock", "street", "valley", "bridge", "sign",
]

FG_CLASS_NAMES_COCO: Sequence[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]
BG_CLASS_NAMES_COCO: Sequence[str] = [
    "ground", "land", "grass", "tree", "building", "wall", "sky", "lake", "water", "river", "sea",
    "railway", "railroad", "helmet", "cloud", "house", "mountain", "ocean", "road",
    "rock", "street", "valley", "bridge",
]


@torch.no_grad()
def build_text_embeddings(cfg, clip_model, device):
    """Precompute normalized CLIP text embeddings for a dataset's fg+bg class names.
    Returns (text_emb_all [num_fg + num_bg, D], num_fg, num_bg)."""
    if cfg.dataset.dataset_name == "voc":
        fg, bg = FG_CLASS_NAMES_VOC, BG_CLASS_NAMES_VOC
    elif cfg.dataset.dataset_name == "coco":
        fg, bg = FG_CLASS_NAMES_COCO, BG_CLASS_NAMES_COCO
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.dataset_name}")

    prompts = [PROMPT_TEMPLATE.format(name) for name in fg + bg]
    tokens = clip.tokenize(prompts).to(device)
    embs = clip_model.encode_text(tokens).float()
    return F.normalize(embs, dim=-1), len(fg), len(bg)  # [num_fg + num_bg, D]


class ClipGradCAM:
    """Softmax-GradCAM pseudo-label generator, following CLIP-ES (Lin et al., CVPR 2023):
    freeze all but the last visual transformer block, backprop the softmax class probability
    into that block's attention input, and weight activations by the spatial-mean gradient."""

    def __init__(self, clip_model, img_size: int = 448, bg_exponent: float = 1.0):
        self.clip = clip_model
        self.visual = clip_model.visual
        patch_size = self.visual.conv1.kernel_size[0]
        self.gh = self.gw = img_size // patch_size
        self.pos_embed = interpolate_pos_embed(self.visual.positional_embedding, (self.gh, self.gw))
        self.bg_exponent = bg_exponent

    def _penultimate(self, image: torch.Tensor) -> torch.Tensor:
        tokens = embed_patches(self.visual, imagenet_to_clip(image), self.pos_embed)
        for block in self.visual.transformer.resblocks[:-1]:
            tokens = block(tokens)
        return tokens  # LND

    def _target_probs(self, x_ln: torch.Tensor, act: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        block = self.visual.transformer.resblocks[-1]
        h = act + block.attn(x_ln, x_ln, x_ln, need_weights=False, attn_mask=None)[0]
        h = h + block.mlp(block.ln_2(h))
        h = self.visual.ln_post(h.permute(1, 0, 2))
        pooled = h[:, 1:, :].mean(dim=1)
        if self.visual.proj is not None:
            pooled = pooled @ self.visual.proj
        image_feat = F.normalize(pooled, dim=-1)
        logits = self.clip.logit_scale.exp() * image_feat @ text_emb.t()
        return logits.softmax(dim=-1)  # [1, C]

    def _cams_for_image(self, image: torch.Tensor, text_emb: torch.Tensor, num_targets: int) -> torch.Tensor:
        """image: [1, 3, H, W] (ImageNet-normalized); text_emb: [C, D] normalized, target classes first
        (may include trailing non-target classes used only as softmax competitors).
        Returns raw (ReLU'd, un-normalized) CAMs for the first `num_targets` classes: [num_targets, gh, gw]."""
        with torch.no_grad():
            act = self._penultimate(image)
        block = self.visual.transformer.resblocks[-1]

        cams = []
        for idx in range(num_targets):
            x_ln = block.ln_1(act).detach().requires_grad_(True)
            probs = self._target_probs(x_ln, act, text_emb)
            grad = torch.autograd.grad(probs[0, idx], x_ln)[0]  # [L, 1, W]
            weight = grad[1:].mean(dim=0)[0]  # [W]
            cam = F.relu((weight * x_ln[1:, 0].detach()).sum(-1)).reshape(self.gh, self.gw)  # [gh, gw]
            cams.append(cam)
        return torch.stack(cams)

    @torch.enable_grad()
    def __call__(self, images, targets, text_emb_all, num_all_fg, num_bg):
        """Per-image softmax-GradCAM pseudolabels for a batch. Returns (pseudolabels_batch,
        class_indices_batch) where pseudolabels_batch[b] is [num_fg_present + 1, gh, gw]
        (min-max normalized fg CAMs; last channel = background = (1 - max over fg CAMs) ** bg_exponent,
        CLIP-ES style)."""
        pseudolabels_batch, class_indices_batch = [], []
        for b in range(images.shape[0]):
            target_np = targets[b].cpu().numpy()
            class_indices = np.unique(target_np)
            class_indices = class_indices[(class_indices != 255) & (class_indices != 0)]

            image = images[b : b + 1]
            if len(class_indices) == 0:
                pix_prob = torch.ones((1, self.gh, self.gw), dtype=torch.float32, device=image.device)
            else:
                fg_idx_in_all = [int(c) - 1 for c in class_indices]
                bg_idx_in_all = list(range(num_all_fg, num_all_fg + num_bg))
                # Background classes are included as softmax competitors (sharpens fg CAMs)
                # but are not themselves backprop targets, following CLIP-ES.
                sub_text_emb = text_emb_all[fg_idx_in_all + bg_idx_in_all]
                fg_cams_raw = self._cams_for_image(
                    image, sub_text_emb, num_targets=len(fg_idx_in_all)
                )  # [num_fg, gh, gw], raw (un-normalized)

                fg_min = fg_cams_raw.amin(dim=(1, 2), keepdim=True)
                fg_max = fg_cams_raw.amax(dim=(1, 2), keepdim=True)
                fg_cams = (fg_cams_raw - fg_min) / (fg_max - fg_min + 1e-8)

                # CLIP-ES style background score: (1 - max over fg classes) ** exponent.
                bg_cam = (1 - fg_cams.amax(dim=0, keepdim=True)) ** self.bg_exponent

                pix_prob = torch.cat([fg_cams, bg_cam], dim=0).detach()

            pseudolabels_batch.append(pix_prob)
            class_indices_batch.append(class_indices)
        return pseudolabels_batch, class_indices_batch


def apply_hard_supervision(probs: torch.Tensor, percentage: float) -> tuple:
    """probs: [k, H, W] per-pixel soft probabilities over the k classes present in an image
    (including background). For each class, hardens its top percentage/k share of pixels
    (ranked by that class's confidence) into a one-hot label; pixels left unclaimed keep their
    original soft pseudolabel probabilities.
    Returns (hardened_probs [k, H, W], hard_class_map [H, W] long, -1 where left as soft pseudolabel
    else the channel index it was hardened to)."""
    k, H, W = probs.shape
    n_per_class = max(1, round(percentage / k * H * W))

    hardened = probs.clone()
    flat_hardened = hardened.view(k, -1)
    claimed = torch.zeros(H * W, dtype=torch.bool, device=probs.device)
    hard_class_map = torch.full((H * W,), -1, dtype=torch.long, device=probs.device)

    flat_probs = probs.view(k, -1)
    for c in range(k):
        available = (~claimed).nonzero(as_tuple=True)[0]
        if available.numel() == 0:
            break
        n = min(n_per_class, available.numel())
        top_local = torch.topk(flat_probs[c, available], n).indices
        chosen = available[top_local]

        one_hot = torch.zeros(k, dtype=probs.dtype, device=probs.device)
        one_hot[c] = 1.0
        flat_hardened[:, chosen] = one_hot.unsqueeze(1)

        claimed[chosen] = True
        hard_class_map[chosen] = c

    return hardened, hard_class_map.view(H, W)
