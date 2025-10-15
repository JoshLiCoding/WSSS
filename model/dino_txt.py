# Please see https://github.com/facebookresearch/dinov2/issues/530, credits go to @jbdel

from __future__ import annotations
import contextlib, itertools, math, random, sys, urllib.request
from pathlib import Path
from typing import Sequence, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from sklearn.cluster import MiniBatchKMeans 
import os
from tqdm import tqdm

os.environ["OPENBLAS_NUM_THREADS"] = "64"
DINOV3_LOCATION = '/u501/j234li/wsss/model/dinov3'
import sys
sys.path.append(DINOV3_LOCATION)
from dinov3.data.transforms import make_classification_eval_transform

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
CLASS_NAMES = {0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person", 16: "potted plant", 17: "sheep", 18: "sofa", 19: "train", 20: "tv/monitor", 255: "ignore"}
BACKGROUND_CLASS_NAMES: Sequence[str] = [
    "ground", "land", "grass", "tree", "building", "wall", "sky", "lake", "water", "river", "sea",
    "railway", "railroad", "keyboard", "helmet", "cloud", "house", "mountain", "ocean", "road",
    "rock", "street", "valley", "bridge", "sign"
]
PROMPT_TEMPLATES: Tuple[str, ...] = (
    "a photo of {}", "an image of {}", "a photograph of {}", "a picture of {}",
    "a photo of a {}", "an image of a {}", "a photo of the {}", "an image of the {}",
    "a close-up photo of {}", "a cropped image featuring {}",
)

CANONICAL_SIZE = (448, 448)
CROP_AREAS      = [0.01]
NUM_CLUSTERS    = 128
MAX_KMEANS_SAMPLES = 20_000

OUTPUT_DIR    = Path("pseudolabels")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DINO_EMBED_DIM = 1024

def prepare_model():
    model, tokenizer = torch.hub.load(DINOV3_LOCATION,
            'dinov3_vitl16_dinotxt_tet1280d20h24l', 
            source='local',
            weights=os.path.join(DINOV3_LOCATION, 'dinov3_vitl16_dinotxt_vision_head_and_text_encoder.pth'), 
            backbone_weights=os.path.join(DINOV3_LOCATION, 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'))
    model = model.to(DEVICE)
    return model, tokenizer

# ----------------------------- text embeddings ----------------------------- #
@torch.no_grad()
def build_text_embeddings(model, tokenizer, class_names: Sequence[str]) -> torch.Tensor:
    prompts, owners = [], []
    for c_idx, name in enumerate(class_names):
        for tpl in PROMPT_TEMPLATES:
            prompts.append(tpl.format(name))
            owners .append(c_idx)
    toks  = tokenizer.tokenize(prompts).to(DEVICE)
    embs  = model.encode_text(toks)[:, 1024:]             # [N, D]
    C, D  = len(class_names), embs.size(1)
    agg   = torch.zeros(C, D, device=embs.device)
    cnt   = torch.zeros(C,    device=embs.device)
    for i, c in enumerate(owners):
        agg[c] += embs[i];   cnt[c] += 1
    return F.normalize(agg / cnt.unsqueeze(1), p=2, dim=1)  # [C, D]

# ----------------------------- crop generator ------------------------------ #
def generate_crops(h: int, w: int) -> List[Tuple[int,int,int,int]]:
    """Dense sliding-window squares"""
    crops = []
    for area in CROP_AREAS:
        side  = int(round(math.sqrt(area * h * w)))
        side  = max(side, 32)                       # safety
        stride = max(8, side // 2)
        # extra crop to fill out borders
        for y in list(range(0, h - side + 1, stride)) + [h-side]:
            for x in list(range(0, w - side + 1, stride)) + [w-side]:
                crops.append((x, y, x + side, y + side))
    print("Generated crops: ", len(crops))
    return crops

# ---------------------------- feature encoder ------------------------------ #
def encode_patches(model, img_tensor: torch.Tensor) -> torch.Tensor:
    # returns [1, P, D]
    ctx = torch.autocast("cuda", dtype=torch.float) if DEVICE.type=="cuda" else contextlib.nullcontext()
    with torch.no_grad(), ctx:
        image_class_tokens, image_patch_tokens, backbone_patch_tokens = model.encode_image_with_patch_tokens(img_tensor)
    return image_patch_tokens

# ----------------------- aggregate window features ------------------------- #
@torch.no_grad()
def aggregate_features(
    model, preprocess, pil_image: Image.Image
) -> torch.Tensor:                                # → [D, H, W]
    H, W  = CANONICAL_SIZE
    feat_sum = torch.zeros((DINO_EMBED_DIM, H, W), device=DEVICE)  # 1280 × H × W
    hit_cnt  = torch.zeros((H, W), device=DEVICE)

    for (x0, y0, x1, y1) in generate_crops(H, W):
        crop = pil_image.crop((x0, y0, x1, y1))
        crop_tensor = preprocess(crop).unsqueeze(0).to(DEVICE)      # [1, 3, *, *]
        patch_tokens = encode_patches(model, crop_tensor)[0]        # [P, D]
        p = int(math.sqrt(patch_tokens.size(0)))                    # √P
        assert p * p == patch_tokens.size(0), "non-square patch grid"

        grid = (
            patch_tokens.movedim(1, 0)         # swap (P, D) → (D, P)
            .unflatten(1, (p, p))              # unflatten the *token* dim (now dim 1)
        )
        grid = F.interpolate(grid.unsqueeze(0), size=(y1 - y0, x1 - x0),
                             mode="bilinear", align_corners=False)[0]  # [D, h, w]
        feat_sum[:, y0:y1, x0:x1] += grid
        hit_cnt [  y0:y1, x0:x1] += 1

    # avoid div-by-0 (shouldn't happen, but safety)
    hit_cnt = torch.clamp(hit_cnt, min=1)
    return feat_sum / hit_cnt                       # [D, H, W]

# ----------------------- k-means + zero-shot classifier -------------------- #
def run_kmeans_on_pixels(feat_map: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor]:
    """MiniBatch-KMeans; returns (H×W labels, centroids [k,D])."""
    D, H, W   = feat_map.shape
    flat      = feat_map.permute(1,2,0).reshape(-1, D).cpu().numpy()
    sample_ix = np.random.choice(len(flat),
                                 size=min(MAX_KMEANS_SAMPLES, len(flat)),
                                 replace=False)
    kmeans = MiniBatchKMeans(n_clusters=NUM_CLUSTERS, batch_size=4096,
                             n_init=3, random_state=0).fit(flat[sample_ix])
    labels   = kmeans.predict(flat).astype(np.int16)
    centers  = torch.from_numpy(kmeans.cluster_centers_).to(feat_map.device)  # [k, D]
    return labels.reshape(H, W), F.normalize(centers.float(), p=2, dim=1)

def centroid_zero_shot(centers: torch.Tensor, text_emb: torch.Tensor, class_names: list) -> np.ndarray:
    """
    Cos-sim on centroids → class id for each centroid → returns [k] numpy ints.
    """
    sim   = torch.einsum("kd,cd->kc", centers, text_emb)                       # [k, C]
    # max over background classes
    bg_sim = torch.max(sim[:, len(class_names):], dim=1, keepdim=True).values  # [k, 1]
    sim = torch.cat([sim[:, :len(class_names)], bg_sim], dim=1)                # [k, C']

    return sim.cpu().numpy()
    # sim *= scale
    # return sim.softmax(dim=1).cpu().numpy()

def generate_pseudolabels(dataset):
    random.seed(0); torch.manual_seed(0)
    model, tokenizer = prepare_model()
    preprocess = make_classification_eval_transform()
    for i, (img, target) in tqdm(enumerate(dataset), desc="Generating pseudo-labels"):
        # 1. load & canonically resize once for cropping geometry ----------------
        pil_img = img.resize((CANONICAL_SIZE[1], CANONICAL_SIZE[0]))
        class_indices = np.unique(np.array(target))
        class_indices = class_indices[(class_indices != 255) & (class_indices != 0)] # remove ignore index and class 0 (background)
        class_names = [CLASS_NAMES[i] for i in class_indices]
        # 2. prompt-ensemble text embeddings -------------------------------------
        text_emb = build_text_embeddings(model, tokenizer, class_names + BACKGROUND_CLASS_NAMES)
        # 3. sliding-window feature aggregation ----------------------------------
        feat_map = aggregate_features(model, preprocess, pil_img)
        # 4. k-means on per-pixel features ---------------------------------------
        pix_labels, centroids = run_kmeans_on_pixels(feat_map)
        # 5. zero-shot classify centroids, propagate to pixels -------------------
        centroid_logits = centroid_zero_shot(centroids, text_emb, class_names)
        pix_prob    = centroid_logits[pix_labels]                                    # [H, W, C']
        
        OUTPUT_DIR.mkdir(exist_ok=True)
        np.save(OUTPUT_DIR/f'pseudolabels_{i}.npy', pix_prob)
        np.save(OUTPUT_DIR/f'class_indices_{i}.npy', class_indices)