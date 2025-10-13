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
from sklearn.cluster import MiniBatchKMeans          #  <-- new dep
import os

os.environ["OPENBLAS_NUM_THREADS"] = "64"

# --------------------------------------------------------------------------- #
# DINOv3 import (local clone or pip-installed package)                         #
# --------------------------------------------------------------------------- #
DINOV3_LOCATION = '/u501/j234li/wsss/model/dinov3'
import sys
sys.path.append(DINOV3_LOCATION)
from dinov3.data.transforms import make_classification_eval_transform

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"
CLASS_NAMES: Sequence[str] = [
    "person",
    "plane",
    "ground", "land", "grass", "tree", "building", "wall", "sky", "lake", "water", "river", "sea",
    "railway", "railroad", "keyboard", "helmet", "cloud", "house", "mountain", "ocean", "road",
    "rock", "street", "valley", "bridge", "sign"
]
PROMPT_TEMPLATES: Tuple[str, ...] = (
    "a photo of {}", "an image of {}", "a photograph of {}", "a picture of {}",
    "a photo of a {}", "an image of a {}", "a photo of the {}", "an image of the {}",
    "a close-up photo of {}", "a cropped image featuring {}",
)

CANONICAL_SIZE = (480, 640)                 # (H, W)
CROP_AREAS      = [0.003, 0.01, 0.1, 0.3, 1]        # 1 %, 10 %, 100 % of image area
CROP_JITTER     = 0.10                      # 10 % coordinate noise
NUM_CLUSTERS    = 128
MAX_KMEANS_SAMPLES = 20_000                 # subsample for speed

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
OUTPUT_DIR    = Path(".")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DINO_EMBED_DIM = 2048

# --------------------------------------------------------------------------- #
# Helper utilities                                                             #
# --------------------------------------------------------------------------- #
def download_image(url: str) -> Image.Image:
    return Image.open('/u501/j234li/wsss/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg').convert("RGB")
    # with urllib.request.urlopen(url) as resp:
    #     return Image.open(resp).convert("RGB")

class Denormalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(mean)[:, None, None]
        self.std  = torch.tensor(std)[:, None, None]
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t * self.std + self.mean

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
    embs  = model.encode_text(toks)             # [N, D]
    C, D  = len(class_names), embs.size(1)
    agg   = torch.zeros(C, D, device=embs.device)
    cnt   = torch.zeros(C,    device=embs.device)
    for i, c in enumerate(owners):
        agg[c] += embs[i];   cnt[c] += 1
    return F.normalize(agg / cnt.unsqueeze(1), p=2, dim=1)  # [C, D]

# ----------------------------- crop generator ------------------------------ #
def generate_crops(h: int, w: int) -> List[Tuple[int,int,int,int]]:
    """Dense sliding-window squares with jitter ≈ quadrilateral crops."""
    crops = []
    for area in CROP_AREAS:
        side  = int(round(math.sqrt(area * h * w)))
        side  = max(side, 32)                       # safety
        stride = max(8, side // 2)
        for y in range(0, h - side + 1, stride):
            for x in range(0, w - side + 1, stride):
                jx = int(random.uniform(-CROP_JITTER, CROP_JITTER) * side)
                jy = int(random.uniform(-CROP_JITTER, CROP_JITTER) * side)
                x0 = min(max(x + jx, 0), w - side)
                y0 = min(max(y + jy, 0), h - side)
                crops.append((x0, y0, x0 + side, y0 + side))
    print("Generated crops: ", len(crops))
    return crops

# ---------------------------- feature encoder ------------------------------ #
def encode_patches(model, img_tensor: torch.Tensor) -> torch.Tensor:
    # returns [1, P, D]
    ctx = torch.autocast("cuda", dtype=torch.float) if DEVICE.type=="cuda" else contextlib.nullcontext()
    with torch.no_grad(), ctx:
        image_class_tokens, image_patch_tokens, backbone_patch_tokens = model.encode_image_with_patch_tokens(img_tensor)
    return image_class_tokens

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
        class_tokens = encode_patches(model, crop_tensor)           # [1, D]
        grid = F.interpolate(class_tokens.unsqueeze(2).unsqueeze(3), size=(y1 - y0, x1 - x0),
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
    centers  = torch.from_numpy(kmeans.cluster_centers_).to(feat_map.device)  # [k,D]
    return labels.reshape(H, W), F.normalize(centers.float(), p=2, dim=1)

def centroid_zero_shot(centers: torch.Tensor, text_emb: torch.Tensor) -> np.ndarray:
    """
    Cos-sim on centroids → class id for each centroid → returns [k] numpy ints.
    """
    sim   = torch.einsum("kd,cd->kc", centers, text_emb)      # [k, C]
    return sim.argmax(1).cpu().numpy()                        # [k]

# -------------------------- visualisation helpers -------------------------- #
def save_reference(pil_image: Image.Image, fname: Path) -> np.ndarray:
    """Save RGB reference at canonical size without any extra warping."""
    # pil_image is already 640×480
    np_img = np.asarray(pil_image).astype(np.float32) / 255.0
    plt.imsave(fname, np_img);  return np_img

def save_overlay(img_np: np.ndarray, idx_map: np.ndarray, labels: Sequence[str],
                 fname: Path, alpha: float = 0.7):
    H, W = idx_map.shape
    cmap = plt.get_cmap("gist_ncar", len(labels))
    seg = np.zeros((H, W, 4), dtype=np.float32)
    for i in range(len(labels)):
        mask = idx_map == i
        seg[mask] = (*cmap(i)[:3], alpha)
    overlay = np.array(seg)
    overlay[..., :3] *= img_np
    plt.figure(figsize=(6,4));  plt.imshow(overlay); plt.axis("off")
    for i, txt in enumerate(labels):
        plt.text(10, 40+30*i, txt, color=cmap(i)[:3], fontsize=10,
                 bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"))
    plt.tight_layout(pad=0); plt.savefig(fname.with_name(fname.name + 'overlay.png'), dpi=300); plt.close()

    plt.figure(figsize=(6,4));  plt.imshow(seg); plt.axis("off")
    plt.tight_layout(pad=0); plt.savefig(fname.with_name(fname.name + 'seg.png'), dpi=300); plt.close()

# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    random.seed(0); torch.manual_seed(0)
    model, tokenizer = prepare_model()
    preprocess = make_classification_eval_transform()

    # 1. load & canonically resize once for cropping geometry ----------------
    pil_img = download_image(IMAGE_URL).resize((CANONICAL_SIZE[1], CANONICAL_SIZE[0]))
    # 2. prompt-ensemble text embeddings -------------------------------------
    text_emb = build_text_embeddings(model, tokenizer, CLASS_NAMES)           # [C,D]
    # 3. sliding-window feature aggregation ----------------------------------
    feat_map = aggregate_features(model, preprocess, pil_img)                 # [D,H,W]
    # 4. k-means on per-pixel features ---------------------------------------
    pix_labels, centroids = run_kmeans_on_pixels(feat_map)                    # [H,W], [k,D]
    # 5. zero-shot classify centroids, propagate to pixels -------------------
    centroid2cls = centroid_zero_shot(centroids, text_emb)                    # [k]
    pred_map     = centroid2cls[pix_labels]                                   # [H,W]

    # 6. visuals -------------------------------------------------------------
    OUTPUT_DIR.mkdir(exist_ok=True)
    ref_np = save_reference((pil_img), OUTPUT_DIR/"dino_txt_img.png")
    save_overlay(ref_np, pred_map, CLASS_NAMES, OUTPUT_DIR/"dino_txt_")

if __name__ == "__main__":
    main()