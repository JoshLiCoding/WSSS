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
CLASS_NAMES: Sequence[str] = [
    "plane", "human"
]
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

CANONICAL_SIZE = (448, 448)                 # (H, W)
CROP_AREAS      = [1]        # 1 %, 10 %, 100 % of image area
CROP_JITTER     = 0.10                      # 10 % coordinate noise
NUM_CLUSTERS    = 128
MAX_KMEANS_SAMPLES = 20_000                 # subsample for speed

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
OUTPUT_DIR    = Path(".")
DEVICE        = torch.device("cpu") # torch.device("cuda" if torch.cuda.is_available() else "cpu")
DINO_EMBED_DIM = 1024

# --------------------------------------------------------------------------- #
# Helper utilities                                                             #
# --------------------------------------------------------------------------- #
def download_image() -> Image.Image:
    return Image.open('/u501/j234li/wsss/VOCdevkit/VOC2012/JPEGImages/2010_003432.jpg').convert("RGB")

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
            weights=os.path.join(DINOV3_LOCATION, 'weights', 'dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth'), 
            backbone_weights=os.path.join(DINOV3_LOCATION, 'weights', 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'))
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
    """Dense sliding-window squares with jitter ≈ quadrilateral crops."""
    crops = []
    for area in CROP_AREAS:
        side  = int(round(math.sqrt(area * h * w)))
        side  = max(side, 32)                       # safety
        stride = max(8, side // 2)
        for y in list(range(0, h - side + 1, stride)):
            for x in list(range(0, w - side + 1, stride)):
                jx = 0 # int(random.uniform(-CROP_JITTER, CROP_JITTER) * side)
                jy = 0 # int(random.uniform(-CROP_JITTER, CROP_JITTER) * side)
                x0 = min(max(x + jx, 0), w - side)
                y0 = min(max(y + jy, 0), h - side)
                crops.append((x0, y0, x0 + side, y0 + side))
                # crops.append((w-(x0+side), h-(y0+side), w-x0, h-y0))  # horiz flip
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
        print( "p: ", p)

        # ---- FIX ----
        grid = (
            patch_tokens.movedim(1, 0)         # swap (P, D) → (D, P)
            .unflatten(1, (p, p))              # unflatten the *token* dim (now dim 1)
        )
        grid = F.interpolate(grid.unsqueeze(0), size=(y1 - y0, x1 - x0),
                             mode="bilinear")[0]  # [D, h, w]
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

def centroid_zero_shot(centers: torch.Tensor, text_emb: torch.Tensor, class_names: list, scale: float = 20.0) -> np.ndarray:
    """
    Cos-sim on centroids → class id for each centroid → returns [k] numpy ints.
    """
    sim   = torch.einsum("kd,cd->kc", centers, text_emb)      # [k, C]
    bg_sim = torch.max(sim[:, len(class_names):], dim=1, keepdim=True)[0]  # [k,1]
    sim = torch.cat([sim[:, :len(class_names)], bg_sim], dim=1)
    print("scale: ", scale)
    sim = sim * scale
    return sim.softmax(dim=1).cpu().numpy()

# -------------------------- visualisation helpers -------------------------- #
def save_reference(pil_image: Image.Image, fname: Path) -> np.ndarray:
    """Save RGB reference at canonical size without any extra warping."""
    # pil_image is already 640×480
    np_img = np.asarray(pil_image).astype(np.float32) / 255.0
    plt.imsave(fname, np_img);  return np_img

def save_overlay(img_np: np.ndarray, pix_prob: np.ndarray, labels: Sequence[str],
                 fname: Path, alpha: float = 1.0):
    H, W = img_np.shape[:2]
    cmap = plt.get_cmap("tab10", len(labels)+1)
    seg = np.zeros((H, W, 3), dtype=np.float32)
    # make the overlay a soft blend of the color by the pix prob
    for c in range(len(labels)+1):
        color = np.array(cmap(c)[:3])
        if c == len(labels):
            color = np.array([0, 0, 0]) # background black
        seg[..., :3] += alpha * color * pix_prob[..., c:c+1]
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
    pil_img = download_image().resize((CANONICAL_SIZE[1], CANONICAL_SIZE[0]))
    # 2. prompt-ensemble text embeddings -------------------------------------
    text_emb = build_text_embeddings(model, tokenizer, CLASS_NAMES + BACKGROUND_CLASS_NAMES)           # [C,D]
    # 3. sliding-window feature aggregation ----------------------------------
    feat_map = aggregate_features(model, preprocess, pil_img)                 # [D,H,W]
    # 4. k-means on per-pixel features ---------------------------------------
    pix_labels, centroids = run_kmeans_on_pixels(feat_map)                    # [H,W], [k,D]
    # 5. zero-shot classify centroids, propagate to pixels -------------------
    centroid_prob = centroid_zero_shot(centroids, text_emb, CLASS_NAMES)                    # [k, C+1]
    pix_prob    = centroid_prob[pix_labels]                                   # [H, W, C+1]

    # 6. visuals -------------------------------------------------------------
    OUTPUT_DIR.mkdir(exist_ok=True)
    ref_np = save_reference((pil_img), OUTPUT_DIR/"dino_txt_img.png")
    save_overlay(ref_np, pix_prob, CLASS_NAMES, OUTPUT_DIR/"dino_txt_")

    hard_seg = np.argmax(pix_prob, axis=2)
    hard_seg_onehot = np.zeros_like(pix_prob)
    hard_seg_onehot[np.arange(hard_seg.shape[0])[:, None], np.arange(hard_seg.shape[1]), hard_seg] = 1
    save_overlay(ref_np, hard_seg_onehot, CLASS_NAMES, OUTPUT_DIR/"dino_txt_hard_seg_")

CLASS_NAMES = {0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person", 16: "potted plant", 17: "sheep", 18: "sofa", 19: "train", 20: "tv/monitor", 255: "ignore"}
if __name__ == "__main__":
    # main()
    tags = np.load('/u501/j234li/wsss/pseudolabels_full_res/class_indices_2400.npy')
    pseudolabels = np.load('/u501/j234li/wsss/pseudolabels_full_res/pseudolabels_2400.npy')
    class_names = [CLASS_NAMES[i] for i in tags]
    print("Class names: ", class_names)
    pseudolabels = torch.from_numpy(pseudolabels).float()
    pseudolabels = F.interpolate(pseudolabels.permute(2, 0, 1).unsqueeze(0), size=(CANONICAL_SIZE[0], CANONICAL_SIZE[1]), mode="bilinear", align_corners=False)[0]
    pseudolabels /= 0.01
    print(pseudolabels.shape)
    pseudolabels = pseudolabels.softmax(dim=0)  # Apply softmax across channel dimension
    # reshape pseudoalabel to canonical size
    pseudolabels = pseudolabels.permute(1, 2, 0)  # Convert back to [H, W, C]
    pseudolabels = pseudolabels.cpu().numpy()
    pil_img = download_image().resize((CANONICAL_SIZE[1], CANONICAL_SIZE[0]))
    ref_np = save_reference((pil_img), OUTPUT_DIR/"dino_txt_img.png")
    save_overlay(ref_np, pseudolabels, class_names, OUTPUT_DIR/"dino_txt_pseudolabels_")

    # draw histogram of pseudolabel probabilities by channel
    for i in range(pseudolabels.shape[2]):
        if i == len(class_names):
            class_name = "background"
        else:
            class_name = class_names[i]
        plt.figure(figsize=(6,4));  plt.hist(pseudolabels[:,:,i].flatten(), bins=100); plt.title(f"{class_name} probabilities")
        plt.tight_layout(pad=0); plt.savefig(OUTPUT_DIR/f"dino_txt_pseudolabels_{class_name}_probabilities.png", dpi=300); plt.close()

    hard_pseudolabels = np.argmax(pseudolabels, axis=2)
    # Convert hard labels to one-hot for visualization
    hard_pseudolabels_onehot = np.zeros_like(pseudolabels)
    hard_pseudolabels_onehot[np.arange(hard_pseudolabels.shape[0])[:, None], 
                           np.arange(hard_pseudolabels.shape[1]), 
                           hard_pseudolabels] = 1
    save_overlay(ref_np, hard_pseudolabels_onehot, class_names, OUTPUT_DIR/"dino_txt_hard_pseudolabels_")

    # Threshold-based visualization (0.6 threshold)
    threshold = 0.8
    max_probs = np.max(pseudolabels, axis=2)  # Get max probability for each pixel
    thresholded_labels = np.argmax(pseudolabels, axis=2)  # Start with argmax
    thresholded_labels[max_probs < threshold] = len(class_names)  # Set low-confidence to background
    
    # Convert thresholded labels to one-hot for visualization
    thresholded_onehot = np.zeros_like(pseudolabels)
    thresholded_onehot[np.arange(thresholded_labels.shape[0])[:, None], 
                      np.arange(thresholded_labels.shape[1]), 
                      thresholded_labels] = 1
    save_overlay(ref_np, thresholded_onehot, class_names, OUTPUT_DIR/"dino_txt_thresholded_pseudolabels_")
    