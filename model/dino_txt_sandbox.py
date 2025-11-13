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
    "diningtable", "potted plant", "bottle", "person"
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

CANONICAL_SIZE = (224, 224)                 # (H, W)
CROP_AREAS      = [0.01, 0.1, 0.3, 1]        # 1 %, 10 %, 100 % of image area
CROP_JITTER     = 0.10                      # 10 % coordinate noise
NUM_CLUSTERS    = 128
MAX_KMEANS_SAMPLES = 10_000                 # subsample for speed

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
OUTPUT_DIR    = Path("tmp")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DINO_EMBED_DIM = 1024

# --------------------------------------------------------------------------- #
# Helper utilities                                                             #
# --------------------------------------------------------------------------- #
def download_image() -> Image.Image:
    return Image.open('/u501/j234li/wsss/VOCdevkit/VOC2012/JPEGImages/2011_001928.jpg').convert("RGB") # 2011_003276.jpg, 2010_005022.jpg

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
        print(side, stride)
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
    cmap = plt.get_cmap("gist_ncar", len(labels)+1)
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
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
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

def plot_probability_histograms(pseudolabels, class_names, output_path):
    """
    Plot histograms of probabilities for all classes in a single figure.
    
    Args:
        pseudolabels: numpy array of shape [H, W, C] with probability values
        class_names: list of class names (excluding background)
        output_path: Path object or string for output file
    """
    num_channels = pseudolabels.shape[2]
    ncols = 3
    nrows = (num_channels + ncols - 1) // ncols  # Ceiling division
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows))
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes
    
    for i in range(num_channels):
        if i == len(class_names):
            class_name = "background"
        else:
            class_name = class_names[i]
        axes[i].hist(pseudolabels[:,:,i].flatten(), bins=100)
        axes[i].set_title(f"{class_name} probabilities")
        axes[i].set_xlabel("Probability")
        axes[i].set_ylabel("Frequency")
    
    # Hide unused subplots
    for i in range(num_channels, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

CLASS_NAMES = {0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person", 16: "potted plant", 17: "sheep", 18: "sofa", 19: "train", 20: "tv/monitor", 255: "ignore"}
# CLASS_NAMES = {0: "background", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant", 12: "stop sign", 13: "parking meter", 14: "bench", 15: "bird", 16: "cat", 17: "dog", 18: "horse", 19: "sheep", 20: "cow", 
# 21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe", 25: "backpack", 26: "umbrella", 27: "handbag", 28: "tie", 29: "suitcase", 30: "frisbee", 31: "skis", 32: "snowboard", 33: "sports ball", 34: "kite", 35: "baseball bat", 36: "baseball glove", 37: "skateboard", 38: "surfboard", 39: "tennis racket", 40: "bottle", 
# 41: "wine glass", 42: "cup", 43: "fork", 44: "knife", 45: "spoon", 46: "bowl", 47: "banana", 48: "apple", 49: "sandwich", 50: "orange", 51: "broccoli", 52: "carrot", 53: "hot dog", 54: "pizza", 55: "donut", 56: "cake", 57: "chair", 58: "couch", 59: "potted plant", 60: "bed", 61: "dining table", 62: "toilet", 63: "tv", 
# 64: "laptop", 65: "mouse", 66: "remote", 67: "keyboard", 68: "cell phone", 69: "microwave", 70: "oven", 71: "toaster", 72: "sink", 73: "refrigerator", 74: "book", 75: "clock", 76: "vase", 77: "scissors", 78: "teddy bear", 79: "hair drier", 80: "toothbrush", 255: "ignore"}
if __name__ == "__main__":
    # main()
    tags = np.load('/u501/j234li/wsss/pseudolabels_full_img/class_indices_500.npy') # 700
    pseudolabels = np.load('/u501/j234li/wsss/pseudolabels_full_img/pseudolabels_500.npy') # 700
    class_names = [CLASS_NAMES[i] for i in tags]
    print("Class names: ", class_names)
    pseudolabels = torch.from_numpy(pseudolabels).float()
    print(pseudolabels.shape)
    pseudolabels = F.interpolate(pseudolabels.permute(2, 0, 1).unsqueeze(0), size=(CANONICAL_SIZE[0], CANONICAL_SIZE[1]), mode="nearest")[0]
    pseudolabels = pseudolabels / 0.03
    pseudolabels = pseudolabels.softmax(dim=0)  # Apply softmax across channel dimension
    # reshape pseudoalabel to canonical size
    pseudolabels = pseudolabels.permute(1, 2, 0)  # Convert back to [H, W, C]
    pseudolabels = pseudolabels.cpu().numpy()
    pil_img = download_image().resize((CANONICAL_SIZE[1], CANONICAL_SIZE[0]))
    OUTPUT_DIR.mkdir(exist_ok=True)
    ref_np = save_reference((pil_img), OUTPUT_DIR/"dino_txt_img.png")
    save_overlay(ref_np, pseudolabels, class_names, OUTPUT_DIR/"dino_txt_soft_pseudolabels_")
    plot_probability_histograms(pseudolabels, class_names, OUTPUT_DIR/f"dino_txt_pseudolabels_raw_histograms.png")

    # def harden_top_k(prob_tensor, top_percent=1):
    #     soft = prob_tensor  # original soft probabilities
    #     hardened = soft.copy()
    #     h, w, c = hardened.shape

    #     for class_idx in range(c-1):
    #         channel = soft[:, :, class_idx]
    #         threshold = np.percentile(channel, 100 - top_percent)

    #         mask = channel > threshold
    #         hardened[:, :, class_idx][mask] = 1.0

    #         # If you still need exclusivity, apply it using the mask computed from `soft`
    #         for other_idx in range(c):
    #             if other_idx != class_idx:
    #                 hardened[:, :, other_idx][mask] = 0.0

    #     return hardened
    
    # hard_pseudolabels = harden_top_k(pseudolabels, top_percent=0.5)
    # plot_probability_histograms(hard_pseudolabels, class_names, OUTPUT_DIR/f"dino_txt_pseudolabels_hardened_histograms.png")
    # save_overlay(ref_np, hard_pseudolabels, class_names, OUTPUT_DIR/"dino_txt_hard_pseudolabels_")

    # # Normalize each channel
    # pseudolabels_normalized = pseudolabels.copy()
    # for j in range(1):
    #     # for i in range(pseudolabels_normalized.shape[2]):
    #     #     channel = pseudolabels_normalized[:, :, i]
    #     #     min_val = channel.min()
    #     #     max_val = channel.max()
    #     #     pseudolabels_normalized[:, :, i] = (channel - min_val) / (max_val - min_val)
    #     # # # Softmax along axis 2 using numpy
    #     # # exp_x = np.exp(pseudolabels_normalized / 0.1)
    #     # # pseudolabels_normalized = exp_x / np.sum(exp_x, axis=2, keepdims=True)
    #     # pseudolabels_normalized = pseudolabels_normalized / np.sum(pseudolabels_normalized, axis=2, keepdims=True)
        
    #     def project_rows_to_simplex(X):
    #         # Projects each row of X to the probability simplex (L2 projection).
    #         # Implements the algorithm from:
    #         #   Wang & Carreira-Perpinan / "Projection onto the probability simplex"
    #         Xp = np.copy(X)
    #         # shape (N, C)
    #         N, C = Xp.shape
    #         for i in range(N):
    #             v = Xp[i]
    #             if v.sum() == 1 and np.all(v >= 0):
    #                 continue
    #             # sort descending
    #             u = np.sort(v)[::-1]
    #             cssv = np.cumsum(u)
    #             rho = np.nonzero(u * np.arange(1, C+1) > (cssv - 1))[0][-1]
    #             theta = (cssv[rho] - 1) / (rho + 1.0)
    #             w = np.maximum(v - theta, 0.0)
    #             Xp[i] = w
    #         return Xp
        
    #     def naive_normalize(X, eps=1e-12):
    #         # renormalize to probability simplex
    #         Xc = X.copy()
    #         Xc = Xc / np.sum(Xc, axis=1, keepdims=True)
    #         return Xc
        
    #     def softmax_normalize(X, eps=1e-12, temperature=0.1):
    #         # softmax normalize each column to [0, 1]
    #         Xc = X.copy()
    #         Xc = np.exp(Xc / temperature) / np.sum(np.exp(Xc / temperature), axis=1, keepdims=True)
    #         return Xc

    #     def min_max_normalize(X, eps=1e-12):
    #         # min-max normalize each column to [0, 1]
    #         Xc = X.copy()
    #         low = np.min(Xc, axis=0)
    #         high = np.max(Xc, axis=0)
    #         span = np.maximum(high - low, eps)
    #         Xc = (Xc - low[np.newaxis, :]) / span[np.newaxis, :]
    #         return Xc
        
    #     def percentile_normalize(X, lower_pct=5, upper_pct=95, eps=1e-12):
    #         """
    #         Normalize each column of X using the values at the given lower/upper percentiles.
    #         Values below the lower percentile become 0, above the upper percentile become 1.
    #         """
    #         Xp = X.copy()
    #         low = np.percentile(Xp, lower_pct, axis=0)
    #         high = np.percentile(Xp, upper_pct, axis=0)

    #         span = np.maximum(high - low, eps)
    #         Xp = (Xp - low[np.newaxis, :]) / span[np.newaxis, :]
    #         return np.clip(Xp, 0.0, 1.0)

    #     pseudolabels_normalized = pseudolabels_normalized.reshape(-1, pseudolabels_normalized.shape[2])
    #     pseudolabels_normalized = percentile_normalize(pseudolabels_normalized)
    #     pseudolabels_normalized = naive_normalize(pseudolabels_normalized)
    #     pseudolabels_normalized = pseudolabels_normalized.reshape(CANONICAL_SIZE[0], CANONICAL_SIZE[1], -1)
    #     # # Visualize normalized pseudolabels
    #     if j % 1 == 0:
    #         save_overlay(ref_np, pseudolabels_normalized, class_names, OUTPUT_DIR/f"dino_txt_normalized_pseudolabels_{j}.png")
    #         plot_probability_histograms(pseudolabels_normalized, class_names, OUTPUT_DIR/f"dino_txt_pseudolabels_normalized_histograms_{j}.png")

    # hard_pseudolabels = np.argmax(pseudolabels_normalized, axis=2)
    # # Convert hard labels to one-hot for visualization
    # hard_pseudolabels_onehot = np.zeros_like(pseudolabels_normalized)
    # hard_pseudolabels_onehot[np.arange(hard_pseudolabels.shape[0])[:, None], 
    #                        np.arange(hard_pseudolabels.shape[1]), 
    #                        hard_pseudolabels] = 1
    # save_overlay(ref_np, hard_pseudolabels_onehot, class_names, OUTPUT_DIR/"dino_txt_hard_pseudolabels_")

    # # Threshold-based visualization (0.6 threshold)
    # threshold = 0.9
    # max_probs = np.max(pseudolabels_normalized, axis=2)  # Get max probability for each pixel
    # thresholded_labels = np.argmax(pseudolabels_normalized, axis=2)  # Start with argmax
    # thresholded_labels[max_probs < threshold] = len(class_names)  # Set low-confidence to background
    
    # # Convert thresholded labels to one-hot for visualization
    # thresholded_onehot = np.zeros_like(pseudolabels_normalized)
    # thresholded_onehot[np.arange(thresholded_labels.shape[0])[:, None], 
    #                   np.arange(thresholded_labels.shape[1]), 
    #                   thresholded_labels] = 1
    # save_overlay(ref_np, thresholded_onehot, class_names, OUTPUT_DIR/"dino_txt_thresholded_pseudolabels_")
    