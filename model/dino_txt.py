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
import os
from tqdm import tqdm

os.environ["OPENBLAS_NUM_THREADS"] = "64"
DINOV3_LOCATION = '/u501/j234li/wsss/model/dinov3'
import sys
sys.path.append(DINOV3_LOCATION)
from dinov3.data.transforms import make_classification_eval_transform, make_eval_transform

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
PROMPT_TEMPLATES: Tuple[str, ...] = (
    "a photo of {}", "an image of {}", "a photograph of {}", "a picture of {}",
    "a photo of a {}", "an image of a {}", "a photo of the {}", "an image of the {}",
    "a close-up photo of {}", "a cropped image featuring {}",
)

CANONICAL_SIZE = (224, 224)
CROP_AREAS      = [0.01, 0.1, 0.3, 1]

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    """Dense sliding-window squares"""
    crops = []
    for area in CROP_AREAS:
        side  = int(round(math.sqrt(area * h * w)))
        stride = side // 2
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


# ----------------------- direct cosine similarity on patches ---------------- #
@torch.no_grad()
def compute_patch_cosine_similarity(
    model, preprocess, pil_image: Image.Image, text_emb: torch.Tensor, 
    num_foreground_classes: int
) -> torch.Tensor:                                # → [C+1, H, W]
    """
    Compute cosine similarity directly on patch tokens, then upsample.
    No k-means, just direct similarity computation.
    """
    H, W = CANONICAL_SIZE
    C = text_emb.size(0)  # Total classes (foreground + background)
    num_fg = num_foreground_classes
    
    # Initialize similarity sum and hit counter
    sim_sum = torch.zeros((C, H, W), device=DEVICE)  # [C, H, W]
    hit_cnt = torch.zeros((H, W), device=DEVICE)
    
    for (x0, y0, x1, y1) in generate_crops(H, W):
        crop = pil_image.crop((x0, y0, x1, y1))
        crop_tensor = preprocess(crop).unsqueeze(0).to(DEVICE)      # [1, 3, *, *]
        patch_tokens = encode_patches(model, crop_tensor)[0]        # [P, D]
        
        # Normalize patch tokens for cosine similarity (text_emb is already normalized)
        patch_tokens_norm = F.normalize(patch_tokens, p=2, dim=1)   # [P, D]
        
        # Compute cosine similarity: [P, D] @ [D, C] = [P, C]
        # text_emb is already normalized from build_text_embeddings
        cosine_sim = torch.mm(patch_tokens_norm, text_emb.t())  # [P, C]
        
        # Reshape to spatial grid
        p = int(math.sqrt(patch_tokens.size(0)))                    # √P
        assert p * p == patch_tokens.size(0), "non-square patch grid"
        
        # Reshape to [C, p, p]
        sim_grid = cosine_sim.t().view(C, p, p)                     # [C, p, p]
        
        # Upsample similarity grid to crop size
        sim_grid_upsampled = F.interpolate(
            sim_grid.unsqueeze(0), 
            size=(y1 - y0, x1 - x0),
            mode="bilinear", 
            align_corners=False
        )[0]  # [C, h, w]
        
        # Accumulate into full-size map
        sim_sum[:, y0:y1, x0:x1] += sim_grid_upsampled
        hit_cnt[y0:y1, x0:x1] += 1
    
    # Average across all crops (avoid div-by-0)
    hit_cnt = torch.clamp(hit_cnt, min=1)
    sim_avg = sim_sum / hit_cnt.unsqueeze(0)  # [C, H, W]
    
    # Condense background classes: take max over background classes
    fg_sim = sim_avg[:num_fg]  # [num_fg, H, W]
    bg_sim_all = sim_avg[num_fg:]  # [num_bg, H, W]
    bg_sim = torch.max(bg_sim_all, dim=0, keepdim=True).values  # [1, H, W]
    
    # Combine: [num_fg + 1, H, W]
    condensed_sim = torch.cat([fg_sim, bg_sim], dim=0)
    
    # Permute to [H, W, C] for consistency with original format
    return condensed_sim.permute(1, 2, 0)  # [H, W, num_fg + 1]

def _project_rows_to_simplex(X: np.ndarray) -> np.ndarray:
    """Project each row of X onto the probability simplex."""
    Xp = np.copy(X)
    N, C = Xp.shape
    for i in range(N):
        v = Xp[i]
        if np.all(v >= 0) and np.isclose(v.sum(), 1.0, atol=1e-6):
            continue
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * (np.arange(1, C + 1)) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
        Xp[i] = np.maximum(v - theta, 0.0)
    return Xp


def _min_max_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Min-max normalize each column to [0, 1]."""
    Xc = np.copy(X)
    low = np.min(Xc, axis=0)
    high = np.max(Xc, axis=0)
    span = np.maximum(high - low, eps)
    return (Xc - low[np.newaxis, :]) / span[np.newaxis, :]


def process_pseudolabels(pix_prob: np.ndarray, temperature: float = 0.05, num_iters: int = 5) -> np.ndarray:
    """Apply temperature-scaled softmax followed by iterative min-max + simplex projection."""
    probs = pix_prob.astype(np.float32)
    scaled = probs / temperature
    scaled -= np.max(scaled, axis=2, keepdims=True)
    exp_scaled = np.exp(scaled)
    probs = exp_scaled / np.sum(exp_scaled, axis=2, keepdims=True)

    h, w, c = probs.shape
    probs_norm = probs
    for _ in range(num_iters):
        flat = probs_norm.reshape(-1, c)
        flat = _min_max_normalize(flat)
        flat = _project_rows_to_simplex(flat)
        probs_norm = flat.reshape(h, w, c)

    return probs_norm

def generate_pseudolabels(dataset, dataset_name, start_index=0, end_index=None):
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    if dataset_name == 'voc':
        # CLASS_NAMES = {0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person", 16: "potted plant", 17: "sheep", 18: "sofa", 19: "train", 20: "tv/monitor", 255: "ignore"}
        CLASS_NAMES = {0: "background", 1: "airplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow", 11: "table", 12: "dog", 13: "horse", 14: "motorcycle", 15: "people", 16: "potted plant", 17: "sheep", 18: "sofa", 19: "train", 20: "television receiver", 255: "ignore"}
        BACKGROUND_CLASS_NAMES: Sequence[str] = [
            "ground", "land", "grass", "tree", "building", "wall", "sky", "lake", "water", "river", "sea",
            "railway", "railroad", "keyboard", "helmet", "cloud", "house", "mountain", "ocean", "road",
            "rock", "street", "valley", "bridge", "sign"
        ]
        OUTPUT_DIR = Path("pseudolabels_3")
    elif dataset_name == 'coco':
        CLASS_NAMES = {0: "background", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant", 12: "stop sign", 13: "parking meter", 14: "bench", 15: "bird", 16: "cat", 17: "dog", 18: "horse", 19: "sheep", 20: "cow", 
        21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe", 25: "backpack", 26: "umbrella", 27: "handbag", 28: "tie", 29: "suitcase", 30: "frisbee", 31: "skis", 32: "snowboard", 33: "sports ball", 34: "kite", 35: "baseball bat", 36: "baseball glove", 37: "skateboard", 38: "surfboard", 39: "tennis racket", 40: "bottle", 
        41: "wine glass", 42: "cup", 43: "fork", 44: "knife", 45: "spoon", 46: "bowl", 47: "banana", 48: "apple", 49: "sandwich", 50: "orange", 51: "broccoli", 52: "carrot", 53: "hot dog", 54: "pizza", 55: "donut", 56: "cake", 57: "chair", 58: "couch", 59: "potted plant", 60: "bed", 61: "dining table", 62: "toilet", 63: "tv", 
        64: "laptop", 65: "mouse", 66: "remote", 67: "keyboard", 68: "cell phone", 69: "microwave", 70: "oven", 71: "toaster", 72: "sink", 73: "refrigerator", 74: "book", 75: "clock", 76: "vase", 77: "scissors", 78: "teddy bear", 79: "hair drier", 80: "toothbrush", 255: "ignore"}
        BACKGROUND_CLASS_NAMES: Sequence[str] = [
            "ground", "land", "grass", "tree", "building", "wall", "sky", "lake", "water", "river", "sea",
            "railway", "railroad", "helmet", "cloud", "house", "mountain", "ocean", "road",
            "rock", "street", "valley", "bridge"
        ]
        OUTPUT_DIR = Path("pseudolabels_coco")

    model, tokenizer = prepare_model()
    # Use eval transform without center crop to preserve full spatial coverage for segmentation
    preprocess = make_eval_transform(crop_size=None, resize_square=True, resize_size=224)

    total_len = len(dataset)
    stop = total_len if end_index is None else min(end_index, total_len)
    if start_index >= stop:
        print(f"No samples to process: start_index ({start_index}) >= stop ({stop}).")
        return

    print(f"Generating pseudo-labels from index {start_index} to {stop}")
    for i in tqdm(range(start_index, stop), desc="Generating pseudo-labels"):
        img, target = dataset[i]
        # 1. load & canonically resize once for cropping geometry ----------------
        pil_img = img.resize((CANONICAL_SIZE[1], CANONICAL_SIZE[0]))
        class_indices = np.unique(np.array(target))
        class_indices = class_indices[(class_indices != 255) & (class_indices != 0)] # remove ignore index and class 0 (background)
        class_names = [CLASS_NAMES[i] for i in class_indices]
        # 2. prompt-ensemble text embeddings -------------------------------------
        text_emb = build_text_embeddings(model, tokenizer, class_names + BACKGROUND_CLASS_NAMES)
        # 3. compute cosine similarity directly on patches and upsample ----------
        num_foreground_classes = len(class_names)
        pix_prob = compute_patch_cosine_similarity(
            model, preprocess, pil_img, text_emb, num_foreground_classes
        )  # [H, W, num_fg + 1]
        # Convert to numpy and process
        pix_prob = pix_prob.cpu().numpy()
        # pix_prob = process_pseudolabels(pix_prob)
        
        OUTPUT_DIR.mkdir(exist_ok=True)
        np.save(OUTPUT_DIR/f'pseudolabels_{i}.npy', pix_prob)
        np.save(OUTPUT_DIR/f'class_indices_{i}.npy', class_indices)