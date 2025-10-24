# Please see https://github.com/facebookresearch/dinov2/issues/530, credits go to @jbdel

from __future__ import annotations
import contextlib, math, random, sys
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
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

OUTPUT_DIR    = Path("pseudolabels_full_res")
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

# ---------------------------- feature encoder ------------------------------ #
def encode_patches(model, img_tensor: torch.Tensor) -> torch.Tensor:
    # returns [1, P, D] where P is the number of patch tokens (1/16th resolution)
    ctx = torch.autocast("cuda", dtype=torch.float) if DEVICE.type=="cuda" else contextlib.nullcontext()
    with torch.no_grad(), ctx:
        image_class_tokens, image_patch_tokens, backbone_patch_tokens = model.encode_image_with_patch_tokens(img_tensor)
    return image_patch_tokens

# ----------------------- compute cosine similarity on patch tokens --------- #
@torch.no_grad()
def compute_patch_cosine_similarity(
    model, preprocess, pil_image: Image.Image, text_emb: torch.Tensor, 
    num_foreground_classes: int
) -> torch.Tensor:                                # → [C+1, H_patch, W_patch]
    # Process the full image once
    img_tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]
    patch_tokens = encode_patches(model, img_tensor)[0]          # [P, D]
    
    # Calculate patch grid dimensions (1/16th of original resolution)
    p = int(math.sqrt(patch_tokens.size(0)))                    # √P
    assert p * p == patch_tokens.size(0), "non-square patch grid"
    
    # Compute cosine similarity between patch tokens and text embeddings
    # patch_tokens: [P, D], text_emb: [C, D]
    # Normalize patch tokens
    patch_tokens_norm = F.normalize(patch_tokens, p=2, dim=1)   # [P, D]
    
    # Compute cosine similarity: [P, D] @ [D, C] = [P, C]
    cosine_sim = torch.mm(patch_tokens_norm, text_emb.t())      # [P, C]
    
    # Reshape to spatial grid: [C, H_patch, W_patch]
    logits = cosine_sim.t().view(text_emb.size(0), p, p)        # [C, p, p]
    
    # Condense background classes by taking max cosine similarity among them
    # Split foreground and background classes
    foreground_logits = logits[:num_foreground_classes]         # [num_fg, p, p]
    background_logits = logits[num_foreground_classes:]         # [num_bg, p, p]
    
    # Take maximum across background classes for each patch token
    condensed_background = torch.max(background_logits, dim=0)[0]  # [p, p]
    
    # Combine foreground classes with condensed background
    condensed_logits = torch.cat([
        foreground_logits, 
        condensed_background.unsqueeze(0)  # Add channel dimension: [1, p, p]
    ], dim=0)  # [num_fg + 1, p, p]
    
    return condensed_logits


def generate_pseudolabels(dataset, start_index=0):
    random.seed(0); torch.manual_seed(0)
    model, tokenizer = prepare_model()
    preprocess = make_classification_eval_transform()

    print(f"Generating pseudo-labels from index {start_index} to {len(dataset)}")
    for i in tqdm(range(start_index, len(dataset)), desc="Generating pseudo-labels"):
        img, target = dataset[i]
        # 1. load & canonically resize once for processing ----------------
        pil_img = img.resize((CANONICAL_SIZE[1], CANONICAL_SIZE[0]))
        class_indices = np.unique(np.array(target))
        class_indices = class_indices[(class_indices != 255) & (class_indices != 0)] # remove ignore index and class 0 (background)
        class_names = [CLASS_NAMES[i] for i in class_indices]
        # 2. prompt-ensemble text embeddings -------------------------------------
        text_emb = build_text_embeddings(model, tokenizer, class_names + BACKGROUND_CLASS_NAMES)
        # 3. compute cosine similarity on patch tokens ---------------------------
        num_foreground_classes = len(class_names)
        patch_logits = compute_patch_cosine_similarity(model, preprocess, pil_img, text_emb, num_foreground_classes)
        # 4. convert to numpy and save -------------------------------------------
        pix_prob = patch_logits.permute(1, 2, 0).cpu().numpy()
        
        OUTPUT_DIR.mkdir(exist_ok=True)
        np.save(OUTPUT_DIR/f'pseudolabels_{i}.npy', pix_prob)
        np.save(OUTPUT_DIR/f'class_indices_{i}.npy', class_indices)