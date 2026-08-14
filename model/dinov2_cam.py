"""DINOv2 ViT-B/14 pseudo-label generation.

DINOv2 has no text encoder, so class identity comes from the image-level tags: a per-patch
classification head over the (by default frozen) backbone's patch tokens, trained with a
multiple-instance loss. Its score map on the patch grid is the CAM.

No autograd Grad-CAM: for a linear head pooled into an image-level score it collapses onto
that same class-score map, since the pooling gradient is one constant for every patch. The
class competition "softmax Grad-CAM" is after runs downstream instead, in
utils.pseudolabels.soft_pseudolabels.

Cached per image as `{name}.npz`, the previous contract: `sim` [K, p, p] (the classes
present, ascending, then background) and `classes` [K-1].
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

from utils.dataset import MEAN, STD

DINOV2_HUB_REPO = 'facebookresearch/dinov2'


def load_backbone(cfg, device):
    """DINOv2 ViT from torch.hub. `pseudolabel.dinov2_location`, when set, is a local clone of
    facebookresearch/dinov2, so nothing is fetched at run time."""
    location = cfg.pseudolabel.get('dinov2_location', None)
    repo_or_dir = location if location else DINOV2_HUB_REPO
    backbone = torch.hub.load(
        repo_or_dir, cfg.pseudolabel.backbone,
        source='local' if location else 'github',
    )
    return backbone.to(device)


def cls_attention_hook(backbone):
    """Monkeypatches the backbone's last block to also stash its CLS -> patch attention, since
    DINOv2's fused SDPA/xformers kernels never materialize attention weights. Returns the dict
    they land in under 'cls_attn': (B, heads, P) attention from CLS onto every patch token."""
    block = backbone.blocks[-1]
    attn = (block[-1] if isinstance(block, nn.ModuleList) else block).attn
    storage = {}

    def forward(x, is_causal=False):
        B, N, C = x.shape
        qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, C // attn.num_heads)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))  # (B, heads, N, d)
        weights = (q @ k.transpose(-2, -1) * attn.scale).softmax(dim=-1)
        storage['cls_attn'] = weights[:, :, 0, 1 + backbone.num_register_tokens:]
        out = (weights @ v).transpose(1, 2).reshape(B, N, C)
        return attn.proj_drop(attn.proj(out))

    attn.forward = forward
    return storage


class PatchClassifier(nn.Module):
    """DINOv2 backbone plus a 1x1 conv scoring every patch token. 'cosine' normalizes both the
    tokens and the conv weight, which drops DINOv2's wildly varying patch norm out of the map;
    'linear' is the classic CAM head, raw dot product plus a bias. The backbone is always
    frozen: fine-tuning it on image-level labels is the classic way to make CAMs *worse*."""

    def __init__(self, backbone, num_fg, head='cosine', logit_scale=10.0):
        super().__init__()
        if head not in ('cosine', 'linear'):
            raise ValueError(f"Unknown head: {head}")
        self.backbone = backbone
        self.patch_size = backbone.patch_size
        self.num_fg = num_fg
        self.head = head

        self.classifier = nn.Conv2d(backbone.embed_dim, num_fg, 1, bias=(head == 'linear'))
        if head == 'cosine':
            nn.init.normal_(self.classifier.weight, std=0.02)
            self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale)).log())

        backbone.requires_grad_(False)
        self._attn = cls_attention_hook(backbone)

    def head_parameters(self):
        params = list(self.classifier.parameters())
        return params + [self.logit_scale] if self.head == 'cosine' else params

    def train(self, mode=True):
        super().train(mode)
        # Keep the frozen backbone in eval mode so its stochastic depth stays off.
        self.backbone.eval()
        return self

    def patch_tokens(self, images):
        """(B, D, h, w) normalized patch tokens on the ViT grid."""
        tokens = self.backbone.forward_features(images)['x_norm_patchtokens']  # (B, P, D)
        B, P, D = tokens.shape
        h, w = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size
        assert h * w == P, f"patch grid {h}x{w} does not match {P} tokens"
        return tokens.transpose(1, 2).reshape(B, D, h, w)

    def objectness_map(self):
        """(B, h, w) CLS attention from the backbone's last block, averaged over heads: a
        salient-object prior independent of any specific class. Populated as a side effect of
        the most recent patch_tokens/score_maps call, since it comes off the same forward
        pass -- call this right after one of those, not on its own."""
        attn = self._attn['cls_attn'].mean(dim=1)  # (B, P)
        p = int(round(attn.shape[1] ** 0.5))
        return attn.reshape(-1, p, p)

    def score_maps(self, images):
        """(B, num_fg, h, w) per-patch class scores; cosines in [-1, 1] for the cosine head."""
        tokens = self.patch_tokens(images)
        if self.head == 'linear':
            return self.classifier(tokens)
        return F.conv2d(F.normalize(tokens, p=2, dim=1),
                        F.normalize(self.classifier.weight, p=2, dim=1))

    def forward(self, images):
        """(score maps, image-level logits). The cosine head is bounded, so it needs the
        learnable scale to reach confident logits; the linear head scales its own weights.
        Image-level logits are the mean over patches (plain GAP), which keeps the CAM identity
        exact: the per-patch class-score map *is* the Grad-CAM map, no autograd needed."""
        scores = self.score_maps(images)
        logits = scores if self.head == 'linear' else \
            self.logit_scale.exp().clamp(max=100.0) * scores
        return scores, logits.mean(dim=(-2, -1))


class _ClassificationImages(Dataset):
    """Images with their multi-hot foreground tags, for training the classification head."""

    def __init__(self, dataset, num_fg, image_size, augment):
        self.dataset = dataset
        self.num_fg = num_fg
        ops = (
            [transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(3. / 4., 4. / 3.)),
             transforms.RandomHorizontalFlip(),
             transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)]
            if augment else
            [transforms.Resize((image_size, image_size))]
        )
        self.transform = transforms.Compose(
            ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        ids = torch.from_numpy(self.dataset.labels(idx)).long()  # 1-based foreground ids
        target = torch.zeros(self.num_fg)
        target[ids - 1] = 1.0
        return self.transform(image), target


def train_head(cfg, dataset, model, device):
    """Train the head on the image-level tags."""
    image_size = cfg.pseudolabel.image_size
    epochs = cfg.pseudolabel.head_epochs

    loader = DataLoader(
        _ClassificationImages(dataset, model.num_fg, image_size, augment=True),
        batch_size=cfg.pseudolabel.head_batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        drop_last=True,
    )

    groups = [{'params': model.head_parameters(), 'lr': cfg.pseudolabel.head_lr}]
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.pseudolabel.head_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * len(loader))
    )
    autocast = torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda')

    model.train()
    for epoch in range(epochs):
        running_loss, running_correct, seen = 0.0, 0.0, 0
        for images, targets in tqdm(loader, desc=f"{model.head} head epoch {epoch + 1}/{epochs}"):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            with autocast:
                _, logits = model(images)
                loss = F.binary_cross_entropy_with_logits(logits.float(), targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * images.shape[0]
            running_correct += ((logits > 0) == (targets > 0.5)).float().mean().item() * images.shape[0]
            seen += images.shape[0]
        print(f"  epoch {epoch + 1}: BCE {running_loss / seen:.4f}, "
              f"tag accuracy {running_correct / seen:.4f}")
    model.eval()


@torch.no_grad()
def cam_batch(model, images, scales, flip):
    """(B, num_fg, p, p) CAMs and (B, p, p) CLS-attention objectness maps, both averaged over
    scales and the horizontal flip. Each scale is rounded to a whole number of patches and
    resampled back onto the scale-1.0 grid."""
    base = images.shape[-1] // model.patch_size
    total_scores, total_obj = None, None
    for scale in scales:
        size = max(1, round(images.shape[-1] * scale / model.patch_size)) * model.patch_size
        scaled = (images if size == images.shape[-1] else
                  F.interpolate(images, size=(size, size), mode='bilinear', align_corners=False))
        views = [scaled, torch.flip(scaled, dims=[-1])] if flip else [scaled]
        for view, flipped in zip(views, (False, True)):
            scores = model.score_maps(view).float()
            obj = model.objectness_map().float().unsqueeze(1)
            if flipped:
                scores = torch.flip(scores, dims=[-1])
                obj = torch.flip(obj, dims=[-1])
            if scores.shape[-1] != base:
                scores = F.interpolate(scores, size=(base, base), mode='bilinear',
                                       align_corners=False)
                obj = F.interpolate(obj, size=(base, base), mode='bilinear', align_corners=False)
            total_scores = scores if total_scores is None else total_scores + scores
            total_obj = obj if total_obj is None else total_obj + obj
    n = len(scales) * (2 if flip else 1)
    return total_scores / n, (total_obj / n).squeeze(1)


def save_objectness_vis(image, obj, path):
    """Save `path` as the (normalized, model-input-size) image beside its CLS-attention
    objectness map as a jet heatmap, upsampled with nearest so individual patches stay
    visible."""
    img = image.clone()
    for channel, mean, std in zip(img, MEAN, STD):
        channel.mul_(std).add_(mean)
    img = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    obj_resized = F.interpolate(obj[None, None], size=img.shape[:2], mode='nearest')[0, 0].numpy()
    heat = (plt.get_cmap('jet')(obj_resized)[..., :3] * 255).astype(np.uint8)

    Image.fromarray(np.concatenate([img, heat], axis=1)).save(path)


class _EvalImages(Dataset):
    """Square-resized images plus their multi-hot tags and dataset index."""

    def __init__(self, dataset, num_classes, image_size):
        self.dataset = dataset
        self.num_classes = num_classes
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        present = torch.zeros(self.num_classes, dtype=torch.bool)
        present[torch.from_numpy(self.dataset.labels(idx)).long()] = True
        return self.transform(image), present, idx


def ensure_pseudolabels(cfg, dataset, out_dir, device):
    """Generate the DINOv2 pseudo-labels missing from out_dir, then leave them cached there."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [
        i for i, name in enumerate(dataset.names)
        if not (out_dir / f'{name}.npz').exists()
    ]
    if not todo:
        print(f"Using cached DINOv2 pseudo-labels for {len(dataset)} images in {out_dir}")
        return
    print(f"Generating DINOv2 pseudo-labels for {len(todo)}/{len(dataset)} images into {out_dir}")

    num_classes = cfg.model.num_classes
    head = cfg.pseudolabel.head
    model = PatchClassifier(
        load_backbone(cfg, device),
        num_fg=num_classes - 1,
        head=head,
        logit_scale=cfg.pseudolabel.logit_scale,
    ).to(device)

    # Trained on the whole split, not just the images still missing a cache.
    head_path = Path(cfg.pseudolabel.head_path)
    if head_path.exists():
        print(f"Loading {head} head from {head_path}")
        checkpoint = torch.load(head_path, map_location=device)
        if checkpoint['head'] != head:
            raise ValueError(f"{head_path} holds a '{checkpoint['head']}' head but "
                             f"pseudolabel.head is '{head}'")
        model.load_state_dict(checkpoint['state'], strict=False)
        model.eval()
    else:
        train_head(cfg, dataset, model, device)
        head_path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v for k, v in model.state_dict().items() if not k.startswith('backbone.')}
        torch.save({'head': head, 'state': state}, head_path)
        print(f"Saved {head} head to {head_path}")

    loader = DataLoader(
        Subset(_EvalImages(dataset, num_classes, cfg.pseudolabel.image_size), todo),
        batch_size=cfg.pseudolabel.batch_size,
        num_workers=cfg.training.num_workers,
    )
    scales = list(cfg.pseudolabel.cam_scales)
    bg_score = float(cfg.pseudolabel.bg_score)
    levels = (0.5, 0.9, 0.99, 1.0)
    quantiles = torch.tensor(levels)
    observed = []

    for images, present, indices in tqdm(loader, desc="DINOv2 pseudo-labels"):
        cams, obj = cam_batch(model, images.to(device), scales, cfg.pseudolabel.cam_flip)
        cams, obj = cams.cpu(), obj.cpu()
        obj = obj - obj.amin(dim=(-2, -1), keepdim=True)
        obj = obj / obj.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)  # per-image, to [0, 1]
        for b, index in enumerate(indices.tolist()):
            classes = present[b].nonzero().flatten()
            classes = classes[classes > 0]  # channel 0 is background, which has no prototype
            sim = cams[b, classes - 1]  # (K-1, p, p)
            # Standard CAM normalization: drop negative evidence, then scale each class by its
            # own peak, which puts bg_score on a fixed [0, 1] scale across retrainings.
            sim = sim.clamp(min=0)
            sim = sim / sim.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
            observed.append(torch.quantile(sim.flatten(), quantiles))
            # bg_score is suppressed wherever the backbone's own CLS attention says a patch
            # belongs to some salient object, so non-discriminative foreground interior isn't
            # outvoted by a flat threshold the way a uniform background level would.
            bg_map = bg_score * (1 - obj[b])
            sim = torch.cat([sim, bg_map[None]], dim=0)
            name = dataset.names[index]
            np.savez(out_dir / f'{name}.npz', sim=sim.numpy(),
                     classes=classes.numpy().astype(np.int16))
            save_objectness_vis(images[b], obj[b], out_dir / f'{name}_attn.png')

    if observed:
        means = torch.stack(observed).mean(dim=0)
        print("Normalized CAM percentiles (mean over images): "
              + ", ".join(f"p{round(q * 100)}={v:.3f}" for q, v in zip(levels, means))
              + f"  --  pseudolabel.bg_score is {bg_score}, so roughly everything above it is "
                "foreground")
