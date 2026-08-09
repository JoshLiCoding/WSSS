import math
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet import Bottleneck

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def imagenet_to_clip(x: torch.Tensor) -> torch.Tensor:
    """Images arrive ImageNet-normalized (utils/dataset.py); CLIP expects its own stats."""
    x = x * IMAGENET_STD.to(x) + IMAGENET_MEAN.to(x)
    return (x - CLIP_MEAN.to(x)) / CLIP_STD.to(x)


def interpolate_pos_embed(pos_embed: torch.Tensor, grid_size: tuple) -> torch.Tensor:
    """Resize a [1+N, D] CLIP positional embedding to an arbitrary [1+gh*gw, D] grid."""
    cls_pos, patch_pos = pos_embed[:1], pos_embed[1:]
    n, d = patch_pos.shape
    side = int(math.sqrt(n))
    gh, gw = grid_size
    if (gh, gw) == (side, side):
        return pos_embed
    patch_pos = patch_pos.reshape(1, side, side, d).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(gh, gw), mode="bilinear", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(gh * gw, d)
    return torch.cat([cls_pos, patch_pos], dim=0)


def embed_patches(visual, x: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
    """conv1 -> +cls token -> +precomputed pos embed -> ln_pre -> LND, ready for transformer.resblocks."""
    x = visual.conv1(x.type(visual.conv1.weight.dtype))  # [B, D, gh, gw]
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, P, D]
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)
    x = visual.ln_pre(x + pos_embed.to(x.dtype))
    return x.permute(1, 0, 2)  # LND


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class ClipWSSS(nn.Module):
    """Frozen CLIP ViT backbone + single-stage trainable decoder (transformer blocks -> 4x upsample -> ResNet bottlenecks)."""

    def __init__(
        self,
        clip_name: str = "ViT-B/16",
        num_transformer_blocks: int = 2,
        num_conv_blocks: int = 2,
        out_channels: int = 21,
        img_size: int = 448,
        device: str = "cuda",
        feature_layers: tuple = None,
    ):
        super().__init__()
        clip_model, _ = clip.load(clip_name, device=device, jit=False)
        self.clip = clip_model.float().eval()
        for p in self.clip.parameters():
            p.requires_grad_(False)

        visual = self.clip.visual
        self.patch_size = visual.conv1.kernel_size[0]
        width = visual.transformer.width
        heads = visual.transformer.resblocks[0].attn.num_heads
        self.width = width

        depth = len(visual.transformer.resblocks)
        self.feature_layers = feature_layers or tuple(sorted(set(
            max(1, round(depth * frac)) for frac in (0.25, 0.5, 0.75, 1.0)
        )))
        self.fuse = nn.Linear(width * len(self.feature_layers), width)

        self.gh = self.gw = img_size // self.patch_size
        pos_embed = interpolate_pos_embed(visual.positional_embedding, (self.gh, self.gw))
        self.register_buffer("pos_embed", pos_embed, persistent=False)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(width, heads) for _ in range(num_transformer_blocks)]
        )
        self.ln = nn.LayerNorm(width)

        planes = width // Bottleneck.expansion
        self.conv_blocks = nn.ModuleList(
            [Bottleneck(width, planes, norm_layer=nn.BatchNorm2d) for _ in range(num_conv_blocks)]
        )
        self.classifier = nn.Conv2d(width, out_channels, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for block in self.transformer_blocks:
            for m in block.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        for block in self.conv_blocks:
            for m in block.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.classifier.bias, 0)
        nn.init.trunc_normal_(self.fuse.weight, std=0.02)
        nn.init.zeros_(self.fuse.bias)

    @torch.no_grad()
    def encode_patch_grid(self, x: torch.Tensor) -> list:
        """Full frozen CLIP visual forward; returns patch tokens [B, P, D] at each of self.feature_layers."""
        visual = self.clip.visual
        tokens = embed_patches(visual, imagenet_to_clip(x), self.pos_embed)
        features = []
        for i, block in enumerate(visual.transformer.resblocks, start=1):
            tokens = block(tokens)
            if i in self.feature_layers:
                features.append(tokens.permute(1, 0, 2)[:, 1:])  # NLD, patches only
        return features

    def forward(self, x: torch.Tensor) -> dict:
        multi_level_tokens = self.encode_patch_grid(x)
        patch_tokens = self.fuse(torch.cat(multi_level_tokens, dim=-1))
        gh, gw = self.gh, self.gw
        for block in self.transformer_blocks:
            patch_tokens = block(patch_tokens)
        patch_tokens = self.ln(patch_tokens)

        B, P, D = patch_tokens.shape
        feat = patch_tokens.permute(0, 2, 1).reshape(B, D, gh, gw)
        feat = F.interpolate(feat, scale_factor=4, mode="bilinear", align_corners=False)
        for block in self.conv_blocks:
            feat = block(feat)

        return {"seg": self.classifier(feat)}
