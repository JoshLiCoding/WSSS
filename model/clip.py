"""Weakly supervised segmentation model with frozen OpenAI CLIP (ViT) backbone."""

from __future__ import annotations

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from clip.model import VisionTransformer

import sys

DINOV3_LOCATION = "/u501/j234li/wsss/model/dinov3"
sys.path.append(DINOV3_LOCATION)

from dinov3.layers import SelfAttentionBlock, SwiGLUFFN
from dinov3.models.vision_transformer import init_weights_vit
from dinov3.utils import named_apply
from model.resnet import Bottleneck


def _vit_sequence(visual: VisionTransformer, x: torch.Tensor) -> torch.Tensor:
    """Run CLIP ViT through transformer blocks; returns [B, 1+P, width] (before ln_post/proj).

    Expects **spatial size matching** ``visual.input_resolution`` (e.g. 224 for ViT-L/14) so
    ``positional_embedding`` aligns with the patch grid from ``conv1`` without interpolation.
    """
    x = x.type(visual.conv1.weight.dtype)
    x = visual.conv1(x)
    b, w, _, _ = x.shape
    x = x.reshape(b, w, -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(
        b, 1, w, dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls, x], dim=1)

    pos = visual.positional_embedding.to(x.dtype)
    assert pos.shape[0] == x.shape[1], (
        f"Input H×W must match CLIP pretrained resolution ({visual.input_resolution}). "
        f"Got {x.shape[1] - 1} patch tokens but pos emb has {pos.shape[0] - 1} patch positions."
    )
    x = x + pos

    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    return x


class ClipWSSS(nn.Module):
    def __init__(
        self,
        clip_model: nn.Module,
        num_transformer_blocks: int = 1,
        num_conv_blocks: int = 2,
        out_channels: int = 21,
        transformer_drop_path: float = 0.0,
        use_bottleneck: bool = False,
        use_transpose_conv: bool = False,
    ):
        super().__init__()
        if not isinstance(clip_model.visual, VisionTransformer):
            raise TypeError("ClipWSSS expects a ViT-based CLIP model (e.g. ViT-L/14).")

        self.clip = clip_model
        self.visual = clip_model.visual
        self.num_transformer_blocks = num_transformer_blocks
        self.num_conv_blocks = num_conv_blocks
        self.use_bottleneck = use_bottleneck
        self.use_transpose_conv = use_transpose_conv

        self.backbone_dim = self.visual.transformer.width
        self.num_heads = self.backbone_dim // 64

        block_list = [
            SelfAttentionBlock(
                self.backbone_dim,
                self.num_heads,
                ffn_layer=partial(SwiGLUFFN, align_to=64),
                init_values=1e-5,
                drop_path=transformer_drop_path,
            )
            for _ in range(num_transformer_blocks)
        ]
        self.transformer_blocks = nn.ModuleList(block_list)
        self.ln = nn.LayerNorm(self.backbone_dim)

        if use_bottleneck:
            planes = self.backbone_dim // Bottleneck.expansion
            conv_list = [
                Bottleneck(
                    inplanes=self.backbone_dim,
                    planes=planes,
                    stride=1,
                    downsample=None,
                    groups=1,
                    base_width=64,
                    dilation=1,
                    norm_layer=nn.BatchNorm2d,
                )
                for _ in range(num_conv_blocks)
            ]
        else:
            conv_list = [
                nn.Sequential(
                    nn.Conv2d(self.backbone_dim, self.backbone_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(self.backbone_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.backbone_dim, self.backbone_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(self.backbone_dim),
                )
                for _ in range(num_conv_blocks)
            ]
        self.conv_blocks = nn.ModuleList(conv_list)
        self.lin_classifier = nn.Conv2d(self.backbone_dim, out_channels, 1, bias=True)

        if use_transpose_conv:
            self.upsample_conv1 = nn.ConvTranspose2d(
                self.backbone_dim,
                self.backbone_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                groups=self.backbone_dim,
                bias=False,
            )
            self.upsample_conv2 = nn.ConvTranspose2d(
                self.backbone_dim,
                self.backbone_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                groups=self.backbone_dim,
                bias=False,
            )
            self._init_bilinear_transpose_conv(self.upsample_conv1)
            self._init_bilinear_transpose_conv(self.upsample_conv2)

        self.init_weights()

    def _init_bilinear_transpose_conv(self, conv_transpose: nn.ConvTranspose2d) -> None:
        filt = np.array(
            [[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]], dtype=np.float32
        )
        with torch.no_grad():
            weight = torch.zeros(conv_transpose.weight.shape, dtype=torch.float32)
            for i in range(conv_transpose.out_channels):
                weight[i, 0, :, :] = torch.from_numpy(filt)
            conv_transpose.weight.copy_(weight)

    def init_weights(self) -> None:
        if self.num_transformer_blocks > 0:
            for block in self.transformer_blocks:
                named_apply(init_weights_vit, block)
            self.ln.reset_parameters()
        if self.num_conv_blocks > 0:
            for block in self.conv_blocks:
                for m in block.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.lin_classifier.weight, mean=0.0, std=0.3)
        nn.init.constant_(self.lin_classifier.bias, 0)

    def forward(self, x: torch.Tensor) -> dict:
        with torch.no_grad():
            seq = _vit_sequence(self.visual, x)
            patch_joint = self.visual.ln_post(seq[:, 1:, :]) @ self.visual.proj

        tokens = seq
        for block in self.transformer_blocks:
            tokens = block(tokens)
        if self.num_transformer_blocks > 0:
            tokens = self.ln(tokens)

        seg_patch_tokens = tokens[:, 1:]
        p = int(math.sqrt(seg_patch_tokens.size(1)))
        assert p * p == seg_patch_tokens.size(1)

        patch_tokens_spatial = seg_patch_tokens.permute(0, 2, 1).reshape(
            seg_patch_tokens.size(0), seg_patch_tokens.size(2), p, p
        )

        H, W = x.shape[2:]
        target_h, target_w = H // 4, W // 4

        # Upsample frozen backbone features first, then decode at target resolution.
        if self.use_transpose_conv:
            patch_tokens_spatial = self.upsample_conv1(patch_tokens_spatial)
            patch_tokens_spatial = self.upsample_conv2(patch_tokens_spatial)
        else:
            patch_tokens_spatial = F.interpolate(
                patch_tokens_spatial,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        for conv_block in self.conv_blocks:
            if self.use_bottleneck:
                patch_tokens_spatial = conv_block(patch_tokens_spatial)
            else:
                identity = patch_tokens_spatial
                patch_tokens_spatial = conv_block(patch_tokens_spatial)
                patch_tokens_spatial = patch_tokens_spatial + identity
                patch_tokens_spatial = F.relu(patch_tokens_spatial)

        segmentation = self.lin_classifier(patch_tokens_spatial)

        return {"seg": segmentation, "clip": patch_joint}
