import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import os
from functools import partial
import sys

DINOV3_LOCATION = '/u501/j234li/wsss/model/dinov3'
sys.path.append(DINOV3_LOCATION)

from dinov3.layers import SelfAttentionBlock, SwiGLUFFN
from dinov3.models.vision_transformer import init_weights_vit
from dinov3.utils import named_apply
from model.resnet import Bottleneck


class DinoWSSS(nn.Module):
    def __init__(
        self,
        backbone_name: str = "dinov3_vitl16",
        num_transformer_blocks: int = 2,
        num_conv_blocks: int = 3,
        out_channels: int = 21,
        transformer_drop_path: float = 0.0,
        use_bottleneck: bool = False,
    ):
        super().__init__()

        self.num_transformer_blocks = num_transformer_blocks
        self.num_conv_blocks = num_conv_blocks
        self.use_bottleneck = use_bottleneck

        self.backbone = self._load_pretrained_backbone(backbone_name)
        self.backbone_dim = self.backbone.embed_dim
        self.num_heads = self.backbone.num_heads
        self.patch_token_layer = 1
        self.num_register_tokens = 0
        if hasattr(self.backbone, "num_register_tokens"):
            self.num_register_tokens = self.backbone.num_register_tokens
        elif hasattr(self.backbone, "n_storage_tokens"):
            self.num_register_tokens = self.backbone.n_storage_tokens
        
        # Transformer blocks
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
            # Use bottleneck blocks from resnet.py
            # Bottleneck expansion is 4, so planes = backbone_dim // 4 to get output of backbone_dim
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
                    norm_layer=nn.BatchNorm2d
                )
                for _ in range(num_conv_blocks)
            ]
        else:
            # Use basic blocks (original implementation)
            conv_list = [
                nn.Sequential(
                    nn.Conv2d(self.backbone_dim, self.backbone_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(self.backbone_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.backbone_dim, self.backbone_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(self.backbone_dim)
                )
                for _ in range(num_conv_blocks)
            ]
        self.conv_blocks = nn.ModuleList(conv_list)
        self.conv_final = nn.Conv2d(self.backbone_dim, out_channels, kernel_size=1, padding=0, bias=True)

        self.init_weights()

    def _load_pretrained_backbone(self, backbone_name: str):
        """Load pretrained DINOv3 backbone"""
        if backbone_name != "dinov3_vitl16":
            raise NotImplementedError(f"Backbone {backbone_name} not implemented")
        
        backbone = torch.hub.load(
            DINOV3_LOCATION,
            backbone_name,
            source="local",
            weights=os.path.join(DINOV3_LOCATION, 'weights', 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
        )
        return backbone

    def init_weights(self):
        """Initialize weights for segmentation blocks"""
        if self.num_transformer_blocks > 0:
            for block in self.transformer_blocks:
                named_apply(init_weights_vit, block)
            self.ln.reset_parameters()
        if self.num_conv_blocks > 0:
            for block in self.conv_blocks:
                for m in block.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
        
        nn.init.kaiming_normal_(self.conv_final.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv_final.bias, 0)
    
    def get_backbone_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.backbone.get_intermediate_layers(
            x,
            n=self.patch_token_layer, # take last layer
            return_class_token=True,
            return_extra_tokens=True,
        )
        class_token = tokens[-1][1]
        patch_tokens = tokens[0][0]
        register_tokens = tokens[0][2]
        return class_token, patch_tokens, register_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and segmentation blocks"""
        # Extract features from pretrained backbone
        with torch.no_grad():
            class_token, patch_tokens, register_tokens = self.get_backbone_features(x)
        
        tokens = torch.cat([class_token.unsqueeze(1), register_tokens, patch_tokens], dim=1)
        for block in self.transformer_blocks:
            tokens = block(tokens)
        tokens = self.ln(tokens)
        
        # Extract patch tokens for spatial processing
        patch_tokens = tokens[:, self.num_register_tokens + 1:] # (B, P, D)
        
        # Get patch grid dimensions from patch tokens
        p = int(math.sqrt(patch_tokens.size(1)))
        assert p * p == patch_tokens.size(1), "non-square patch grid"
        
        # Reshape patch tokens to spatial format [B, embed_dim, H, W]
        patch_tokens_spatial = patch_tokens.permute(0, 2, 1).view(patch_tokens.size(0), patch_tokens.size(2), p, p)

        # Upsample patch tokens to original image size (4x downsampled)
        H, W = x.shape[2:]
        patch_tokens_spatial = F.interpolate(patch_tokens_spatial, size=(H // 4, W // 4), mode='bilinear', align_corners=False)

        # Process through conv blocks
        for conv_block in self.conv_blocks:
            if self.use_bottleneck:
                # Bottleneck block already includes residual connection and ReLU
                patch_tokens_spatial = conv_block(patch_tokens_spatial)
            else:
                # Basic block: add residual connection and ReLU manually
                identity = patch_tokens_spatial
                patch_tokens_spatial = conv_block(patch_tokens_spatial)
                patch_tokens_spatial = patch_tokens_spatial + identity
                patch_tokens_spatial = F.relu(patch_tokens_spatial)
        
        # Final classification layer
        output = self.conv_final(patch_tokens_spatial)
        output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        
        return output

