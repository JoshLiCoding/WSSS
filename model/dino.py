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

DINOV3_LOCATION = '/u501/j234li/reg_loss/model/dinov3'
sys.path.append(DINOV3_LOCATION)

from dinov3.layers import SelfAttentionBlock, SwiGLUFFN
from dinov3.models.vision_transformer import init_weights_vit
from dinov3.utils import named_apply
from dinov3.eval.text.vision_tower import VisionHead
from model.resnet import Bottleneck


class DinoWSSS(nn.Module):
    def __init__(
        self,
        backbone_name: str = "dinov3_vitl16",
        num_transformer_blocks: int = 1,
        num_conv_blocks: int = 2,
        out_channels: int = 21,
        transformer_drop_path: float = 0.0,
        use_bottleneck: bool = False,
        use_transpose_conv: bool = False,
    ):
        super().__init__()

        self.num_transformer_blocks = num_transformer_blocks
        self.num_conv_blocks = num_conv_blocks
        self.use_bottleneck = use_bottleneck
        self.use_transpose_conv = use_transpose_conv

        # Load backbone and dino.txt head together so backbone is only loaded once
        self.backbone, self.dinotxt_head = self._load_pretrained_backbone_and_dinotxt_head(backbone_name)
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
        self.lin_classifier = nn.Conv2d(self.backbone_dim, out_channels, kernel_size=1, padding=0, bias=True)
        
        # Transpose convolutions for upsampling (if enabled)
        if use_transpose_conv:
            # Two depthwise transpose convs, each upsampling 2x (total 4x upsampling)
            # kernel_size=3, stride=2, padding=1, output_padding=1 gives 2x upsampling
            # groups=backbone_dim makes it depthwise (1 filter per input channel)
            self.upsample_conv1 = nn.ConvTranspose2d(
                self.backbone_dim, self.backbone_dim, 
                kernel_size=3, stride=2, padding=1, output_padding=1, 
                groups=self.backbone_dim, bias=False
            )
            self.upsample_conv2 = nn.ConvTranspose2d(
                self.backbone_dim, self.backbone_dim,
                kernel_size=3, stride=2, padding=1, output_padding=1,
                groups=self.backbone_dim, bias=False
            )
            # Initialize transpose convs to mimic bilinear upsampling
            self._init_bilinear_transpose_conv(self.upsample_conv1)
            self._init_bilinear_transpose_conv(self.upsample_conv2)

        self.init_weights()

    def _load_pretrained_backbone_and_dinotxt_head(self, backbone_name: str):
        if backbone_name != "dinov3_vitl16":
            raise NotImplementedError(f"Backbone {backbone_name} not implemented")
        # Load dinotxt composite (visual backbone + head) once from local hub
        dinotxt_model, _ = torch.hub.load(
            DINOV3_LOCATION,
            'dinov3_vitl16_dinotxt_tet1280d20h24l',
            source='local',
            weights=os.path.join(
                DINOV3_LOCATION,
                'weights',
                'dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth',
            ),
            backbone_weights=os.path.join(
                DINOV3_LOCATION,
                'weights',
                'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
            ),
        )
        # Extract backbone and head; both are already initialized with pretrained weights
        backbone = dinotxt_model.visual_model.backbone
        dinotxt_head = dinotxt_model.visual_model.head
        return backbone, dinotxt_head

    def _init_bilinear_transpose_conv(self, conv_transpose):
        """
        Initialize depthwise ConvTranspose2d weights to mimic bilinear upsampling.
        For 2x upsampling with kernel_size=3, stride=2, padding=1, output_padding=1.
        
        This creates a bilinear interpolation kernel that matches PyTorch's
        F.interpolate(..., mode='bilinear', align_corners=False) behavior.
        """
        kernel_size = conv_transpose.kernel_size[0]

        filt = np.array([
            [0.25, 0.5, 0.25],
            [0.5,  1.0, 0.5],
            [0.25, 0.5, 0.25]
        ], dtype=np.float32)
        
        # Set weights: for depthwise convolution, weight shape is [out_channels, 1, kernel_h, kernel_w]
        # Each channel gets its own copy of the bilinear kernel
        with torch.no_grad():
            weight = torch.zeros(conv_transpose.weight.shape, dtype=torch.float32)
            for i in range(conv_transpose.out_channels):
                weight[i, 0, :, :] = torch.from_numpy(filt)
            conv_transpose.weight.copy_(weight)
    
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
        
        # nn.init.kaiming_normal_(self.lin_classifier.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.lin_classifier.weight, mean=0.0, std=0.3)
        nn.init.constant_(self.lin_classifier.bias, 0)
    
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

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through backbone, segmentation blocks and dino.txt vision head.
        
        Returns:
            dict with keys:
                - 'segmentation': segmentation output (same as before)
                - 'dinotxt_patch_tokens': patch tokens from dino.txt head for pseudolabel generation
        """
        # Extract features from pretrained backbone (only once)
        with torch.no_grad():
            class_token, patch_tokens, register_tokens = self.get_backbone_features(x)
        
        # Process for segmentation
        tokens = torch.cat([class_token.unsqueeze(1), register_tokens, patch_tokens], dim=1)
        for block in self.transformer_blocks:
            tokens = block(tokens)
        if self.num_transformer_blocks > 0:
            tokens = self.ln(tokens)
        
        # Extract patch tokens for spatial processing
        seg_patch_tokens = tokens[:, self.num_register_tokens + 1:] # (B, P, D)
        
        # Get patch grid dimensions from patch tokens
        p = int(math.sqrt(seg_patch_tokens.size(1)))
        assert p * p == seg_patch_tokens.size(1), "non-square patch grid"
        
        # Reshape patch tokens to spatial format [B, embed_dim, H, W]
        patch_tokens_spatial = seg_patch_tokens.permute(0, 2, 1).view(seg_patch_tokens.size(0), seg_patch_tokens.size(2), p, p)

        # Upsample patch tokens to original image size (4x downsampled)
        H, W = x.shape[2:]
        target_h, target_w = H // 4, W // 4
        if self.use_transpose_conv:
            # Use transpose convolutions for upsampling (2x2 = 4x total)
            patch_tokens_spatial = self.upsample_conv1(patch_tokens_spatial)
            patch_tokens_spatial = self.upsample_conv2(patch_tokens_spatial)
        else:
            # Use bilinear interpolation (original method)
            patch_tokens_spatial = F.interpolate(patch_tokens_spatial, size=(target_h, target_w), mode='bilinear', align_corners=False)

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
        
        out = {}
        segmentation = self.lin_classifier(patch_tokens_spatial)
        out['seg'] = segmentation
        
        # Process for dino.txt pseudolabel generation (using same backbone features)
        # Process tokens through dino.txt head
        dinotxt_tokens = torch.cat([class_token.unsqueeze(1), register_tokens, patch_tokens], dim=1)
        # Explicitly disable gradients for dino.txt path
        with torch.no_grad():
            dinotxt_output = self.dinotxt_head(dinotxt_tokens)
        # Extract patch tokens from dino.txt head output
        dinotxt_patch_tokens = dinotxt_output[:, self.num_register_tokens + 1:]  # (B, P, embed_dim)
        out['dinotxt'] = dinotxt_patch_tokens
        
        return out

