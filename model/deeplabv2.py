"""ResNet101-based DeepLabV2 (Chen et al., 2018).

The head follows kazuto1011/deeplab-pytorch, the implementation CLIP-ES points to for its
segmentation stage: four parallel 3x3 atrous convolutions (rates 6/12/18/24) mapping the
2048-channel stage-5 features straight to class logits, summed together, no batch norm.

The backbone is torchvision's ImageNet ResNet101 with the last two stages dilated instead
of strided (output stride 8, as in the reference). We use torchvision's ResNet rather than
the reference's Caffe-style one so the released ImageNet weights match the blocks they were
trained with; the reference's Caffe checkpoints are not available here.
"""

import torch.nn as nn
from torchvision.models import ResNet101_Weights, resnet101


class ASPP(nn.Module):
    """DeepLabV2 head: sum of parallel atrous convolutions straight to class logits."""

    def __init__(self, in_channels, num_classes, rates=(6, 12, 18, 24)):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Conv2d(in_channels, num_classes, 3, padding=rate, dilation=rate, bias=True)
            for rate in rates
        )
        for conv in self.branches:
            nn.init.normal_(conv.weight, mean=0.0, std=0.01)
            nn.init.constant_(conv.bias, 0.0)

    def forward(self, x):
        return sum(branch(x) for branch in self.branches)


class DeepLabV2(nn.Module):
    """Dilated ResNet101 + DeepLabV2 ASPP head. Logits come out at 1/8 of the input size."""

    def __init__(self, num_classes=21, pretrained_backbone=True):
        super().__init__()
        weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        resnet = resnet101(weights=weights, replace_stride_with_dilation=[False, True, True])
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.aspp = ASPP(2048, num_classes)

    def forward(self, x):
        return self.aspp(self.backbone(x))

    def train(self, mode=True):
        """Batch norm stays frozen (running stats and affine), as in the reference: the
        backbone statistics come from ImageNet and the batches here are small."""
        super().train(mode)
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def param_groups(self, learning_rate, weight_decay):
        """Reference learning-rate multipliers: 1x on backbone convolutions, 10x on head
        weights, 20x on head biases (without weight decay). Frozen batch norm parameters
        are left out of the optimizer entirely."""
        backbone = [m.weight for m in self.backbone.modules() if isinstance(m, nn.Conv2d)]
        head_weights = [conv.weight for conv in self.aspp.branches]
        head_biases = [conv.bias for conv in self.aspp.branches]
        return [
            {'params': backbone, 'lr': learning_rate, 'weight_decay': weight_decay},
            {'params': head_weights, 'lr': 10 * learning_rate, 'weight_decay': weight_decay},
            {'params': head_biases, 'lr': 20 * learning_rate, 'weight_decay': 0.0},
        ]
