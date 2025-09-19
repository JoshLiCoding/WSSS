import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model, return_layers):
        orig_return_layers = return_layers
        return_layers = {k:v for k, v in return_layers.items()} # prevents mutation of original var
        layers = OrderedDict()

        # loops in registration order of immediate children (i.e. stops at avgpool)
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers
    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

class SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False) # upsample 4x
        return x