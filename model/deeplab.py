# all files in /model dir are re-implemented from https://github.com/VainF/DeepLabV3Plus-Pytorch/tree/master

from .resnet import ResNet101
from ._deeplab import DeepLabHeadV3, DeepLabHeadV3Plus
from .utils import ClassificationAndSegmentationModel, IntermediateLayerGetter, SimpleSegmentationModel

# assumes:
# - output_stride = 8
# - pretrained_backbone = True
def deeplabv3plus_resnet101(num_classes):
    replace_stride_with_dilation = [False, True, True]
    backbone = ResNet101(replace_stride_with_dilation, num_classes)

    return_layers = {'layer4': 'feature', 'layer1': 'low_level', 'cam': 'cam', 'flatten': 'class'}
    backbone = IntermediateLayerGetter(backbone, return_layers)

    inplanes = 2048
    low_level_planes = 256
    aspp_dilate = [12, 24, 36]
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)

    model = SimpleSegmentationModel(backbone, classifier)
    return model

def deeplabv3_resnet101(num_classes):
    replace_stride_with_dilation = [False, True, True]
    backbone = ResNet101(replace_stride_with_dilation, num_classes)

    return_layers = {'layer4': 'feature'}
    backbone = IntermediateLayerGetter(backbone, return_layers)

    inplanes = 2048
    aspp_dilate = [12, 24, 36]
    classifier = DeepLabHeadV3(inplanes, num_classes, aspp_dilate)

    model = SimpleSegmentationModel(backbone, classifier)
    return model