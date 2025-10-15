import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from sklearn.cluster import MiniBatchKMeans
import os

DINOV3_LOCATION = '/u501/j234li/wsss/model/dinov3'
import sys
sys.path.append(DINOV3_LOCATION)
from dinov3.data.transforms import make_classification_eval_transform

class DinoWSSS(nn.Module):
    def __init__(self):
        super(DinoWSSS, self).__init__()

        self.dino_txt, self.tokenizer = torch.hub.load(DINOV3_LOCATION,
            'dinov3_vitl16_dinotxt_tet1280d20h24l', 
            source='local',
            weights=os.path.join(DINOV3_LOCATION, 'dinov3_vitl16_dinotxt_vision_head_and_text_encoder.pth'), 
            backbone_weights=os.path.join(DINOV3_LOCATION, 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'))
        
    def forward(self, x):
        return self.model(x)