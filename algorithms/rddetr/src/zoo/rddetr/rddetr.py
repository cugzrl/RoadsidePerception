"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register

# 实现了 RD-DETR（Real-Time Detection Transformer）模型的整体架构，作为端到端目标检测框架，它整合了骨干网络、编码器和解码器组件，并支持多尺度训练和部署优化

__all__ = ['RDDETR', ]


@register
class RDDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]
    # multi_scale: [480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800]
    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        
    def forward(self, x, targets=None):    # （8，3，640，640）
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
        # （8，3，multi_scale/8，multi_scale/8）
        x = self.backbone(x)    # return (8，512，80，80)  （8，1024，40，40） （8，2048，20，20）
        x = self.encoder(x)        
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
