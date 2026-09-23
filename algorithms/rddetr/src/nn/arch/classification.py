import torch 
import torch.nn as nn 

from src.core import register

# 定义了一个通用的图像分类模型架构，包含主干网络和分类头两个核心组件
# Classification：主模型，负责连接主干网络（backbone）和分类头（head）
# ClassHead：分类头，负责特征池化和类别预测

__all__ = ['Classification', 'ClassHead']


@register
class Classification(nn.Module):
    __inject__ = ['backbone', 'head']

    def __init__(self, backbone: nn.Module, head: nn.Module=None):
        super().__init__()
        
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)

        if self.head is not None:
            x = self.head(x)

        return x 


@register
class ClassHead(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(hidden_dim, num_classes)  

    def forward(self, x):
        x = x[0] if isinstance(x, (list, tuple)) else x 
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.proj(x)
        return x 
