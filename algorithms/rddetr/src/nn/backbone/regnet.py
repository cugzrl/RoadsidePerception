import torch
import torch.nn as nn 
from transformers import RegNetModel


from src.core import register

# 实现了一个基于 Hugging Face Transformers 库的 RegNet 模型封装
# 预训练模型加载：从 Hugging Face 加载预训练的 RegNet-Y-040 模型
# 多尺度特征提取：通过返回不同层级的隐藏状态，支持特征金字塔网络 (FPN) 等应用
# 模块化设计：通过@register装饰器将模型注册到框架中，便于集成到更大的系统中

__all__ = ['RegNet']

@register
class RegNet(nn.Module):
    def __init__(self, configuration, return_idx=[0, 1, 2, 3]):
        super(RegNet, self).__init__()  
        self.model = RegNetModel.from_pretrained("facebook/regnet-y-040")
        self.return_idx = return_idx


    def forward(self, x):
        
        outputs = self.model(x, output_hidden_states = True)
        x = outputs.hidden_states[2:5]

        return x