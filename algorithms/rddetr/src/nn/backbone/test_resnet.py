import torch
import torch.nn as nn 
import torch.nn.functional as F 

from collections import OrderedDict


from src.core import register

# 实现了一个简化版的 ResNet 网络，称为 MResNet，专为图像分类任务设计
# MResNet 是一个基于残差网络 (ResNet) 的图像分类模型，主要包含以下组件：
# BasicBlock：基础残差块，由两个 3x3 卷积层组成
# _ResNet：核心网络结构，堆叠多个残差块形成完整网络
# MResNet：对外接口，封装_ResNet 并提供注册功能


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()         
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))       
        out += self.shortcut(x)          
        out = F.relu(out)
        return out



class _ResNet(nn.Module):
    # 网络结构：
    #     输入层：3x3 卷积，输出 64 通道
    #     四个残差层：分别包含 2,2,2,2 个残差块（默认配置）
    #     输出层：全局平均池化 + 全连接层
    # 下采样策略：
    #     除第一个残差层外，其余每层的第一个残差块使用 stride=2 进行下采样
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion 
        return nn.Sequential(*layers)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)              
        return out
        

@register
class MResNet(nn.Module):
    # 封装接口（MResNet）
    def __init__(self, num_classes=10, num_blocks=[2, 2, 2, 2]) -> None:
        super().__init__()
        self.model = _ResNet(BasicBlock, num_blocks, num_classes)
        
    def forward(self, x):
        return self.model(x)

