import torch
import torch.nn as nn

class DownsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=stride,  
            padding=padding,
            bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class SampleNetwork(nn.Module):
    def __init__(self):
        super(SampleNetwork, self).__init__()
        self.down1 = DownsampleBlock(3, 64)  # 下采样到1/2
        self.down2 = DownsampleBlock(64, 128)  # 下采样到1/4
        self.down3 = DownsampleBlock(128, 256)  # 下采样到1/8
        
    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return x
