"""身份特征骨干

ResNet-50 是 ImageNet 分类权重 行人网络是 DeepSORT 结构 在 MOT 上做身份分类
不同骨干的向量放在互不重叠的维度里 只在同类别内做余弦
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


PERSON_CLASSES = {"ScooterRider", "Cyclist", "Pedestrian", "Tricycle"}
VEHICLE_CLASSES = {"Car", "Van", "Bus"}
PERSON_DIM = 512
VEHICLE_DIM = 2048


class BasicBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, is_downsample: bool = False):
        super().__init__()
        stride = 2 if is_downsample else 1
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.downsample = None
        if is_downsample or c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(y + residual, inplace=True)


def _layers(c_in: int, c_out: int, repeat: int, downsample: bool) -> nn.Sequential:
    blocks = [BasicBlock(c_in if i == 0 else c_out, c_out, downsample if i == 0 else False) for i in range(repeat)]
    return nn.Sequential(*blocks)


class DeepSortNet(nn.Module):
    """输入 128x64 输出 512 维 L2 归一化特征 不使用分类头"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        self.layer1 = _layers(64, 64, 2, False)
        self.layer2 = _layers(64, 128, 2, True)
        self.layer3 = _layers(128, 256, 2, True)
        self.layer4 = _layers(256, 512, 2, True)
        self.avgpool = nn.AvgPool2d((8, 4), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return F.normalize(x, p=2, dim=1)


def load_deepsort(weights: Path, device: torch.device) -> DeepSortNet:
    model = DeepSortNet()
    checkpoint = torch.load(weights, map_location="cpu")
    state = checkpoint.get("net_dict", checkpoint.get("model", checkpoint))
    state = {key: value for key, value in state.items() if not key.startswith("classifier.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not name.startswith("classifier") for name in missing):
        real_missing = [name for name in missing if not name.startswith("classifier")]
        if real_missing or unexpected:
            raise RuntimeError(f"行人 ReID 权重不匹配 missing={real_missing[:6]} unexpected={unexpected[:6]}")
    model.to(device).eval()
    return model


def load_resnet50(weights: Path, device: torch.device) -> nn.Module:
    model = models.resnet50(weights=None)
    state = torch.load(weights, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.fc = nn.Identity()
    model.to(device).eval()
    return model


IMAGENET_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
    return IMAGENET_NORM(tensor)


def resize_stretch(crop: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)


def resize_letterbox(crop: np.ndarray, width: int, height: int) -> np.ndarray:
    """保持长宽比 空白填 ImageNet 均值 避免把骑行者压扁"""
    src_h, src_w = crop.shape[:2]
    if src_h < 1 or src_w < 1:
        return np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), (124, 116, 104), dtype=np.uint8)
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def resnet_batch(model: nn.Module, crops: list[np.ndarray], device: torch.device, full: bool) -> np.ndarray:
    tensors = []
    for crop in crops:
        if full:
            image = resize_stretch(crop, 224, 224)
        else:
            image = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)
            image = image[16:240, 16:240]
        tensors.append(_to_tensor(image))
    batch = torch.stack(tensors).to(device)
    with torch.inference_mode():
        output = model(batch)
        output = F.normalize(output, p=2, dim=1)
    return output.detach().cpu().numpy().astype(np.float32)


def deepsort_batch(model: DeepSortNet, crops: list[np.ndarray], device: torch.device, letterbox: bool) -> np.ndarray:
    tensors = []
    for crop in crops:
        image = resize_letterbox(crop, 64, 128) if letterbox else resize_stretch(crop, 64, 128)
        tensors.append(_to_tensor(image))
    batch = torch.stack(tensors).to(device)
    with torch.inference_mode():
        output = model(batch)
    return output.detach().cpu().numpy().astype(np.float32)
