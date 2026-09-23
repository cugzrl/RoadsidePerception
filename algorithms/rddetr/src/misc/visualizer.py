""""by lyuwenyu
"""

import torch
import torch.utils.data

import torchvision
torchvision.disable_beta_transforms_warning()

import PIL 

# 提供了一个可视化 COCO 数据集样本的工具函数，能够展示图像及其标注框

__all__ = ['show_sample']

def show_sample(sample):
    """for coco dataset/dataloader
        显示图像与对应的边界框标注
        兼容 PIL 图像和 PyTorch 张量输入
        自动处理数据类型转换（确保与绘图函数兼容）
    """
    import matplotlib.pyplot as plt
    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes

    image, target = sample
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)

    image = F.convert_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(image, target["boxes"], colors="yellow", width=3)

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()
    fig.show()
    plt.show()

