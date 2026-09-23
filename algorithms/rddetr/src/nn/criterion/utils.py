import torch 
import torchvision


# 定义了一个用于格式化目标检测任务标签的函数 format_target，主要功能是将输入的目标列表转换为统一格式的张量，便于后续处理
def format_target(targets):
    '''
    Args:
        targets (List[Dict]),
    Return: 
        tensor (Tensor), [im_id, label, bbox,]
    '''
    outputs = []
    for i, tgt in enumerate(targets):
        boxes =  torchvision.ops.box_convert(tgt['boxes'], in_fmt='xyxy', out_fmt='cxcywh') 
        labels = tgt['labels'].reshape(-1, 1)
        im_ids = torch.ones_like(labels) * i
        outputs.append(torch.cat([im_ids, labels, boxes], dim=1))

    return torch.cat(outputs, dim=0)
