"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import math
import os
import sys
import pathlib
from typing import Iterable

import torch
import torch.amp 

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)
import time



def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    
    start = time.time()
    
    first_run = True

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        start_time = time.time()
        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
        results = postprocessors(outputs, orig_target_sizes)
        end_time = time.time()
        if first_run:
            print("响应时间：", (end_time-start_time)/samples.size(0),"s")
            first_run = False
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:  # 1
            coco_evaluator.update(res)
    
    end = time.time()
    img_num = len(base_ds.imgs)
    # print(img_num)
    fps = img_num / (end - start)
    
    print("推理速度：", fps, "FPS")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    
    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    return stats, coco_evaluator




@torch.no_grad()
def beyond_evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir):
    
    intrinsic_params = {  #Bn
        "imageWidth": 1920,
        "imageHeight": 1080,
        "fx": 2956.55207,
        "fy": 2934.03689,
        "cx": 910.87352,
        "cy": 579.9378,
        "radialDistortion1": -0.400458,
        "radialDistortion2": 0.808235,
        "radialDistortion3": -0.002155,
        "radialDistortion4": -0.001043,
        "radialDistortion5": 0.0
    }

    extrinsic_params = {
        "translation": {"x": -885.367839, "y": -540.154756, "z": 17.974449},
        "rotation": {"w": 0.500304, "x": -0.611027, "y": -0.464303, "z": 0.400955}
    }
    # intrinsic_params = {   # Aw
    #     "imageWidth":1920,
    #     "imageHeight":1080,
    #     "fx":3000.0,
    #     "fy":3000.0,
    #     "cx":960.0,
    #     "cy":540.0,
    #     "radialDistortion1":-0.351076,
    #     "radialDistortion2":-0.255129,
    #     "radialDistortion3":-0.002178,
    #     "radialDistortion4":0.004446,
    #     "radialDistortion5":0.0
    # }
    # extrinsic_params = {
    #     "translation":
    #         {"x":-937.415508,
    #          "y":-481.026175,
    #          "z":18.106025},
    #     "rotation":
    #         {
    #         "w":-0.114747,
    #         "x":0.122503,
    #         "y":0.753435,
    #         "z":-0.635737
    #         }
    #     }  

    distance = calculate_60m(intrinsic_params, extrinsic_params)
    # print("distance: ", distance)
    
    model.eval()
    criterion.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    
    start = time.time()
    
    first_run = True

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        start_time = time.time()
        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
        results = postprocessors(outputs, orig_target_sizes)
        end_time = time.time()
        if first_run:
            print("响应时间：", (end_time-start_time)/samples.size(0),"s")
            first_run = False
        # 过滤非超视距范围内的检测结果和真值
        filter_results, filter_targets = filter(results, targets, distance)
        
        res = {target['image_id'].item(): output for target, output in zip(filter_targets, filter_results)}
        if coco_evaluator is not None:  # 1
            coco_evaluator.update(res)
    
    end = time.time()
    img_num = len(base_ds.imgs)
    print(img_num)
    fps = img_num / (end - start)
    
    print("推理速度：", fps, "FPS")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    
    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    return stats, coco_evaluator




def calculate_60m(intrinsic_params, extrinsic_params, image_width=1920, image_height=1080):
    """
    根据相机内参和外参计算60米距离在2D图像中对应的像素y坐标
    """
    
    fx = intrinsic_params['fx']
    fy = intrinsic_params['fy']
    cx = intrinsic_params['cx']
    cy = intrinsic_params['cy']

    focal_length = (fx + fy) / 2

    translation = extrinsic_params['translation']
    rotation_quat = extrinsic_params['rotation']

    camera_x = translation['x']
    camera_y = translation['y']
    camera_z = translation['z']
    # 相机安装高度
    camera_height = 6
    w, x, y, z = rotation_quat['w'], rotation_quat['x'], rotation_quat['y'], rotation_quat['z']
    pitch_angle_rad = math.asin(2 * (w * y - x * z))
    pitch_angle_deg = math.degrees(pitch_angle_rad)
    principal_y = cy
    viewing_angle = math.atan(camera_height / 60) - pitch_angle_rad
    distance = principal_y + focal_length * math.tan(viewing_angle)
    return distance


def filter(results, targets, pixel):
    
    original_height = 1080

    scale_factor = 640 / original_height
    scaled_y_max = int(pixel * scale_factor)
    
    filtered_results = []
    filtered_targets = []
    
    for target, result in zip(targets, results):
        # 过滤检测结果
        if 'boxes' in result and len(result['boxes']) > 0:
            boxes = result['boxes']
            scores = result['scores']
            labels = result['labels']
            
            center_ys = (boxes[:, 1] + boxes[:, 3]) / 2
            
            valid_mask = (center_ys >= 0) & (center_ys <= pixel)            #
            # valid_mask = (center_ys >= pixel) & (center_ys <= original_height)            #
            
            
            filtered_result = {
                'boxes': boxes[valid_mask],
                'scores': scores[valid_mask],
                'labels': labels[valid_mask]
            }
        else:
            filtered_result = result
        
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes']
            labels = target['labels']
            
            center_ys = (boxes[:, 1] + boxes[:, 3]) / 2
            
            valid_mask = (center_ys >= 0) & (center_ys <= scaled_y_max)           #
            # valid_mask = (center_ys >= scaled_y_max) & (center_ys <= original_height)            #
            
            
            filtered_target = target.copy()
            filtered_target['boxes'] = boxes[valid_mask]
            filtered_target['labels'] = labels[valid_mask]
            
            # 保留其他字段
            for key in ['image_id', 'orig_size', 'size']:
                if key in target:
                    filtered_target[key] = target[key]
        else:
            filtered_target = target
        
        filtered_results.append(filtered_result)
        filtered_targets.append(filtered_target)
    
    return filtered_results, filtered_targets
