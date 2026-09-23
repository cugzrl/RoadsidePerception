import torch
import torch.nn as nn 
import torchvision.transforms as T
from torch.cuda.amp import autocast
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import os 
import cv2
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
import numpy as np
from tqdm import tqdm



# category_map = {0:"Car", 1:"Truck", 2:"Van", 3:"Bus", 4:"Pedestrian", 5:"Cyclist", 6:"Tricyclist", 
#                 7:"Barrowlist", 8:"Trafficcone"}
# category_map = {0:"Car", 1:"Van", 2:"Bus", 3:"Tricycle", 4:"ScooterRider", 5:"Cyclist", 6:"Pedestrian"}
# category_map = {0:"Autonomous passenger vehicle", 1:"Autonomous commercial vehicle", 
#                   2:"Autonomous sweeping vehicle", 3:"Autonomous delivery vehicle", 
#                   4:"Movable barrier", 5:"Vehicle debris", 6:"TrafficCone", 
#                   7:"Adult", 8:"Child", 9:"Triclyclist", 10:"Cyclist", 11:"Barrowlist", 
#                   12:"Truck", 13:"Bus", 14:"Van", 15:"Car", 16:"Motocyclist", 
#                   17:"Animal", 18:"Police car", 19:"Fire truck", 20:"Ambulance", 21:"Others"}
category_map = {0:"Car", 1:"Van", 2:"Bus", 3:"Tricycle", 4:"ScooterRider", 5:"Cyclist", 6:"Pedestrian"}

def generate_colors(class_id, bgr=False):
    hex_colors = (
        '042AFF', '0BDBEB', '2C99A8', '00DFB7', '111F68', 'FF6FDD', 'FF444F', 'CCED00', '1A9334', '00D4BB',
               'F3F3F3', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
    color = tuple(int(hex_colors[class_id % len(hex_colors)][i:i+2], 16) for i in (0, 2, 4))
    return (color[2], color[1], color[0]) if bgr else color

def postprocess(labels, boxes, scores, iou_threshold=0.55):
    def calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        return iou
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    used_indices = set()
    for i in range(len(boxes)):
        if i in used_indices:
            continue
        current_box = boxes[i]
        current_label = labels[i]
        current_score = scores[i]
        boxes_to_merge = [current_box]
        scores_to_merge = [current_score]
        used_indices.add(i)
        for j in range(i + 1, len(boxes)):
            if j in used_indices:
                continue
            if labels[j] != current_label:
                continue  
            other_box = boxes[j]
            iou = calculate_iou(current_box, other_box)
            if iou >= iou_threshold:
                boxes_to_merge.append(other_box.tolist())  
                scores_to_merge.append(scores[j])
                used_indices.add(j)
        xs = np.concatenate([[box[0], box[2]] for box in boxes_to_merge])
        ys = np.concatenate([[box[1], box[3]] for box in boxes_to_merge])
        merged_box = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
        merged_score = max(scores_to_merge)
        merged_boxes.append(merged_box)
        merged_labels.append(current_label)
        merged_scores.append(merged_score)
    return [np.array(merged_labels)], [np.array(merged_boxes)], [np.array(merged_scores)]

def slice_image(image, slice_height, slice_width, overlap_ratio):
    img_width, img_height = image.size
    
    slices = []
    coordinates = []
    step_x = int(slice_width * (1 - overlap_ratio))
    step_y = int(slice_height * (1 - overlap_ratio))
    
    for y in range(0, img_height, step_y):
        for x in range(0, img_width, step_x):
            box = (x, y, min(x + slice_width, img_width), min(y + slice_height, img_height))
            slice_img = image.crop(box)
            slices.append(slice_img)
            coordinates.append((x, y))
    return slices, coordinates

def merge_predictions(predictions, slice_coordinates, orig_image_size, slice_width, slice_height, threshold=0.30):
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    orig_height, orig_width = orig_image_size
    for i, (label, boxes, scores) in enumerate(predictions):
        x_shift, y_shift = slice_coordinates[i]
        scores = np.array(scores).reshape(-1)
        valid_indices = scores > threshold
        valid_labels = np.array(label).reshape(-1)[valid_indices]
        valid_boxes = np.array(boxes).reshape(-1, 4)[valid_indices]
        valid_scores = scores[valid_indices]
        for j, box in enumerate(valid_boxes):
            box[0] = np.clip(box[0] + x_shift, 0, orig_width)  
            box[1] = np.clip(box[1] + y_shift, 0, orig_height)
            box[2] = np.clip(box[2] + x_shift, 0, orig_width)  
            box[3] = np.clip(box[3] + y_shift, 0, orig_height) 
            valid_boxes[j] = box
        merged_labels.extend(valid_labels)
        merged_boxes.extend(valid_boxes)
        merged_scores.extend(valid_scores)
    return np.array(merged_labels), np.array(merged_boxes), np.array(merged_scores)
def draw(images, labels, boxes, scores, thrh=0.6, save_dir='', filename=''):

    font_scale = 0.6 
    font_thickness = 1
    
    
    dir_path = os.path.dirname(filename)
    file_name_without_ext = os.path.splitext(filename)[0]
    # save_path = os.path.join(dir_path, file_name_without_ext + '.txt')
    
    try:
        font = ImageFont.truetype("Arial.ttf", size=int(24 * font_scale)) 
    except:
        for img_idx, im in enumerate(images):
            img_cv = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
            scr = scores[img_idx]
            valid_mask = scr > thrh
            lab = labels[img_idx][valid_mask]
            box = boxes[img_idx][valid_mask]
            scrs = scores[img_idx][valid_mask]
            
            for j, b in enumerate(box):
                score = scrs[j]
                class_id = int(lab[j])
                class_color = generate_colors(class_id, bgr=True)
                text = f"{category_map[class_id]} {scrs[j]:.2f}"
                
                
                center_width = ( b[2] + b[0]) / 2 / 1920
                center_height = ( b[1] + b[3]) / 2 / 1080
                width_ratio = (b[2] - b[0]) / 1920
                height_ratio = (b[3] - b[1]) / 1080
                text_save = f"{class_id} {center_width:.8f} {center_height:.8f} {width_ratio:.8f} {height_ratio:.8f}"
                # with open(save_path, 'a', encoding='utf-8') as f:
                #     f.write(text_save + '\n')
                
                
                # 框
                p1, p2 = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))
                cv2.rectangle(img_cv, p1, p2, class_color, thickness=2)
                
                # 文本绘制参数
                font = cv2.FONT_HERSHEY_COMPLEX
                text_w, text_h = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                
                # 计算位置
                text_x = p1[0]
                text_y = p1[1] - 5 if p1[1] - 5 > text_h else p1[1] + text_h + 5
                
                # 背景框
                cv2.rectangle(img_cv, 
                            (text_x, text_y - text_h),
                            (text_x + text_w, text_y),
                            class_color, -1)
                
                # 文本
                cv2.putText(img_cv, text, (text_x, text_y), font, font_scale, 
                          (255, 255, 255), font_thickness, cv2.LINE_AA)
            
            
            im = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            os.makedirs(save_dir, exist_ok=True)
            output_path = os.path.join(save_dir, os.path.basename(filename))
            im.save(output_path)
        return

    # PIL绘制方案
    for img_idx, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[img_idx]
        valid_mask = scr > thrh
        lab = labels[img_idx][valid_mask]
        box = boxes[img_idx][valid_mask]
        scrs = scores[img_idx][valid_mask]

        for j, b in enumerate(box):
            class_color = generate_colors(int(lab[j]))
            text = f"{category_map[int(lab[j])]} {scrs[j]:.2f}"
            
            # 获取文本尺寸
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            
            # 计算绘制位置
            x = b[0]
            y = b[1] - text_h - 3  # 在框上方保留3像素间距
            
            # 绘制半透明背景
            draw.rectangle([x, y, x + text_w, y + text_h], fill=class_color + (100,))
            
            # 绘制文本
            draw.text((x, y), text, font=font, fill=(255, 255, 255))
            
            # 绘制边界框（加粗边框）
            draw.rectangle(list(b), outline=class_color, width=3)

        # 保存结果
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, os.path.basename(filename))
        im.save(output_path)
            
def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    
    if os.path.isdir(args.input_dir):
        img_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) 
                    if f.lower().endswith(('jpg', 'png', 'jpeg'))]
    else:
        img_files = [args.input_dir]

    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)
    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = Model().to(args.device)
    
    for img_path in tqdm(img_files, desc="Processing Images"):
        # im_pil = Image.open(args.im_file).convert('RGB')
        im_pil = Image.open(img_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)
        
        transforms = T.Compose([
            T.Resize((640, 640)),  
            T.ToTensor(),
        ])
        im_data = transforms(im_pil)[None].to(args.device)
        if args.sliced:
            num_boxes = args.numberofboxes
            
            aspect_ratio = w / h
            num_cols = int(np.sqrt(num_boxes * aspect_ratio)) 
            num_rows = int(num_boxes / num_cols)
            slice_height = h // num_rows
            slice_width = w // num_cols
            overlap_ratio = 0.2
            slices, coordinates = slice_image(im_pil, slice_height, slice_width, overlap_ratio)
            predictions = []
            for i, slice_img in enumerate(slices):
                slice_tensor = transforms(slice_img)[None].to(args.device)
                with autocast():  # Use AMP for each slice
                    output = model(slice_tensor, torch.tensor([[slice_img.size[0], slice_img.size[1]]]).to(args.device))
                torch.cuda.empty_cache() 
                labels, boxes, scores = output
                
                labels = labels.cpu().detach().numpy()
                boxes = boxes.cpu().detach().numpy()
                scores = scores.cpu().detach().numpy()
                predictions.append((labels, boxes, scores))
            
            merged_labels, merged_boxes, merged_scores = merge_predictions(predictions, coordinates, (h, w), slice_width, slice_height)
            labels, boxes, scores = postprocess(merged_labels, merged_boxes, merged_scores)
        else:
            output = model(im_data, orig_size)
            labels, boxes, scores = output 
        # draw([im_pil], labels, boxes, scores, 0.6, args.output_dir)
        draw([im_pil], labels, boxes, scores, 0.6, args.output_dir, filename=img_path)
    
  
if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-i', '--input-dir', type=str, default='/home/kemove/devdata1/xy/zzz/output_frames', help='输入图片目录路径')
    parser.add_argument('-o', '--output-dir', type=str, default='/home/kemove/devdata1/RodeTest/output-2', help='输出目录路径')
    parser.add_argument('-c', '--config', type=str, default="/home/kemove/devdata1/RodeTest/roadside_perception/algorithms/rddetr/configs/rddetr/rddetr_r50vd_6x_coco.yml")
    parser.add_argument('-r', '--resume', type=str, default="/home/kemove/devdata1/RodeTest/roadside_perception/models/detection/detection_R39.pth")
    parser.add_argument('-s', '--sliced', type=bool, default=False)
    parser.add_argument('-nc', '--numberofboxes', type=int, default=25)
    args = parser.parse_args()
    main(args)











