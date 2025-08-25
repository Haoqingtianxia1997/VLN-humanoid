

# 单目标检测
# import torch
# import numpy as np
# from PIL import Image, ImageDraw
# from transformers import OwlViTProcessor, OwlViTForObjectDetection
# import cv2
# import os
# import requests

# class TextDrivenSegmenter:
#     def __init__(self):
#         # 初始化OwlViT模型
#         self.owlvit_processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
#         self.owlvit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14")
        
#         # 设备配置
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.owlvit_model = self.owlvit_model.to(self.device)
    
#     def detect_and_segment(self, image_path, text_prompt):
#         """
#         输入: 
#             image_path: 图像路径
#             text_prompt: 文本描述
#         输出:
#             segmented_image: 带分割结果的图像
#             bbox: 检测框坐标(xyxy)
#         """
#         # 步骤1: 使用OwlViT检测物体
#         image = Image.open(image_path).convert("RGB")
#         original_size = image.size
        
#         # OwlViT处理
#         inputs = self.owlvit_processor(
#             text=text_prompt if isinstance(text_prompt, list) else [text_prompt],
#             images=image,
#             return_tensors="pt"
#         ).to(self.device)
        
#         # 推理
#         with torch.no_grad():
#             outputs = self.owlvit_model(**inputs)
        
#         # 解析结果并选择最佳框
#         best_box = self._select_best_box(outputs, original_size)
#         if best_box is None:
#             print("No object detected with the given text prompt.")
#             return image, None
        
#         # 步骤2: 使用OpenCV进行简单分割（基于边界框）
#         mask = self._simple_segmentation(image_path, best_box)
        
#         # 可视化结果
#         segmented_image = self._visualize_results(image, best_box, None)
        
#         return segmented_image, best_box
    
#     def _select_best_box(self, outputs, target_size):
#         # 获取预测结果
#         pred_boxes = outputs.pred_boxes[0].detach().cpu()
#         pred_scores = outputs.logits[0].detach().cpu().sigmoid()
        
#         # 检查是否有检测结果
#         if len(pred_scores) == 0 or pred_scores.max() < 0.01:
#             return None
        
#         # 选择最高分检测框
#         best_idx = pred_scores.argmax()
#         best_score = pred_scores[best_idx].item()
#         box = pred_boxes[best_idx].numpy()
        
#         # 转换坐标格式 (xywh -> xyxy)
#         w, h = target_size
#         box = [
#             max(0, int(box[0] * w - box[2] * w / 2)),    # x_min
#             max(0, int(box[1] * h - box[3] * h / 2)),    # y_min
#             min(w, int(box[0] * w + box[2] * w / 2)),    # x_max
#             min(h, int(box[1] * h + box[3] * h / 2))     # y_max
#         ]
        
#         print(f"Detected object with confidence: {best_score:.3f}, Bounding box: {box}")
#         return box
    
#     def _simple_segmentation(self, image_path, bbox):
#         """使用OpenCV进行简单的基于边界框的分割"""
#         # 读取图像
#         img = cv2.imread(image_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         h, w = img.shape[:2]
        
#         # 创建空白掩码
#         mask = np.zeros((h, w), dtype=np.uint8)
        
#         # 在边界框区域内填充白色
#         x1, y1, x2, y2 = bbox
#         mask[y1:y2, x1:x2] = 1
        
#         return mask
    
#     def _visualize_results(self, image, bbox, mask=None, label=None, score=None):
#         draw = ImageDraw.Draw(image)
#         draw.rectangle(bbox, outline="red", width=3)
        
#         if label:
#             text = f"{label} ({score:.2f})" if score else label
#             draw.text((bbox[0], bbox[1] - 10), text, fill="red")

#         if mask is not None:
#             mask_np = (mask * 255).astype(np.uint8)
#             mask_pil = Image.fromarray(mask_np)
#             mask_rgba = mask_pil.convert("RGBA")
#             mask_rgba.putalpha(128)
#             image = image.convert("RGBA")
#             image.paste(mask_rgba, (0, 0), mask_rgba)
        
#         return image.convert("RGB")


# # 使用示例
# if __name__ == "__main__":
#     # 初始化分割器
#     segmenter = TextDrivenSegmenter()
    
#     # 输入参数
#     image_path = "example2.jpg"  # 替换为您的图像路径
#     text_prompt = "a man"   # 替换为您的文本描述
    
#     # 执行检测与分割
#     result_image, bbox = segmenter.detect_and_segment(image_path, text_prompt)
    
#     # 保存结果
#     if result_image:
#         result_image.save("result.jpg")
#         print(f"Saved result to result.jpg")
#         if bbox:
#             print(f"Bounding box: {bbox}")














# 支持多目标检测
# import torch
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont,ImageOps
# from transformers import OwlViTProcessor, OwlViTForObjectDetection
# import cv2
# import os

# class TextDrivenSegmenter:
#     def __init__(self):
#         self.owlvit_processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
#         self.owlvit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14")
        
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.owlvit_model = self.owlvit_model.to(self.device)

#         # 为每个 label 分配不同颜色
#         self.color_map = {}

#     def detect_and_segment(self, image_path, text_prompts):
#         image = Image.open(image_path).convert("RGB")
#         original_size = image.size
#         w, h = original_size

#         segmented_image = image.copy()
#         all_boxes = []

#         for prompt in text_prompts:
#             inputs = self.owlvit_processor(
#                 text=[prompt],
#                 images=image,
#                 return_tensors="pt"
#             ).to(self.device)

#             with torch.no_grad():
#                 outputs = self.owlvit_model(**inputs)

#             pred_boxes = outputs.pred_boxes[0].detach().cpu()
#             pred_scores = outputs.logits[0].detach().cpu().sigmoid()[:, 0]  # 单一prompt

#             # 获取得分前2的框
#             topk = torch.topk(pred_scores, k=min(1, len(pred_scores)))
#             top_idxs = topk.indices
#             top_scores = topk.values

#             for idx, score in zip(top_idxs, top_scores):
#                 if score < 0.1:  # 分数阈值
#                     continue

#                 box = pred_boxes[idx].numpy()
#                 box = [
#                     max(0, int(box[0] * w - box[2] * w / 2)),
#                     max(0, int(box[1] * h - box[3] * h / 2)),
#                     min(w, int(box[0] * w + box[2] * w / 2)),
#                     min(h, int(box[1] * h + box[3] * h / 2))
#                 ]

#                 mask = self._simple_segmentation(image_path, box)
#                 segmented_image = self._visualize_results(segmented_image, box, mask = None, label=prompt, score=score.item())
#                 all_boxes.append((box, prompt, score.item()))

#         return segmented_image, all_boxes

#     def _simple_segmentation(self, image_path, bbox):
#         img = cv2.imread(image_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         h, w = img.shape[:2]

#         mask = np.zeros((h, w), dtype=np.uint8)
#         x1, y1, x2, y2 = bbox
#         mask[y1:y2, x1:x2] = 1

#         return mask

#     def _visualize_results(self, image, bbox, mask=None, label=None, score=None):
#         if label not in self.color_map:
#             self.color_map[label] = self._get_random_color()

#         color = self.color_map[label]
#         draw = ImageDraw.Draw(image)

#         # Draw bounding box
#         draw.rectangle(bbox, outline=color, width=3)

#         if label:
#             text = f"{label} ({score:.2f})"
#             draw.text((bbox[0], bbox[1] - 10), text, fill=color)

#         if mask is not None:
#             # 将二值mask变成PIL灰度图
#             mask_pil = Image.fromarray((mask * 255).astype(np.uint8))

#             # 着色（前景为目标颜色，背景为透明黑）
#             mask_colored = ImageOps.colorize(mask_pil, black=(0, 0, 0, 0), white=color + (128,))  # 透明度 128
#             mask_colored = mask_colored.convert("RGBA")

#             # 图像合成
#             image = image.convert("RGBA")
#             image.paste(mask_colored, (0, 0), mask_colored)

#         return image.convert("RGB")

#     def _get_random_color(self):
#         return tuple(np.random.choice(range(64, 256), size=3))  # 明亮颜色


# # 使用示例
# if __name__ == "__main__":
#     segmenter = TextDrivenSegmenter()

#     image_path = "example2.jpg"
#     text_prompt = ["pig"] #"chinese", "person", "pig",

#     result_image, bbox = segmenter.detect_and_segment(image_path, text_prompt)

#     if result_image:
#         result_image.save("result.jpg")
#         print("Saved result to result.jpg")
#         if bbox:
#             for box, label, score in bbox:
#                 print(f"Detected {label} at {box} with confidence {score:.2f}")













# # 支持多目标检测 + FastSAM分割
# import torch
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont, ImageOps
# from transformers import OwlViTProcessor, OwlViTForObjectDetection
# import cv2
# import os
# from ultralytics import YOLO


# class TextDrivenSegmenter:
#     def __init__(self, fastsam_model_path='FastSAM-x.pt'):
#         # 初始化OwlViT模型
#         self.owlvit_processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
#         self.owlvit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14")
        
#         # 初始化FastSAM模型
#         self.fastsam_model = YOLO(fastsam_model_path)
        
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.owlvit_model = self.owlvit_model.to(self.device)
#         if torch.cuda.is_available():
#             self.fastsam_model.to(self.device)

#         # 为每个 label 分配不同颜色
#         self.color_map = {}
        
#         # FastSAM默认配置
#         self.fastsam_conf = 0.4
#         self.fastsam_iou = 0.9
#         self.fastsam_imgsz = 1024

#     def detect_and_segment(self, image_path, text_prompts):
#         # 打开并处理原始图像
#         image = Image.open(image_path).convert("RGB")
#         original_size = image.size
#         w, h = original_size
        
#         # 创建用于可视化的图像副本
#         segmented_image = image.copy()
#         all_boxes = []
#         all_masks = []
#         all_points = []

#         # 使用OwlViT进行目标检测
#         for prompt in text_prompts:
#             inputs = self.owlvit_processor(
#                 text=[prompt],
#                 images=image,
#                 return_tensors="pt"
#             ).to(self.device)

#             with torch.no_grad():
#                 outputs = self.owlvit_model(**inputs)

#             pred_boxes = outputs.pred_boxes[0].detach().cpu()
#             pred_scores = outputs.logits[0].detach().cpu().sigmoid()[:, 0]  # 单一prompt

#             # 获取得分前2的框
#             topk = torch.topk(pred_scores, k=min(2, len(pred_scores)))
#             top_idxs = topk.indices
#             top_scores = topk.values

#             for idx, score in zip(top_idxs, top_scores):
#                 if score < 0.1:  # 分数阈值
#                     continue

#                 box = pred_boxes[idx].numpy()
#                 box = [
#                     max(0, int(box[0] * w - box[2] * w / 2)),
#                     max(0, int(box[1] * h - box[3] * h / 2)),
#                     min(w, int(box[0] * w + box[2] * w / 2)),
#                     min(h, int(box[1] * h + box[3] * h / 2))
#                 ]

#                 # 使用FastSAM进行分割
#                 mask = self._fastsam_segmentation(image_path, box)
#                 box_center_point , seg_center_point = self.get_box_and_mask_center(box, mask)
#                 # box_center_point, seg_center_point = (0,0) , (0,0)
#                 all_points.append({"target": prompt ,"box_center_point": box_center_point, "seg_center_point": seg_center_point})
#                 all_boxes.append((box, prompt, score.item()))
#                 all_masks.append(mask)

#         # 使用FastSAM进行分割后可视化
#         segmented_image = self._visualize_results(segmented_image, all_boxes, all_masks)
        
#         return segmented_image, all_boxes , all_points
    

#     def get_box_and_mask_center(self, box, mask):
#         """
#         box: (x1, y1, x2, y2)，均为像素坐标
#         mask: HxW numpy array，掩码区域是1（或255），其余为0
        
#         返回: 
#         - box中心 (cx, cy) ，int类型
#         - mask=1区域的重心 (mcx, mcy) ，int类型；如果mask全为0，返回(None, None)
#         """
#         x1, y1, x2, y2 = box
#         cx = int((x1 + x2) / 2)
#         cy = int((y1 + y2) / 2)
        
#         # mask中心（重心）
#         ys, xs = np.where(mask > 0)  # 找到所有前景像素
#         if len(xs) == 0 or len(ys) == 0:
#             mcx, mcy = None, None  # 没有前景
#         else:
#             mcx = int(np.mean(xs))
#             mcy = int(np.mean(ys))
#         return (cx, cy), (mcx, mcy)    

#     def filter_largest_region(self, mask):
        
#         for i in range(3):# 形态学开运算去杂质
#             kernel = np.ones((21, 21), np.uint8)
#             mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

#             num_labels, labels_im = cv2.connectedComponents(mask.astype(np.uint8))
#             if num_labels <= 1:
#                 return mask
#             max_region = 0
#             max_area = 0
#             for i in range(1, num_labels):  # 跳过背景0
#                 area = np.sum(labels_im == i)
#                 if area > max_area:
#                     max_area = area
#                     max_region = i
#             mask = np.zeros_like(mask, dtype=np.uint8)
#             mask[labels_im == max_region] = 1

  
#         return mask
    

#     def _fastsam_segmentation(self, image_path, bbox):
#         """
#         对检测框box内的区域做分割，并将掩码映射回原图位置。
#         """
#         # 读取原始图片
#         image = Image.open(image_path).convert("RGB")
#         w, h = image.size
#         x1, y1, x2, y2 = bbox
#         x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

#         # 裁剪box区域
#         cropped = image.crop((x1, y1, x2, y2))
#         cropped_np = np.array(cropped)

#         # 保存临时图片（或直接传数组看FastSAM支持与否，ultralytics/YOLO通常支持ndarray）
#         # cropped_path = "tmp_cropped.jpg"
#         # cropped.save(cropped_path)

#         # 用FastSAM对裁剪区域做分割
#         results = self.fastsam_model.predict(
#             source=cropped_np,  # 注意这里用数组，不是路径
#             conf=self.fastsam_conf,
#             iou=self.fastsam_iou,
#             imgsz=self.fastsam_imgsz,
#             device=self.device,
#             verbose=False
#         )

#         # 找最大面积的掩码
#         best_mask = None
#         max_area = -1

#         for result in results:
#             if result.masks is None:
#                 continue
#             for i, mask in enumerate(result.masks.data):
#                 mask_np = mask.cpu().numpy().astype(np.uint8)
#                 area = mask_np.sum()
#                 if area > max_area:
#                     max_area = area
#                     best_mask = mask_np

#         # 创建与原图同尺寸的全零掩码
#         final_mask = np.zeros((h, w), dtype=np.uint8)
#         if best_mask is not None:
#             best_mask = self.filter_largest_region(best_mask)
#             # 将分割掩码放回原图对应位置
#             mask_h, mask_w = best_mask.shape
#             # 确保尺寸和box一致
#             if mask_h != (y2 - y1) or mask_w != (x2 - x1):
#                 best_mask = cv2.resize(best_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
#             final_mask[y1:y2, x1:x2] = best_mask

#         return final_mask



#     def _visualize_results(self, image, all_boxes, all_masks):
#         # 创建用于绘制的图像
#         draw_image = image.copy()
#         draw = ImageDraw.Draw(draw_image)
        
#         # 创建带透明度的掩码图层
#         overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        
#         # 为每个检测结果绘制边界框和掩码
#         for i, ((box, label, score), mask) in enumerate(zip(all_boxes, all_masks)):

                
#             # 获取或创建颜色
#             if label not in self.color_map:
#                 self.color_map[label] = self._get_random_color()
#             color = (255, 0, 0)#self.color_map[label]
            
#             # 绘制边界框
#             draw.rectangle(box, outline=color, width=3)
            
#             # 绘制标签和置信度
#             text = f"{label} ({score:.2f})"
#             try:
#                 font = ImageFont.truetype("arial.ttf", 20)
#             except:
#                 font = ImageFont.load_default()
#             draw.text((box[0], box[1] - 25), text, fill=color, font=font)
            
#             if mask is None or not mask.any():
#                 continue
#             # 绘制分割掩码
#             # 创建彩色掩码
#             mask_color = color + (200,)  # 添加透明度
#             mask_image = Image.fromarray((mask * 255).astype(np.uint8))
            
#             # 为掩码着色
#             colored_mask = Image.new('RGBA', image.size, (0, 0, 0, 0))
#             mask_data = np.array(mask_image)
            
#             # 只处理掩码区域
#             ys, xs = np.where(mask_data > 0)
#             for x, y in zip(xs, ys):
#                 colored_mask.putpixel((x, y), mask_color)
            
#             # 将彩色掩码添加到叠加层
#             overlay = Image.alpha_composite(overlay, colored_mask)
        
#         # 将掩码层与原始图像合并
#         result = Image.alpha_composite(draw_image.convert('RGBA'), overlay)
#         return result.convert('RGB')

#     def _get_random_color(self):
#         return tuple(np.random.choice(range(64, 256), size=3))


# if __name__ == "__main__":
#     # 初始化分割器 - 需要指定FastSAM模型路径
#     segmenter = TextDrivenSegmenter(fastsam_model_path='FastSAM-x.pt')
    
#     image_path = "example1.jpg"
#     text_prompts = ["red car"]  # 可以添加多个提示，如 ["person", "pig", "background"]
    
#     result_image, bbox, center_info = segmenter.detect_and_segment(image_path, text_prompts)
    
#     if result_image:
#         result_image.save("result_fastsam.jpg")
#         print("Saved result to result_fastsam.jpg")
#         if bbox:
#             for  i, ((box, label, score), loction_info) in enumerate(zip(bbox , center_info )):
#                 print(f"Detected {label} at {box} with confidence {score:.2f}")
#                 print(f"Center info for {label}: Box center: {loction_info['box_center_point']}, Segmentation center: {loction_info['seg_center_point']}")








# 支持多目标种类+文字检测 + FastSAM/SAM分割
import re, difflib,functools
import torch, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from ultralytics import YOLO
import easyocr
from transformers import MarianMTModel, MarianTokenizer
import torch
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw
from segment_anything import sam_model_registry, SamPredictor


# 伪彩色表（你可自定义）
COLOR_TABLE = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
]

# ---------------- 辅助函数 ----------------
def apply_mask_color(mask, color, alpha=120):
    """mask: [H, W] np.uint8 0-255, color: (R,G,B), alpha: 0-255"""
    mask_img = Image.fromarray(mask)
    color_img = Image.new("RGBA", mask_img.size, color + (0,))
    # 只在前景上加透明色
    mask_rgba = mask_img.convert("L").point(lambda x: alpha if x > 0 else 0)
    color_img.putalpha(mask_rgba)
    return color_img

def prompt_tokens(prompt: str):
    """把 prompt 拆成长度≥3 的英文单词列表，全小写"""
    return [w for w in re.findall(r"[a-zA-Z']+", prompt.lower()) if len(w) >= 3]

def fuzzy_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def prompt_match(tokens, ocr_tokens, fuzzy_th=0.8):
    """
    所有 tokens 必须全部匹配 OCR 才返回 True：
    ① 若 kw 是某个 OCR t 的子串 → 匹配成功
    ② 或模糊相似度 ≥ fuzzy_th    → 匹配成功
    """
    matched_scores = []

    for kw in tokens:
        kw = kw.lower()
        matched = False
        best_score = 0.0

        for t in ocr_tokens:
            t = t.lower()
            if kw in t:
                matched = True
                best_score = 1.0
                break  # 直接命中，退出内层
            score = fuzzy_ratio(kw, t)
            if score >= fuzzy_th:
                matched = True
                best_score = max(best_score, score)

        if not matched:
            return False, 0  # 只要有一个关键词不匹配 → 全部失败
        matched_scores.append(best_score)

    # 所有都匹配了，返回最小得分（最弱一项）
    return True, min(matched_scores)

# --------------------

class SAMSegmenter:
    def __init__(self, sam_ckpt="src/VLM_agent/sam_hq/sam_vit_h_4b8939.pth", device="cpu"):
        self.device = device
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).to(self.device)
        self.sam_predictor = SamPredictor(self.sam)

    def segment_with_boxes(self, image_path, boxes_xyxy, multimask_output=False):
        """
        image_path      : str
        boxes_xyxy      : (x0,y0,x1,y1)  或  [(...), (...)]   像素坐标
        returns         : numpy.bool_  [N,H,W]  (multi=False)  或  [N,3,H,W]
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        # --- 标准化 boxes 到 [N,4] ---
        if isinstance(boxes_xyxy[0], (int, float)):
            boxes_arr = np.asarray([boxes_xyxy], dtype=np.float32)
        else:
            boxes_arr = np.asarray(boxes_xyxy, dtype=np.float32)

        self.sam_predictor.set_image(img_rgb)
        inp_boxes = torch.as_tensor(boxes_arr, device=self.device)           # [N,4]
        tr_boxes  = self.sam_predictor.transform.apply_boxes_torch(
                        inp_boxes, (H, W))

        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=tr_boxes,
            multimask_output=multimask_output            # => [N,M,H,W]
        )

        masks = masks.bool().cpu().numpy()               # → numpy
        if not multimask_output:                         # [N,1,H,W] → [N,H,W]
            masks = masks[:, 0]

        return masks                   

class TextDrivenSegmenter:
    def __init__(self, fastsam_model_path='src/VLM_agent/FastSAM/FastSAM-x.pt', use_gpu=False):
        """Wrapper that loads OwlViT + (Fast)SAM.

        fastsam_model_path: Pfad zur FastSAM-x.pt. Falls nicht vorhanden wird versucht
        sie relativ zum aktuellen Dateiordner unter FastSAM/FastSAM-x.pt zu finden.
        """
        import os
        from pathlib import Path
        # Robustere Pfadbehandlung (Backslashes aus Windows-String vermeiden)
        fastsam_model_path = fastsam_model_path.replace('\\', '/')
        if not os.path.isfile(fastsam_model_path):
            alt = Path(__file__).parent / 'FastSAM' / 'FastSAM-x.pt'
            if alt.is_file():
                fastsam_model_path = str(alt)
            else:
                raise FileNotFoundError(
                    f"FastSAM Gewichte nicht gefunden: '{fastsam_model_path}' oder '{alt}'. Bitte Datei ablegen.")
        self.owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
        self.owl_model     = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14")
        
        self.fastsam       = YOLO(fastsam_model_path)
        self.sam = SAMSegmenter(sam_ckpt="src/VLM_agent/sam_hq/sam_vit_h_4b8939.pth")
        
        self.reader        = easyocr.Reader(['de'], gpu=use_gpu)
        self.device        = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        self.owl_model.to(self.device)
        if self.device == "cuda":
            self.fastsam.to(self.device)
        self.fastsam_conf, self.fastsam_iou, self.fastsam_size = 0.4, 0.9, 1024
        self.color_map = {}
        trans_name = "Helsinki-NLP/opus-mt-en-de"
        self.trans_tok = MarianTokenizer.from_pretrained(trans_name)
        self.trans_mod = MarianMTModel.from_pretrained(trans_name)
        self.trans_mod.to(self.device)      # 放同一张 GPU / CPU

    @functools.lru_cache(maxsize=512)
    @torch.inference_mode()
    def _en_word2de(self, word: str) -> str:
        batch = self.trans_tok(word, return_tensors="pt").to(self.device)
        gen   = self.trans_mod.generate(**batch, max_length=16)
        return self.trans_tok.decode(gen[0], skip_special_tokens=True)

    # ---------------- 主流程 ----------------
    def detect_and_segment(self, image_path, text_prompts, text_label, multi_task=False, if_sam=True, if_translate=False):
        image = Image.open(image_path).convert("RGB")
        W, H  = image.size
        draw_img   = image.copy()
        all_boxes, all_masks, all_points = [], [], []
        not_match = False

        for prompt, label in zip (text_prompts, text_label):
            if label == "":
                not_match = True

            if if_translate:  # 翻译成德语
                tokens_en = prompt_tokens(label) 
                tokens_de = [self._en_word2de(w).lower() for w in tokens_en]
                p_tokens = tokens_de         # 整句 prompt→单词列表
                print(f'Prompt EN: "{label}"   →   DE: "{tokens_de}"')
            else:
                p_tokens = prompt_tokens(label)
                print(f'Prompt: "{label}"   →   Tokens: {p_tokens}')
            
            inputs = self.owl_processor(text=[prompt], images=image,
                                        return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.owl_model(**inputs)
            boxes  = out.pred_boxes[0].cpu()
            scores = out.logits[0].cpu().sigmoid()[:, 0]
            keep   = torch.topk(scores, k=min(5, len(scores))).indices  # 前 5 个候选

            matched = []   # OCR 命中的候选
            for idx in keep:
                if scores[idx] < 0.1:    # 分数下限
                    continue
                if not_match:            # 无匹配任务，直接用最高分的框
                    box = self._box_xyxy(boxes[idx], W, H)
                    matched.append((box, scores[idx].item()))
                else:
                    box  = self._box_xyxy(boxes[idx], W, H)
                    x1,y1,x2,y2 = box
                    pad = 20                  # 扩张 20 px 免得截掉文字
                    crop = image.crop((max(0,x1-pad), max(0,y1-pad),
                                    min(W,x2+pad), min(H,y2+pad)))
                    ocr_tokens = [t.lower() for t in self.reader.readtext(np.array(crop),
                                                                        detail=0)]
                    exist , matched_score = prompt_match(p_tokens, ocr_tokens)
                    if exist:   # 命中一次即可
                        total_score = matched_score + scores[idx].item()  # 命中分数 + 检测分数
                        print(matched_score, scores[idx].item(), total_score)
                        matched.append((box, total_score))
                
            
            if len(matched)>1 and not multi_task:
                # 多任务模式下，允许多个匹配；否则只保留最高分的一个
                matched = sorted(matched, key=lambda x: x[1], reverse=True)[:1]

            # 若有命中用命中，否则用最高置信度那一框兜底
            cand = matched if matched else [(self._box_xyxy(boxes[keep[0]], W, H),
                                             scores[keep[0]].item())]

            for box, conf in cand:
                
                if if_sam:
                    masks_sam = self.sam.segment_with_boxes(image_path, [box],
                                                            multimask_output=False)
                    mask = masks_sam[0].astype(np.uint8)             # 取第 1 个框的单一 mask
                else:
                    mask = self._fastsam_seg(image_path, box)

                bcen, mcen = self._centers(box, mask)
                all_boxes.append((box, prompt, conf))
                all_masks.append(mask)
                all_points.append({"target": prompt,
                                   "box_center_point": bcen,
                                   "seg_center_point": mcen})

        result = self._visual(draw_img, all_boxes, all_masks)
        return result, all_boxes, all_points


    # ---------- 工具 ----------
    @staticmethod
    def _box_xyxy(box, W, H):
        cx, cy, bw, bh = box.numpy()
        return (int(max(0,(cx-bw/2)*W)), int(max(0,(cy-bh/2)*H)),
                int(min(W,(cx+bw/2)*W)), int(min(H,(cy+bh/2)*H)))

    def _centers(self, box, mask):
        x1,y1,x2,y2 = box
        cx, cy = (x1+x2)//2, (y1+y2)//2
        ys, xs = np.where(mask>0)
        return (cx,cy), (None,None) if len(xs)==0 else (int(xs.mean()),int(ys.mean()))

    def _fastsam_seg(self, img_path, box):
        image = Image.open(img_path).convert("RGB")
        W,H   = image.size
        x1,y1,x2,y2 = map(int, box)
        crop  = np.array(image.crop((x1,y1,x2,y2)))
        res   = self.fastsam.predict(crop, conf=self.fastsam_conf, iou=self.fastsam_iou,
                                     imgsz=self.fastsam_size, device=self.device, verbose=False)
        best,marea = None,-1
        for r in res:
            if r.masks is None: continue
            for m in r.masks.data:
                arr = m.cpu().numpy().astype(np.uint8)
                area = arr.sum()
                if area>marea: best,marea = arr,area
        final = np.zeros((H,W),dtype=np.uint8)
        if best is not None:
            if best.shape!=(y2-y1,x2-x1):
                best = cv2.resize(best,(x2-x1,y2-y1),interpolation=cv2.INTER_NEAREST)
            final[y1:y2,x1:x2] = best
        return final

    # ---------- 可视化 ----------
    def _visual(self, img, boxes, masks):
        draw = ImageDraw.Draw(img)
        overlay = Image.new("RGBA", img.size, (0,0,0,0))
        for (box,label,score),mask in zip(boxes,masks):
            if label not in self.color_map:
                self.color_map[label] = tuple(np.random.randint(64,256,size=3))
            col = self.color_map[label]
            draw.rectangle(box, outline=col, width=3)
            txt = f"{label} ({score:.2f})"
            try: font = ImageFont.truetype("arial.ttf", 20)
            except: font = ImageFont.load_default()
            draw.text((box[0], box[1]-25), txt, fill=col, font=font)
            if mask.any():
                rgba = np.zeros((*mask.shape,4),dtype=np.uint8)
                rgba[mask>0] = (*col,200)
                overlay = Image.alpha_composite(overlay, Image.fromarray(rgba))
        return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

def find_object_central_pixel(target: str, text: str, image_path, is_sam: bool = True, if_translate: bool = False):
    # Verwende forward slashes (Linux) und robuste Lade-Logik in TextDrivenSegmenter
    seg = TextDrivenSegmenter(fastsam_model_path="src/VLM_agent/FastSAM/FastSAM-x.pt")
    img, boxes, points = seg.detect_and_segment(image_path, [target], [text],  multi_task = False, if_sam = is_sam, if_translate = if_translate)
    img.save("result.jpg")
    for (b,l,s),pt in zip(boxes, points):
        print(f"{l} @ {b}  conf={s:.2f}")
        print("   centers:", pt)  

    target_prompt = points[0]["target"]
    box_center_point = points[0]["box_center_point"]
    seg_center_point = points[0]["seg_center_point"]
    bbox = tuple(boxes[0][0])  # 获取第一个目标的边界框
    score = boxes[0][2]  # 获取第一个目标的置信度分数

    return target_prompt, box_center_point, seg_center_point, bbox, score   


# ---------------- demo ----------------
if __name__ == "__main__":
    target_label = "pepper bottle"  
    text = "Schwarzer Pfeffer ganz"
    # text = ""
    target_prompt, box_center_point, seg_center_point, bbox, score = find_object_central_pixel(target_label, text, is_sam = True, if_translate = False)  # 调用函数处理图像中的目标检测
    print(f"🔍 Detected target: {target_label}")
    print(f"📍 Target prompt: {target_prompt}")
    print(f"📏 Bounding box: {bbox}")
    print(f"🎯 Box center point: {box_center_point}")
    print(f"🎯 Segmentation center point: {seg_center_point}")
    print(f"📊 Detection score: {score}")




# scone
