#IOU script for gemini, DINO

import json 
import xml.etree.ElementTree as ET 
import os 
import numpy as np 
from PIL import Image, ImageDraw
# from scipy.optimize import linear_sum_assignment 

# SPECIFIC objects
# base_filename = "ch01"
# xml_path = f"ground_truth/{base_filename}.xml"
# json_path = f"gemini_predictions/{base_filename}.json"

#image_path = f"images/{base_filename}.jpg"  # or .png depending on your format
#img = Image.open(image_path)
#width, height = img.size
#print(f"Image dimensions: {width}x{height}")

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot() 

    boxes = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])

    return boxes

#gemini outputs the transposed version of coordinates
def parse_json(json_path, width, height):
    with open(json_path, 'r') as f: 
        data = json.load(f)

    gemini_width = 1000
    gemini_height = 1000
    
    scaled_coords = []
    for obj in data: 
        box = obj["box_2d"]
    
        # Skip boxes that don't have exactly 4 values
        if len(box) != 4:
            print(f"Skipping invalid box: {box}")
            continue

        y1, x1, y2, x2 = box

        x1 = x1 * width/gemini_width
        y1 = y1 * height/gemini_height
        x2 = x2 * width/gemini_width
        y2 = y2 * height/gemini_height
        scaled_coords.append([x1, y1, x2, y2])

    return scaled_coords

#parse json without transposing for dino 
# def parse_json(json_path, width, height):
#     with open(json_path, 'r') as f: 
#         data = json.load(f)
    
#     boxes = []
#     for obj in data:
#         box = obj["box"]
#         if len(box) == 4:
#             boxes.append(box)

#     return boxes

def compute_iou(box1,box2):
    x_left = max(box1[0], box2[0])
    y_left = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_right = min(box1[3], box2[3])

    width = max(0, x_right - x_left)
    height = max(0, y_right - y_left)
    area_intersection = width * height

    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    area_union = area_box1 + area_box2 - area_intersection

    if area_union == 0: 
        return 0.0 # to deal with undefined with 0 in denominator
    else:
        return area_intersection / area_union

        
def evaluate_iou(xml_path, json_path, image_path):
    img = Image.open(image_path)
    width, height = img.size

    gt_boxes = parse_xml(xml_path) if os.path.exists(xml_path) else []
    pred_boxes = parse_json(json_path, width, height) if os.path.exists(json_path) else []

    HIGH_IOUs = []
    used_preds = set()

    # 1.Greedy approach
    for gt in gt_boxes:
        best_iou = 0
        best_pred_idx = -1
        for idx, pred in enumerate(pred_boxes):
            if idx in used_preds:
                continue
            iou = compute_iou(gt, pred)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = idx
        if best_pred_idx != -1:
            HIGH_IOUs.append(best_iou)
            used_preds.add(best_pred_idx)

    # 2. Penalty for FP 
    # PENALTY_IOUs = []
    # IOU_THRESHOLD = 0.1 

    # for idx, pred in enumerate(pred_boxes):
    #     if idx in used_preds:
    #         continue # alrdy matched
        
    #     overlaps = [compute_iou(gt, pred) for gt in gt_boxes]
    #     max_overlap = max(overlaps) if overlaps else 0.0

    #     if max_overlap < IOU_THRESHOLD:
    #         PENALTY_IOUs.append(0.0)

    # total_ious = HIGH_IOUs + PENALTY_IOUs
    total_ious = HIGH_IOUs
    avg_iou = sum(total_ious) / len(total_ious) if total_ious else 0.0

    TP = len(used_preds)
    FN = len(gt_boxes) - TP # actual boxes, but model didnt pick up
    FP = len(pred_boxes) - TP 

    return avg_iou, TP, FP, FN 
