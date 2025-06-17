# pixel masks to be compared 

import numpy as np
from PIL import Image 
import xml.etree.ElementTree as ET 
import os 

def load_mask_as_binary(mask_path):
    # convert SLIME mask PNG into a binary mask

    img = Image.open(mask_path).convert("L")
    mask = np.array(img)
    binary_mask = (mask > 0).astype(np.uint8)
    return binary_mask 

# to parse the ground truth images and get bounding boxes
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

# get binary mask from bounding boxes
def create_mask_from_boxes(boxes, img_size):
    """
    Create binary mask from bounding boxes.
    """
    mask = Image.new("L", img_size, 0)
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        draw.rectangle(box, outline=1, fill=1)
    return np.array(mask).astype(np.uint8)

def compute_mask_iou(mask1, mask2):
    # compute iou between two binary masks

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0: 
        return 0.0
    
    return intersection/union 


def evaluate_iou(pred_path, gt_xml_path, img_size):
    """
    Evaluate IoU between SLiMe mask and GT bounding box mask.
    Returns: avg_iou, TP, FP, FN
    """
    pred_mask = load_mask_as_binary(pred_path) if pred_path and os.path.exists(pred_path) else None
    gt_boxes = parse_xml(gt_xml_path) if gt_xml_path and os.path.exists(gt_xml_path) else []

    gt_mask = create_mask_from_boxes(gt_boxes, img_size) if gt_boxes else None

    if pred_mask is None and gt_mask is None:
        # Both missing → correctly predicted nothing
        return 1.0, 0, 0, 0

    if gt_mask is None:
        # No GT, but prediction exists → false positive
        has_fp = int(pred_mask.sum() > 0)
        return 0.0, 0, has_fp, 0

    if pred_mask is None:
        # GT exists, no prediction → false negative
        has_fn = int(gt_mask.sum() > 0)
        return 0.0, 0, 0, has_fn

    # Both exist → compute IoU
    iou = compute_mask_iou(pred_mask, gt_mask)

    TP = int(iou > 0)
    FP = int(iou == 0 and pred_mask.sum() > 0)
    FN = int(iou == 0 and gt_mask.sum() > 0)

    return iou, TP, FP, FN