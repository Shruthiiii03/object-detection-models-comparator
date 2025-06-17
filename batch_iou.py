import os 
# from compute_iou import evaluate_iou, parse_json, parse_xml
from compute_SLIME_iou import evaluate_iou
from PIL import Image
import json

#setting up paths 
XML_DIR = "ground_truth"
# JSON_DIR = "gemini_predictions/predictions_temp0.0_topP0.4_topK4"
IMG_DIR = "images"
SLIME_PRED_DIR = "SLIME_predictions"

#get all filenames 
# xml_filenames = set(f.split('.')[0] for f in os.listdir(XML_DIR) if f.endswith(".xml"))
# json_filenames = set(f.split('.')[0] for f in os.listdir(JSON_DIR) if f.endswith(".json"))
all_filenames = set(f.split('.')[0] for f in os.listdir(IMG_DIR) if f.endswith(".png"))

all_iou_sum = 0.0
all_iou_avg = 0.0
count = 0 

total_tp = 0
total_fp = 0
total_fn = 0

#build filepaths

# for SLIME
for base_name in all_filenames:
    xml_path = os.path.join(XML_DIR, f"{base_name}.xml")
    # json_path = os.path.join(JSON_DIR, f"{base_name}.json")
    pred_path = os.path.join(SLIME_PRED_DIR, f"{base_name}.png")
    image_path = os.path.join(IMG_DIR, f"{base_name}.png") 

    if not os.path.exists(image_path):
        print(f"{base_name}: PNG image not found — skipping")
        continue  

    img = Image.open(image_path)
    img_size = img.size

    xml_exists = os.path.exists(xml_path)
    # json_exists = os.path.exists(json_path)
    pred_exists = os.path.exists(pred_path)

    if not xml_exists and not pred_exists:
        avg_iou = 1.0
        TP = FP = FN = 0
        print(f"{base_name}: Correctly predicted nothing")

    elif not xml_exists and pred_exists:
        pred_mask = Image.open(pred_path).convert("L")
        has_fp = int((np.array(pred_mask) > 0).sum() > 0)
        avg_iou = 0.0
        TP = 0
        FP = has_fp
        FN = 0
        print(f"{base_name}: No GT, {has_fp} false positives")

    elif xml_exists and not pred_exists:
        from compute_SLIME_iou import parse_xml
        gt_boxes = parse_xml(xml_path)
        avg_iou = 0.0
        TP = 0
        FP = 0
        FN = len(gt_boxes)
        print(f"{base_name}: No predictions, {FN} false negatives")

    else:
        avg_iou, TP, FP, FN = evaluate_iou(pred_path, xml_path, img_size)
        print(f"{base_name}: Average IoU = {avg_iou:.4f}")

    total_tp += TP
    total_fp += FP
    total_fn += FN

    print(f"{base_name}: Average IoU = {avg_iou:.4f}")
    all_iou_sum += avg_iou
    count += 1

#dino and gemini
    # if not xml_exists and not json_exists:
    #     avg_iou = 1.0 
    #     print("correctly predicted nothing")
    # # 
    # elif not xml_exists and json_exists: 
    #     img = Image.open(image_path)
    #     width, height = img.size
    #     pred_boxes = parse_json(json_path, width, height)
    #     total_fp += len(pred_boxes)
    #     # avg_iou = 0.0
    #     print(f"{base_name}: ❌ No GT, {len(pred_boxes)} false positives")

    # elif xml_exists and not json_exists: 
    #     gt_boxes = parse_xml(xml_path)
    #     total_fn += len(gt_boxes)
    #     # avg_iou = 0.0
    #     print(f"{base_name}: ❌ No predictions, {len(gt_boxes)} false negatives")
    
    # else: 
    #     # just find the IOU normally 
    #     avg_iou, TP, FP, FN = evaluate_iou(xml_path, json_path, image_path)
    #     total_tp += TP
    #     total_fp += FP
    #     total_fn += FN

    #     print(f"{base_name}: ✅ Average IoU = {avg_iou:.4f}")
    #     all_iou_sum += avg_iou
    #     count += 1

precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

all_iou_avg = all_iou_sum / count if count > 0 else 0
print(f"\n Overall Average IoU (on {count} images): {all_iou_avg:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")