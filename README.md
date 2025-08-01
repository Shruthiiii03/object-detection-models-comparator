# Object Detection Models Comparison: IoU & Evaluation Toolkit

This repository provides a lightweight tool to evaluate and compare the performance of object detection models (e.g., Gemini, DINO, SLiMe, TREX) by computing **Intersection over Union (IoU)**, **precision**, **recall**, and **F1-score** across image predictions and ground truth data.

---

## Features

- **IoU Evaluation**: Computes IoU between predicted and ground-truth bounding boxes or masks.
- **Precision, Recall, F1**: Includes logic to compute classification metrics across dataset.
- **Greedy Matching**: One-to-one matching between predicted and GT boxes using max IoU.
- **SLIME Mask Handling**: Supports binary mask comparison for segmentation predictions.
- **Modular**: Can swap models by changing directory paths or enabling relevant lines.

---

## How It Works

### 1. For Bounding Box Models (Gemini, DINO, TREX)
- Ground truth annotations are in `VOC XML` format. Use the tool - LabelImg to manually annotate images and extract the annotations in the XML format. 
- Model predictions are stored as JSON files containing bounding boxes.
- `compute_iou.py` handles:
  - Parsing XML and JSON files
  - Matching predictions to GT boxes greedily
  - Computing per-image and overall IoU + metrics

### 2. For SLIME (Segmentation Masks)
- SLIME predictions are stored as binary PNG masks.
- `compute_SLIME_iou.py`:
  - Converts GT boxes to a binary mask
  - Compares with predicted SLIME mask
  - Computes pixel-wise IoU

---

## Quick Start

### 1. Clone the Repository

### 2. Run IoU Evaluation for Bounding Box Models

Update batch_iou.py with the appropriate prediction directory (e.g., gemini_predictions, TREX_predictions).

Then run:

```
python batch_iou.py
```

### 3. Run IoU Evaluation for SLIME

Uncomment the SLIME evaluation block in batch_iou.py and ensure prediction masks are inside SLIME_predictions/.

```
python batch_iou.py
```

## Notes

Predictions and images are expected to have matching filenames (e.g., image1.png, image1.json, image1.xml).