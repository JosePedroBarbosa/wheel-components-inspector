# Model Card — Wheel Components Inspector - Model 2 (v2.0)

## 1. Overview
- **Goal:** Detect and localize wheel components in images for automotive visual inspection, with a specific focus on more precision bolt (`parafuso`) detection.
- **Task:** Object Detection
- **Classes:** `jante`, `parafuso`, `roda`
- **Intended users:** Automotive workshop technicians, quality assurance auditors in assembly lines, and automated automotive inspection systems.

## 2. Intended Use and Scope
- **Intended environment:** Automotive workshops, garages, outdoor parking lots, assembly lines.
- **Assumptions:** Images captured from cameras at 0.5-3m distance, daylight or indoor lighting, vehicle stationary. Images should ideally be taken frontally with the wheel clearly visible.
- **Out-of-scope uses:** Isolated disassembled tires (without rims), internal hub components (bearings, brake discs), motorcycle/bicycle/heavy vehicle wheels, and damage/deformation detection on parts.

## 3. Training Data
- **Data source:** Images collected by the project group and colleagues using various smartphones (iPhone, Samsung, Xiaomi, Huawei) across diverse locations (workshops, garages, parking lots, outdoors) covering multiple angles and lighting conditions. For this model, the base dataset was intentionally supplemented with additional close-up, high-quality images focusing specifically on bolts.
- **Dataset size:** 200 base images collected (initial split train: 70%, val: 20%, test: 10%); after training-exclusive augmentations, it results in an effective dataset of 480 images (Split: 87.5% train / 8.3% val / 4.2% test).
- **Class distribution:** Increase in high-quality `parafuso` instances.
  - `jante`: > 200 instances
  - `parafuso`: 755 instances
  - `roda`: > 200 instances
- **Labeling guidelines:** Only fully visible objects were annotated; bolts not perfectly visible were initially discarded, but re-evaluated with new detailed data to mitigate False Negatives. `roda` bounding boxes encompass the outer limit of the visible tire, while `jante` bounds strictly the rim. In composite shots, the rim box is contained within the wheel box.
- **Augmentations:**
  - *In Roboflow (Static before training):* Horizontal flip (50%), Crop/zoom (0-15%), Rotation (±15°), Saturation (±20%), Brightness (-25% to +25%).
  - *In YOLOv8 (Dynamic per epoch):* HSV shift (H: 0.015, S: 0.7, V: 0.4), Horizontal flip (50%), Random scale (±50%), Translation (10%), Mosaic (p=1.0).

## 4. Evaluation
- **Test set description:** ~20 test images from the baseline network (4.2%) supplemented with more images focusing on better bolt detection.
- **Metrics:**
| Metric | Overall | Jante | Parafuso | Roda |
|--------|---------|-------|----------|------|
| mAP@0.5 | 0.8878 | 0.9950 | 0.6734 | 0.9950 |
| mAP@0.5:0.95 | 0.7207 | 0.9821 | 0.2293 | 0.9507 |
| Precision | 0.9113 | 0.9867 | 0.7604 | 0.9868 |
| Recall | 0.9041 | 1.0000 | 0.7122 | 1.0000 |

- **Qualitative analysis:** The integration of photographs with varied angles (including lateral and top-down near the rim) and new low-contrast scenarios resulted in a considerably lower number of false negatives for `parafuso`. More robust detection in adverse conditions that previously caused the failures listed in the baseline model.
- **Recommended confidence threshold(s):** min 0.5 for generally good predictions.

## 5. Limitations and Failure Modes
- **Undetected bolts due to low contrast:** Bolts positioned against dark rims or under dark shadows might still fail to be detected if the boundaries blur entirely.
- **Failures on inclined angles:** Extreme capture angles far from the frontal perspective distort the geometry, diminishing the object size and making bolt detection harder.

## 6. Deployment Notes
- **Input requirements:** RGB image, any resolution (resized internally to 640x640).
- **Output format:** JSON array of detections with `class_name`, `confidence`, and `bbox` (xyxy format).
- **Compute:**
  - Model architecture: YOLOv8m (Medium)
  - Model size: ≈ 51 MB (.pt file)
  - Inference latency: ≈ 50-200 ms per image 
  - Validated hardware: AMD Ryzen 7 5700X CPU, NVIDIA GeForce RTX 3060 TI (8GB VRAM), 16GB DDR4 RAM.

## 7. Ethical / Safety / Privacy Considerations
- No personally identifiable information (faces, license plates) is intentionally included in the dataset; any inadvertent captures were blurred manually.
- The model should not be used as the sole safety mechanism in critical automotive systems.

## 8. Versioning and Contact
- **Version:** v2.0
- **Date:** 2026-04
- **Authors:** Jose Barbosa, Pedro Rocha, Agostinho Ferreira