# Model Card — Wheel Components Inspector - Model 1 (v1.0)

## 1. Overview
- **Goal:** Detect and localize wheel components in images for automotive visual inspection.
- **Task:** Object Detection
- **Classes:** `jante`, `parafuso`, `roda`
- **Intended users:** Automotive workshop technicians, quality assurance auditors in assembly lines, and automated automotive inspection systems.

## 2. Intended Use and Scope
- **Intended environment:** Automotive workshops, garages, outdoor parking lots, assembly lines.
- **Assumptions:** Images captured from cameras at 0.5-3m distance, daylight or indoor lighting, vehicle stationary. Images should ideally be taken frontally with the wheel clearly visible.
- **Out-of-scope uses:** Isolated disassembled tires (without rims), internal hub components (bearings, brake discs), motorcycle/bicycle/heavy vehicle wheels, and damage/deformation detection on parts.

## 3. Training Data
- **Data source:** Images collected by the project group and colleagues using various smartphones (iPhone, Samsung, Xiaomi, Huawei) across diverse locations (workshops, garages, parking lots, outdoors). Captures incorporate natural and artificial lighting under multiple angles (frontal, lateral, diagonal, top-down) at distances spanning 0.5 to 3 meters.
- **Dataset size:** 200 base images collected (initial split train: 70%, val: 20%, test: 10%); after training-exclusive augmentations, it results in an effective dataset of 480 images (Split: 87.5% train / 8.3% val / 4.2% test).
- **Class distribution:** Total of annotated instances:
  - `jante`: 200 instances
  - `parafuso`: 730 instances
  - `roda`: 200 instances
- **Labeling guidelines:** Only fully visible objects were annotated; bolts not completely visible were discarded. When mounted, both `roda` and `jante` are boxed independently, with the rim contained within the wheel. `roda` bounding boxes extend to the outer limit of the visible tire; `jante` boxes extend to the outer limit of the rim, excluding the tire. Low contrast bolts were annotated only if clearly identifiable.
- **Augmentations:** 
  - *In Roboflow (Static before training):* Horizontal flip (50%), Crop/zoom (0-15%), Rotation (±15°), Saturation (±20%), Brightness (-25% to +25%).
  - *In YOLOv8 (Dynamic per epoch):* HSV shift (H: 0.015, S: 0.7, V: 0.4), Horizontal flip (50%), Random scale (±50%), Translation (10%), Mosaic (p=1.0).

## 4. Evaluation
- **Test set description:** 20 images (accounting for 4.2% of the base dataset), explicitly held out and not used during training or validation.
- **Metrics:** 
| Metric | Overall | Jante | Parafuso | Roda |
|--------|---------|-------|----------|------|
| mAP@0.5 | 0.8564 | 0.9950 | 0.5793 | 0.9950 |
| mAP@0.5:0.95 | 0.7116 | 0.9683 | 0.2282 | 0.9382 |
| Precision | 0.8616 | 0.9859 | 0.6413 | 0.9575 |
| Recall | 0.8939 | 1.0000 | 0.6816 | 1.0000 |

- **Qualitative analysis:** `roda` and `jante` achieve near-perfect values across all metrics due to their large size and consistent presence. However, for `parafuso`, the ~64% Precision and low ~23% mAP@0.5-95 reflect inherent difficulties with contrast, extremely small sizes, and severe perspective angles as analyzed in the report.
- **Recommended confidence threshold(s):** min 0.5 for generally good predictions.

## 5. Limitations and Failure Modes
- **Undetected bolts due to low contrast:** Bolts positioned against dark rims or under dark shadows fail to be detected due to the lack of clear edges.
- **Failures on inclined angles:** Capture angles far from the frontal perspective distort the geometry of the rim, severely reducing the visible area of the bolts.

## 6. Deployment Notes
- **Input requirements:** RGB image, any resolution (resized internally to 640x640).
- **Output format:** JSON array of detections with `class_name`, `confidence`, and `bbox` (xyxy format).
- **Compute:** Characteristics tested during the project's development:
  - Model architecture: YOLOv8m (Medium)
  - Model size: ≈ 51 MB (.pt file)
  - Inference latency: ≈ 50-200 ms per image 
  - Validated hardware: AMD Ryzen 7 5700X CPU, NVIDIA GeForce RTX 3060 TI (8GB VRAM), 16GB DDR4 RAM.

## 7. Ethical / Safety / Privacy Considerations
- No personally identifiable information (faces, license plates) is intentionally included in the dataset; any inadvertent captures were blurred manually.
- The model should not be used as the sole safety mechanism in critical automotive systems.

## 8. Versioning and Contact
- **Version:** v1.0
- **Date:** 2026-04
- **Authors:** Jose Barbosa, Pedro Rocha, Agostinho Ferreira