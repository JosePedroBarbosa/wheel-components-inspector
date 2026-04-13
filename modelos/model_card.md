# Model Card — Wheel Components Inspector (v1.0)

## 1. Overview
- **Goal:** Detect and localize wheel components in images for automotive visual inspection.
- **Task:** Object Detection
- **Classes:** `jante` (rim), `parafuso` (bolt), `roda` (wheel)
- **Intended users:** Quality assurance engineers, automotive workshop technicians, AI course evaluators.

## 2. Intended Use and Scope
- **Intended environment:** Automotive workshops, garages, outdoor parking lots, assembly lines.
- **Assumptions:** Images captured from smartphone cameras at 0.5-3m distance, daylight or indoor lighting, vehicle stationary.
- **Out-of-scope uses:** Damage detection, tire brand classification, bolt torque estimation, night-time operation without adequate lighting.

## 3. Training Data
- **Data source:** Images collected by the project group using smartphone cameras.
- **Dataset size:** ~200 images total (70% train / 20% val / 10% test).
- **Class distribution:** <!-- Fill after final dataset export -->
  - `jante`: _N_ instances
  - `parafuso`: _N_ instances
  - `roda`: _N_ instances
- **Labeling guidelines:** Bounding boxes cover visible extent. Partially occluded objects (>30% visible) are labeled. Bolts occluded >70% are excluded.
- **Augmentations:** HSV shift, horizontal flip (50%), scale (50%), mosaic augmentation applied during training. Roboflow preprocessing: auto-orientation, resize to 640x640.

## 4. Evaluation
- **Test set description:** ~20 images held out from the dataset, not seen during training or validation.
- **Metrics:** <!-- Fill after training -->

| Metric | Overall | Jante | Parafuso | Roda |
|--------|---------|-------|----------|------|
| mAP@0.5 | — | — | — | — |
| mAP@0.5:0.95 | — | — | — | — |
| Precision | — | — | — | — |
| Recall | — | — | — | — |

- **Qualitative analysis:** <!-- Add example successes and failures after training -->
- **Recommended confidence threshold:** 0.25 (default); adjust to 0.35 for higher precision.

## 5. Limitations and Failure Modes
- Reflective/chrome rims with direct glare may cause missed detections.
- Small bolts at distance (<30px) are harder to detect.
- Unusual rim designs (e.g., racing rims) may be under-represented in training data.
- Dark bolts on dark rims may produce false negatives.
- Not validated for nighttime or low-light conditions.

## 6. Deployment Notes
- **Input requirements:** RGB image, any resolution (resized internally to 640x640).
- **Output format:** JSON array of detections with `class_name`, `confidence`, and `bbox` (xyxy format).
- **Compute:** <!-- Fill after training -->
  - Model size: ~50 MB (YOLOv8m)
  - CPU inference: ~200-400 ms per image
  - GPU inference: ~10-30 ms per image

## 7. Ethical / Safety / Privacy Considerations
- No personally identifiable information (faces, license plates) is intentionally included in the dataset.
- Images that inadvertently captured license plates had them blurred before inclusion.
- The model should not be used as the sole safety mechanism in critical automotive systems.

## 8. Versioning and Contact
- **Version:** v1.0
- **Date:** 2026-04
- **Authors:** Jose Barbosa, [other group members]
- **Course:** Inteligencia Artificial, Engenharia Informatica, ESTG - P.Porto