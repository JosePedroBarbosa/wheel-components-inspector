"""
Inference module for Wheel Components Inspector.

Run YOLOv8 inference on a single image and produce:
  - A JSON file with structured detections
  - An annotated image with bounding boxes
"""

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def run_inference(
    model_path: str,
    image_path: str,
    output_dir: str = "output",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    device: str = "cpu",
) -> tuple[Path, Path, list[dict]]:
    """Run object detection on a single image.

    Returns:
        Tuple of (annotated_image_path, json_path, detections_list).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format: {image_path.suffix}")

    model = YOLO(model_path)

    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
        max_det=300,
        device=device,
        verbose=False,
    )

    result = results[0]
    detections = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class_id": int(box.cls),
            "class_name": model.names[int(box.cls)],
            "confidence": round(float(box.conf), 4),
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "width": round(x2 - x1, 2),
                "height": round(y2 - y1, 2),
            },
        })

    json_path = output_dir / f"{image_path.stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2, ensure_ascii=False)

    annotated = result.plot(conf=True, labels=True, boxes=True, line_width=2)
    output_image_path = output_dir / image_path.name
    cv2.imwrite(str(output_image_path), annotated)

    print(f"Detections: {len(detections)}")
    print(f"JSON:       {json_path}")
    print(f"Image:      {output_image_path}")

    return output_image_path, json_path, detections


def main():
    parser = argparse.ArgumentParser(description="YOLO inference on a single image")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu', '0', 'mps'")

    args = parser.parse_args()
    run_inference(args.model, args.image, args.output, args.conf, args.iou, args.device)


if __name__ == "__main__":
    main()
