import argparse
import json
import os
from pathlib import Path
import cv2
from ultralytics import YOLO

def run_inference(model_path, image_path, output_dir="output", conf_threshold=0.25):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    print(f"Loading model {model_path}...")
    model = YOLO(model_path)
    
    print(f"Running inference on {image_path.name}...")
    
    # ── Run prediction ────────────────────────────────────────────────────────
    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,  # Minimum confidence threshold to consider a detection.
                              # Detections below this value are discarded. Range: 0.0–1.0. 
        iou=0.7,              # IoU threshold for Non-Maximum Suppression (NMS).
                              # Overlapping boxes above this threshold are merged.
        imgsz=640,            # Resize input to this size before inference.
                              # Should match the size used during training.
        max_det=300,          # Maximum number of detections per image.
        device="cpu",         # Inference device. Use "cpu", 0, [0,1], or "mps". -> will use default environment
        verbose=False,        # Suppress per-image console output.
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
                "x1": round(x1, 2), "y1": round(y1, 2),
                "x2": round(x2, 2), "y2": round(y2, 2),
                "width": round(x2 - x1, 2), "height": round(y2 - y1, 2)
            }
        })
        
    json_path = output_dir / f"{image_path.stem}.json"
    with open(json_path, "w") as f:
        json.dump(detections, f, indent=2)
        
    # ── Save annotated image ──────────────────────────────────────────────────
    # result.plot() returns a BGR numpy array with bounding boxes drawn on it
    annotated_image = result.plot(
        conf=True,      # Show confidence score on each label
        labels=True,    # Show class name on each box
        boxes=True,     # Draw bounding boxes
        line_width=2,   # Box border thickness in pixels
    )
    output_image_path = output_dir / image_path.name
    cv2.imwrite(str(output_image_path), annotated_image)
    
    print(f"Found {len(detections)} detection(s)")
    print(f"JSON saved to {json_path}")
    print(f"Annotated image saved to {output_image_path}")
    
    return output_image_path, json_path, detections

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO inference on a single image")
    parser.add_argument("--model", type=str, required=True, help="Path to model (.pt)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    run_inference(args.model, args.image, args.output, args.conf)