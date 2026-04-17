"""
Training pipeline for Wheel Components Inspector.

Downloads the dataset from Roboflow and trains a YOLOv8 model
for detecting wheels (roda), rims (jante), and bolts (parafuso).
"""

import os
import json
import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv
import roboflow
from ultralytics import YOLO

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
RUNS_DIR = ROOT_DIR / "src" / "runs" / "train"

TRAIN_DEFAULTS = {
    # Full list of available pre-trained models: https://docs.ultralytics.com/models/yolov8/#performance-metrics
    "model_variant": "yolov8m.pt",         # Load the pre-trained model weights

    # ── Core training settings ────────────────────────────────────────────────
    "epochs": 100,      # Total number of training epochs.
                        # More epochs = more learning, but risk of overfitting.
                        # Typical range: 50–300.

    "patience": 50,     # Early stopping: halt training if no improvement is seen
                        # for this many epochs. Set to 0 to disable.

    "batch": 16,        # Number of images per training batch.
                        # Higher = faster but needs more VRAM. Use -1 for AutoBatch.

    "imgsz": 640,       # Input image size (pixels). Images are resized to this square.
                        # Common values: 416, 512, 640, 1280.

    # ── Hardware ──────────────────────────────────────────────────────────────
    "workers": 8,       # Number of DataLoader worker threads for loading images.
                        # Reduce if you see memory errors or CPU bottlenecks.

    # ── Optimiser ─────────────────────────────────────────────────────────────
    "optimizer": "auto", # Optimiser algorithm. Options:
                         # "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"
                         # "auto" selects SGD or AdamW based on the model.

    "lr0": 0.01,        # Initial learning rate.
                        # Lower values (e.g. 0.001) are safer for fine-tuning.

    "lrf": 0.01,        # Final learning rate as a fraction of lr0.
                        # The scheduler decays lr0 → lr0 * lrf over training.

    "momentum": 0.937,  # SGD momentum / Adam beta1.
                        # Controls how much past gradients influence the update.

    "weight_decay": 0.0005, # L2 regularisation penalty — discourages large weights
                            # and helps prevent overfitting.

    "warmup_epochs": 3.0,   # Number of epochs for learning-rate warm-up at the start.
                            # LR gradually rises from 0 to lr0 during this phase.

    "warmup_momentum": 0.8, # Initial momentum during the warm-up phase.
    "warmup_bias_lr": 0.1,  # Initial learning rate for bias parameters during warm-up.

    # ── Loss weights ──────────────────────────────────────────────────────────
    "box": 7.5,         # Weight for the bounding-box regression loss.
                        # Increase to make the model focus more on box accuracy.

    "cls": 0.5,         # Weight for the classification loss.
    "dfl": 1.5,         # Weight for the Distribution Focal Loss (box refinement).

    # ── Augmentation ──────────────────────────────────────────────────────────
    "hsv_h": 0.015,     # Random hue shift (fraction of 360°). Adds colour variety.
    "hsv_s": 0.7,       # Random saturation shift. Range: 0.0–1.0.
    "hsv_v": 0.4,       # Random brightness (value) shift. Range: 0.0–1.0.

    "degrees": 0.0,     # Random rotation range in degrees (e.g. 10 → ±10°).
    "translate": 0.1,   # Random translation as a fraction of image size (e.g. 0.1 = 10%).
    "scale": 0.5,       # Random scale factor (e.g. 0.5 means zoom between 50%–150%).
    "shear": 0.0,       # Random shear angle in degrees.
    "perspective": 0.0, # Random perspective distortion. Range: 0.0–0.001.
    "flipud": 0.0,      # Probability of vertical flip. 0.0 = disabled.
    "fliplr": 0.5,      # Probability of horizontal flip. 0.5 = 50% chance per image.

    "mosaic": 1.0,      # Probability of Mosaic augmentation (combines 4 images).
                        # Very effective for small object detection. 0.0 = disabled.

    "mixup": 0.0,       # Probability of MixUp augmentation (blends 2 images).
                        # Useful for improving generalisation.

    "copy_paste": 0.0,  # Probability of Copy-Paste augmentation (pastes objects
                        # from one image onto another). Good for instance segmentation.

    # ── Reproducibility ───────────────────────────────────────────────────────
    "seed": 0,          # Random seed for reproducibility. Use the same seed to get
                        # identical results across runs.
}

def download_dataset(workspace_name: str, project_name: str, version_num: int) -> str:
    """Download a YOLOv8 dataset from Roboflow into ``dataset/``."""
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY not found in .env file.")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dataset target: {DATASET_DIR}")

    rf = roboflow.Roboflow(api_key=api_key)
    project = rf.workspace(workspace_name).project(project_name)
    dataset = project.version(version_num).download(
        "yolov8", location=str(DATASET_DIR), overwrite=True
    )

    print(f"Dataset downloaded to: {dataset.location}")
    return dataset.location


def export_train_config(args: argparse.Namespace, data_path: str) -> Path:
    """Save a reproducible ``train_config.yaml`` alongside the model weights."""
    config = {
        "model_variant": "yolov8m.pt",
        "data": os.path.join(data_path, "data.yaml"),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        **{k: v for k, v in TRAIN_DEFAULTS.items() if k not in ("epochs", "imgsz", "batch")},
    }

    out = ROOT_DIR / "modelos" / "train_config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Training config saved to: {out}")
    return out


def train_model(
    data_path: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cpu",
    run_name: str = "wheel-inspector-model",
):
    """Train a YOLOv8m model on the wheel-components dataset."""
    yaml_path = os.path.join(data_path, "data.yaml")
    print(f"Training: data={yaml_path}  device={device}")

    model = YOLO(TRAIN_DEFAULTS["model_variant"])

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        patience=TRAIN_DEFAULTS["patience"],
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=TRAIN_DEFAULTS["workers"],
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,
        save=True,
        save_period=-1,
        pretrained=True,
        freeze=0,
        optimizer=TRAIN_DEFAULTS["optimizer"],
        lr0=TRAIN_DEFAULTS["lr0"],
        lrf=TRAIN_DEFAULTS["lrf"],
        momentum=TRAIN_DEFAULTS["momentum"],
        weight_decay=TRAIN_DEFAULTS["weight_decay"],
        warmup_epochs=TRAIN_DEFAULTS["warmup_epochs"],
        warmup_momentum=TRAIN_DEFAULTS["warmup_momentum"],
        warmup_bias_lr=TRAIN_DEFAULTS["warmup_bias_lr"],
        box=TRAIN_DEFAULTS["box"],
        cls=TRAIN_DEFAULTS["cls"],
        dfl=TRAIN_DEFAULTS["dfl"],
        hsv_h=TRAIN_DEFAULTS["hsv_h"],
        hsv_s=TRAIN_DEFAULTS["hsv_s"],
        hsv_v=TRAIN_DEFAULTS["hsv_v"],
        degrees=TRAIN_DEFAULTS["degrees"],
        translate=TRAIN_DEFAULTS["translate"],
        scale=TRAIN_DEFAULTS["scale"],
        shear=TRAIN_DEFAULTS["shear"],
        perspective=TRAIN_DEFAULTS["perspective"],
        flipud=TRAIN_DEFAULTS["flipud"],
        fliplr=TRAIN_DEFAULTS["fliplr"],
        mosaic=TRAIN_DEFAULTS["mosaic"],
        mixup=TRAIN_DEFAULTS["mixup"],
        copy_paste=TRAIN_DEFAULTS["copy_paste"],
        val=True,
        plots=True,
        seed=TRAIN_DEFAULTS["seed"],
        deterministic=True,
        resume=False,
        amp=False,
        fraction=1.0,
        profile=False,
        verbose=True,
    )

    print_metrics_summary(results)
    return results


def print_metrics_summary(metrics: dict) -> None:
    """Print a formatted summary of the final validation metrics."""
    if not metrics:
        print("Training complete!")
        return

    keys = {
        "precision":  "metrics/precision(B)",
        "recall":     "metrics/recall(B)",
        "mAP50":      "metrics/mAP50(B)",
        "mAP50-95":   "metrics/mAP50-95(B)",
    }

    print("\nTraining complete!")
    print("\n── Final Metrics ────────────────────────────────────────")
    for label, key in keys.items():
        value = metrics.get(key)
        if value is not None:
            print(f"  {label:<12}: {value:.4f}")
    print("─────────────────────────────────────────────────────────")


def main():
    parser = argparse.ArgumentParser(
        description="Download Roboflow dataset and train YOLOv8 model"
    )
    parser.add_argument("--workspace", type=str, required=True, help="Roboflow workspace")
    parser.add_argument("--project", type=str, required=True, help="Roboflow project")
    parser.add_argument("--version", type=int, required=True, help="Dataset version")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu', '0', 'mps'")
    parser.add_argument("--run-name", type=str, default="wheel-inspector-model", help="Run name")
    parser.add_argument("--download-only", action="store_true", help="Download dataset only")

    args = parser.parse_args()

    data_loc = download_dataset(args.workspace, args.project, args.version)

    export_train_config(args, data_loc)

    if args.download_only:
        print(f"Download complete: {data_loc}")
    else:
        train_model(data_loc, args.epochs, args.imgsz, args.batch, args.device, args.run_name)

if __name__ == "__main__":
    main()