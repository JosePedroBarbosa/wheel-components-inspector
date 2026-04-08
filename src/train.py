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
    "model_variant": "yolov8m.pt",
    "epochs": 100,
    "patience": 50,
    "batch": 16,
    "imgsz": 640,
    "workers": 8,
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "seed": 0,
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

    print("Training complete!")
    return results


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
