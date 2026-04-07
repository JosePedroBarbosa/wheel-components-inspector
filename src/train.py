import os
import argparse
from dotenv import load_dotenv
import roboflow
from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter
from ultralytics.utils.callbacks.tensorboard import callbacks as tb_callbacks

load_dotenv()

def download_dataset(workspace_name, project_name, version_num):
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY not found in .env file.")

    src_dir = os.path.dirname(os.path.abspath(__file__))    
    target_path = os.path.abspath(os.path.join(src_dir, "..", "dataset"))  
    os.makedirs(target_path, exist_ok=True)

    print(f"Dataset será guardado em: {target_path}")

    print("Logging into Roboflow...")
    rf = roboflow.Roboflow(api_key=api_key)

    print(f"Downloading dataset {workspace_name}/{project_name} v{version_num}...")
    project = rf.workspace(workspace_name).project(project_name)
    dataset = project.version(version_num).download("yolov8", location=target_path, overwrite=True)

    print(f"Dataset extraído para: {dataset.location}")
    return dataset.location

def train_model(data_path, epochs=100, imgsz=640, batch=16, device="cpu"):
    print(f"Starting training with data={data_path}/data.yaml on device={device}")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Load the model
    # Full list of available pre-trained models: https://docs.ultralytics.com/models/yolov8/#performance-metrics
    model = YOLO("yolov8m.pt")  # Load the pre-trained model weights
    
    yaml_path = os.path.join(data_path, "data.yaml")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # model.train() — Full hyperparameter reference
    # ─────────────────────────────────────────────────────────────────────────────
    results = model.train(
        # ── Dataset ──────────────────────────────────────────────────────────────
        data=yaml_path,     # Path to the dataset config file (YAML). Must contain train/val/test paths and class names.

        # ── Core training settings ────────────────────────────────────────────────
        epochs=epochs,      # Total number of training epochs. Typical range: 50–300.
        patience=50,        # Early stopping: halt training if no improvement is seen for this many epochs. Set to 0 to disable.
        batch=batch,        # Number of images per training batch.
        imgsz=imgsz,        # Input image size (pixels). Images are resized to this square.
        
        # ── Hardware ──────────────────────────────────────────────────────────────
        device=device,      # Device to train on ("cpu", 0, [0, 1], "mps").
        workers=8,          # Number of DataLoader worker threads for loading images.
        
        # ── Checkpointing & output ────────────────────────────────────────────────
        project="runs/train",   # Root folder where training results are saved.
        name="wheel-inspector-model", # Sub-folder name for this specific run.
        exist_ok=True,      # If True, the existing folder is overwritten.
        save=True,          # Save checkpoints (best.pt and last.pt) during training.
        save_period=-1,     # Save a checkpoint every N epochs. -1 = disabled.
        
        # ── Transfer learning ─────────────────────────────────────────────────────
        pretrained=True,    # Start from pre-trained weights (recommended).
        freeze=0,           # Freeze the first N layers of the backbone (do not update their weights).

        # ── Optimiser ─────────────────────────────────────────────────────────────
        optimizer="auto",   # Optimiser algorithm. Options: "SGD", "Adam", "AdamW", "auto"
        lr0=0.01,           # Initial learning rate.
        lrf=0.01,           # Final learning rate as a fraction of lr0.
        momentum=0.937,     # SGD momentum / Adam beta1.
        weight_decay=0.0005,# L2 regularisation penalty — discourages large weights and helps prevent overfitting.
        warmup_epochs=3.0,  # Number of epochs for learning-rate warm-up at the start.
        warmup_momentum=0.8,# Initial momentum during the warm-up phase.
        warmup_bias_lr=0.1, # Initial learning rate for bias parameters during warm-up.

        # ── Loss weights ──────────────────────────────────────────────────────────
        box=7.5,            # Weight for the bounding-box regression loss.
        cls=0.5,            # Weight for the classification loss.
        dfl=1.5,            # Weight for the Distribution Focal Loss (box refinement).

        # ── Augmentation ──────────────────────────────────────────────────────────
        hsv_h=0.015,        # Random hue shift (fraction of 360°). Adds colour variety.
        hsv_s=0.7,          # Random saturation shift.
        hsv_v=0.4,          # Random brightness (value) shift.
        degrees=0.0,        # Random rotation range in degrees.
        translate=0.1,      # Random translation as a fraction of image size.
        scale=0.5,          # Random scale factor.
        shear=0.0,          # Random shear angle in degrees.
        perspective=0.0,    # Random perspective distortion. Range: 0.0–0.001.
        flipud=0.0,         # Probability of vertical flip. 0.0 = disabled.
        fliplr=0.5,         # Probability of horizontal flip. 0.5 = 50% chance per image.
        mosaic=1.0,         # Probability of Mosaic augmentation (combines 4 images).
        mixup=0.0,          # Probability of MixUp augmentation (blends 2 images).
        copy_paste=0.0,     # Probability of Copy-Paste augmentation.

        # ── Evaluation ────────────────────────────────────────────────────────────
        val=True,           # Run validation after each epoch to track mAP, loss, etc.
        plots=True,         # Generate and save training plots.

        # ── Reproducibility ───────────────────────────────────────────────────────
        seed=0,             # Random seed for reproducibility.
        deterministic=True, # Force deterministic CUDA operations.

        # ── Misc ──────────────────────────────────────────────────────────────────
        resume=False,       # Resume training from the last saved checkpoint (last.pt).
        amp=False,          # Automatic Mixed Precision (FP16).
        fraction=1.0,       # Fraction of the training dataset to use.
        profile=False,      # Profile ONNX and TensorRT speeds during training.
        verbose=True,       # Print detailed training logs to the console.
    )
    print("Training complete!")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset and train YOLOv8 model")
    parser.add_argument("--workspace", type=str, required=True, help="Roboflow workspace name")
    parser.add_argument("--project", type=str, required=True, help="Roboflow project name")
    parser.add_argument("--version", type=int, required=True, help="Roboflow dataset version number")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training ('cpu', '0', 'mps')")
    parser.add_argument("--download-only", action="store_true", help="Download dataset and exit without training")
    
    args = parser.parse_args()
    
    data_loc = download_dataset(args.workspace, args.project, args.version)
    
    if args.download_only:
        print(f"Download concluído com sucesso para: {data_loc}")
    else:
        train_model(data_loc, args.epochs, args.imgsz, args.batch, args.device)