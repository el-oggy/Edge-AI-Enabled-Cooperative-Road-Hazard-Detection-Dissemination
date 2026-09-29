"""
PFPD-ll YOLOv8s Training Pipeline on Google Colab GPU
---------------------------------------------------
This script runs remotely on a Google Colab GPU instance via `colab exec`.
It performs:
1. Google Drive verification / mounting
2. Locating and caching dataset to local high-speed VM storage
3. Installing & initializing Ultralytics YOLOv8s
4. Training for 50 epochs (imgsz=640, batch=16, device=0)
5. Automatically saving trained weights (best.pt) and metrics back to Google Drive
"""

import os
import sys
import glob
import shutil
import zipfile
from pathlib import Path

def main():
    print("=" * 60)
    print("  PFPD-ll: YOLOv8s Road Hazard Detection Training Pipeline")
    print("=" * 60)

    # Step 1: Check GPU availability
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        print(f"[OK] GPU Detected: {gpu_name} (Total: {gpu_count})")
    else:
        print("[WARNING] No GPU detected! Running on CPU will be slow.")

    # Step 2: Ensure Google Drive is mounted
    drive_root = Path("/content/drive/MyDrive")
    if not drive_root.exists():
        print("\n[Step 1/5] Mounting Google Drive...")
        from google.colab import drive
        drive.mount("/content/drive")
    else:
        print("\n[Step 1/5] Google Drive is already mounted.")

    # Step 3: Find dataset on Drive
    print("\n[Step 2/5] Locating dataset on Google Drive...")
    candidates = [
        drive_root / "unified_yolov8_dataset",
        drive_root / "unified_yolov8_dataset.zip",
        drive_root / "archive.zip",
        drive_root / "PFPD-ll" / "data_traning" / "unified_yolov8_dataset",
        drive_root / "PFPD-ll" / "data_traning" / "unified_yolov8_dataset.zip",
        drive_root / "PFPD-ll" / "unified_yolov8_dataset",
        drive_root / "PFPD-ll" / "data_traning" / "archive.zip",
        drive_root / "PFPD-ll" / "archive.zip",
    ]

    dataset_source = None
    for c in candidates:
        if c.exists():
            dataset_source = c
            break

    # If not found in standard paths, search Drive for data.yaml or archive.zip
    if not dataset_source:
        print("Searching Drive for dataset...")
        zips = list(drive_root.glob("**/unified_yolov8_dataset.zip")) + list(drive_root.glob("**/archive.zip"))
        if zips:
            dataset_source = zips[0]
        else:
            yamls = list(drive_root.glob("**/data.yaml"))
            if yamls:
                dataset_source = yamls[0].parent

    if not dataset_source:
        print(f"\n[ERROR] Could not find 'unified_yolov8_dataset' or 'archive.zip' on your Google Drive!")
        print(f"Please ensure your dataset folder or zip is uploaded to your Google Drive (MyDrive).")
        sys.exit(1)

    print(f"[OK] Found dataset source at: {dataset_source}")

    # Step 3: Fast local caching on VM SSD
    local_data_dir = Path("/content/dataset")
    local_data_dir.mkdir(parents=True, exist_ok=True)

    if dataset_source.is_file() and dataset_source.suffix.lower() == ".zip":
        print(f"\n[Step 3/5] Extracting {dataset_source.name} to local VM NVMe SSD ({local_data_dir})...")
        with zipfile.ZipFile(dataset_source, 'r') as zip_ref:
            zip_ref.extractall(local_data_dir)
        # Check if extracted inside a subfolder
        if (local_data_dir / "unified_yolov8_dataset").exists():
            data_root = local_data_dir / "unified_yolov8_dataset"
        elif (local_data_dir / "images").exists():
            data_root = local_data_dir
        else:
            subdirs = [d for d in local_data_dir.iterdir() if d.is_dir() and (d / "images").exists()]
            data_root = subdirs[0] if subdirs else local_data_dir
    else:
        print(f"\n[Step 3/5] Copying dataset to local VM storage for high-speed I/O...")
        data_root = local_data_dir / "unified_yolov8_dataset"
        if not data_root.exists():
            shutil.copytree(dataset_source, data_root)

    print(f"[OK] Local dataset root: {data_root}")

    # Generate localized data.yaml
    train_yaml = data_root / "data.yaml"
    yaml_content = f"""path: {data_root.as_posix()}
train: images/train
val: images/val

names:
  0: pothole
  1: crack_longitudinal
  2: crack_transverse
  3: crack_alligator
  4: construction_zone
  5: accident_vehicle
  6: road_obstruction
  7: traffic_cone
  8: barricade
  9: debris
"""
    with open(train_yaml, "w") as f:
        f.write(yaml_content)
    print(f"[OK] Configured YAML: {train_yaml}")

    # Step 4: Install & Run YOLOv8 Training
    print("\n[Step 4/5] Initializing Ultralytics YOLOv8s...")
    try:
        from ultralytics import YOLO
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        from ultralytics import YOLO

    model = YOLO("yolov8s.pt")
    print(f"[OK] Pretrained model loaded: yolov8s.pt")

    print("\nStarting Training: 50 epochs, batch=16, imgsz=640...")
    results = model.train(
        data=str(train_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else "cpu",
        project="/content/runs/detect",
        name="pfpd_yolov8s",
        save=True,
        plots=True,
        workers=4
    )

    # Step 5: Backup weights & results back to Google Drive
    print("\n[Step 5/5] Backing up trained model weights to Google Drive...")
    backup_dir = drive_root / "PFPD_models" / "yolov8s_run"
    backup_dir.mkdir(parents=True, exist_ok=True)

    weights_best = Path("/content/runs/detect/pfpd_yolov8s/weights/best.pt")
    weights_last = Path("/content/runs/detect/pfpd_yolov8s/weights/last.pt")
    results_dir = Path("/content/runs/detect/pfpd_yolov8s")

    if weights_best.exists():
        shutil.copy2(weights_best, backup_dir / "best.pt")
        print(f"[SUCCESS] Saved best weights -> {backup_dir / 'best.pt'}")
    if weights_last.exists():
        shutil.copy2(weights_last, backup_dir / "last.pt")
        print(f"[SUCCESS] Saved last weights -> {backup_dir / 'last.pt'}")

    # Copy plots and metrics summary
    for plot_file in results_dir.glob("*.png"):
        shutil.copy2(plot_file, backup_dir / plot_file.name)
    if (results_dir / "results.csv").exists():
        shutil.copy2(results_dir / "results.csv", backup_dir / "results.csv")

    print("\n" + "=" * 60)
    print(f"  Training Complete! Model weights preserved in Google Drive:")
    print(f"  {backup_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
