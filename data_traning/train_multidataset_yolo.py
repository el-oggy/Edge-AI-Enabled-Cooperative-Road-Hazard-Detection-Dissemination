"""
Autonomous Multi-Dataset YOLOv8s Training and Jetson Nano Export on Colab GPU
----------------------------------------------------------------------------
Downloads road hazard benchmarks directly via Kaggle API on Colab VM:
- muskanverma24/pothole-detection-dataset-yolov11-optimized
- lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes
- vidishbijalwan/rdd2022-india-pothole-d40
Merges them with Drive baseline data, trains YOLOv8s, and exports to ONNX/TensorRT for Jetson Nano.
"""

import os
import sys
import glob
import json
import shutil
import zipfile
import subprocess
from pathlib import Path

def run_cmd(cmd):
    print(f"[EXEC] {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    print("=" * 65)
    print("  Edge-AI Road Hazard Multi-Dataset Training & Jetson Export")
    print("=" * 65)

    # Step 1: Check GPU
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[OK] GPU Detected: {gpu_name}")
    else:
        print("[WARN] No GPU detected! Proceeding anyway...")

    # Step 2: Configure Kaggle API on Colab
    print("\n[Step 1/6] Configuring Kaggle API...")
    kaggle_dir = Path("/root/.kaggle")
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    creds = {"username": "adarshswarupmaharana", "key": "49aa335ed3ba357601d3df953ffff5ce"}
    with open(kaggle_dir / "kaggle.json", "w") as f:
        json.dump(creds, f)
    os.chmod(kaggle_dir / "kaggle.json", 0o600)

    # Step 3: Install dependencies
    print("\n[Step 2/6] Installing Kaggle and Ultralytics...")
    run_cmd("pip install -q kaggle ultralytics onnx")

    # Step 4: Download Datasets from Kaggle
    print("\n[Step 3/6] Downloading high-impact Road Hazard Datasets via Kaggle...")
    raw_dir = Path("/content/raw_kaggle")
    raw_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("muskanverma24/pothole-detection-dataset-yolov11-optimized", raw_dir / "ds_potholes"),
        ("lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes", raw_dir / "ds_lorenzo"),
        ("vidishbijalwan/rdd2022-india-pothole-d40", raw_dir / "ds_rdd2022"),
    ]

    for ds_slug, target_path in datasets:
        target_path.mkdir(parents=True, exist_ok=True)
        print(f"Fetching: {ds_slug} -> {target_path}...")
        try:
            run_cmd(f"kaggle datasets download -d {ds_slug} --unzip -p \"{target_path}\"")
        except Exception as e:
            print(f"[WARN] Could not download {ds_slug}: {e}")

    # Step 5: Mount Google Drive if available
    drive_mounted = False
    drive_root = Path("/content/drive/MyDrive")
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        drive_mounted = True
        print("[OK] Google Drive mounted successfully.")
    except Exception as e:
        print(f"[INFO] Google Drive mount check: {e}")
        if drive_root.exists():
            drive_mounted = True

    # Step 6: Consolidate & Merge Datasets
    print("\n[Step 4/6] Consolidating and standardizing datasets...")
    unified_dir = Path("/content/merged_dataset")
    img_train_dir = unified_dir / "images" / "train"
    img_val_dir = unified_dir / "images" / "val"
    lbl_train_dir = unified_dir / "labels" / "train"
    lbl_val_dir = unified_dir / "labels" / "val"

    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    copied_images = 0

    # Scan raw datasets for images and labels
    for sub in raw_dir.glob("**/*"):
        if sub.is_dir() and ("train" in sub.name.lower() or "val" in sub.name.lower() or "valid" in sub.name.lower()):
            is_val = "val" in sub.name.lower() or "valid" in sub.name.lower() or "test" in sub.name.lower()
            dst_img = img_val_dir if is_val else img_train_dir
            dst_lbl = lbl_val_dir if is_val else lbl_train_dir

            imgs = list(sub.glob("*.jpg")) + list(sub.glob("*.png")) + list(sub.glob("images/*.jpg")) + list(sub.glob("images/*.png"))
            for img in imgs:
                lbl_name = img.stem + ".txt"
                lbl_candidates = [
                    img.parent / lbl_name,
                    img.parent.parent / "labels" / lbl_name,
                    sub / "labels" / lbl_name,
                ]
                lbl_file = next((f for f in lbl_candidates if f.exists()), None)
                if lbl_file:
                    dest_img_path = dst_img / f"{img.parent.stem}_{img.name}"
                    dest_lbl_path = dst_lbl / f"{img.parent.stem}_{lbl_name}"
                    if not dest_img_path.exists():
                        shutil.copy2(img, dest_img_path)
                        shutil.copy2(lbl_file, dest_lbl_path)
                        copied_images += 1

    train_count = len(list(img_train_dir.glob("*.*")))
    val_count = len(list(img_val_dir.glob("*.*")))
    print(f"[OK] Merged Dataset Ready! Train: {train_count} images | Val: {val_count} images (Total: {train_count + val_count})")

    # Build data.yaml
    data_yaml_path = unified_dir / "data.yaml"
    yaml_text = f"""path: {unified_dir.as_posix()}
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
    with open(data_yaml_path, "w") as f:
        f.write(yaml_text)

    # Step 7: Train YOLOv8s
    print("\n[Step 5/6] Starting YOLOv8s Training on Tesla T4...")
    from ultralytics import YOLO

    model = YOLO("yolov8s.pt")
    results = model.train(
        data=str(data_yaml_path),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else "cpu",
        project="/content/runs/detect",
        name="pfpd_yolov8s_multidataset",
        save=True,
        plots=True,
        workers=4,
        seed=42,
        patience=15
    )

    # Step 8: Export for NVIDIA Jetson Nano
    print("\n[Step 6/6] Exporting best model to ONNX for NVIDIA Jetson Nano...")
    best_weights = Path("/content/runs/detect/pfpd_yolov8s_multidataset/weights/best.pt")
    if best_weights.exists():
        trained_model = YOLO(str(best_weights))
        try:
            exported_onnx = trained_model.export(format="onnx", opset=12, dynamic=False, imgsz=640)
            print(f"[SUCCESS] Exported ONNX for Jetson Nano: {exported_onnx}")
        except Exception as e:
            print(f"[WARN] ONNX export note: {e}")

    # Save to Drive if mounted
    if drive_mounted:
        dest_dir = drive_root / "PFPD_models" / "yolov8s_multidataset"
        dest_dir.mkdir(parents=True, exist_ok=True)
        if best_weights.exists():
            shutil.copy2(best_weights, dest_dir / "best.pt")
            print(f"[SAVED] Uploaded best.pt to Drive: {dest_dir / 'best.pt'}")
        run_res = Path("/content/runs/detect/pfpd_yolov8s_multidataset")
        for p in run_res.glob("*.png"):
            shutil.copy2(p, dest_dir / p.name)
        for onnx_p in best_weights.parent.glob("*.onnx"):
            shutil.copy2(onnx_p, dest_dir / onnx_p.name)
        if (run_res / "results.csv").exists():
            shutil.copy2(run_res / "results.csv", dest_dir / "results.csv")
        print(f"[SAVED] All training curves & model artifacts backed up to Drive: {dest_dir}")

    print("\n" + "=" * 65)
    print("  All Stages Complete! Ready for Edge-AI deployment & SUMO integration.")
    print("=" * 65)

if __name__ == "__main__":
    main()
