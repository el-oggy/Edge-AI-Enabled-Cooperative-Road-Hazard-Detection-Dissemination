# YOLO11n Pothole Detection Baseline

## Dataset

- Images: 665
- Cleaned bounding boxes: 1,739
- Classes: pothole
- Train: 532
- Validation: 66
- Test: 67
- Random seed: 42

## Training

- Model: YOLO11n
- Image size: 640
- Epochs: 50
- Batch size: 16
- GPU: NVIDIA Tesla T4
- Workers: 2
- Seed: 42
- Patience: 15
- Optimizer: auto

## Held-Out Test Evaluation

| Metric | Value |
|---|---:|
| Precision | 0.8959 |
| Recall | 0.6522 |
| mAP@50 | 0.7926 |
| mAP@50-95 | 0.5108 |

The test set contains 67 images and 161 pothole instances.

The test set is frozen for future model comparisons.