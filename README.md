# Edge-AI Enabled Cooperative Road Hazard Detection & Dissemination

An Edge-AI and VANET-oriented project for detecting road hazards
and disseminating warnings to nearby vehicles.

## Current Status

### Pothole Detection Baseline

- Model: YOLO11n
- Dataset: 665 images
- Cleaned annotations: 1,739
- Classes: pothole
- Train: 532 images
- Validation: 66 images
- Test: 67 images
- Random seed: 42

### Held-Out Test Results

| Metric | Result |
|---|---:|
| Precision | 89.59% |
| Recall | 65.22% |
| mAP@50 | 79.26% |
| mAP@50-95 | 51.08% |

The test set is frozen for future model comparisons.

## Repository Structure

```text
.
├── configs/
├── data/
├── docs/
├── models/
├── notebooks/
├── reports/
├── results/
├── simulation/
└── src/
```

## Roadmap

1. Establish reproducible pothole baseline — COMPLETE
2. Expand with additional pothole and road-damage datasets
3. Normalize annotations and prevent data leakage
4. Retrain and compare against frozen baseline test set
5. Evaluate edge deployment
6. Connect hazard detection to V2X/VANET messaging
7. Integrate with SUMO traffic simulation

## Data Policy

Large datasets, generated image folders, model checkpoints and caches
are intentionally excluded from Git.

The repository stores code, configurations, documentation, experiment
results, plots, dataset metadata and reproducibility information.