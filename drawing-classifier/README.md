# AR-BOOK Drawing Classifier

Hand-drawn image classification baseline using MobileNetV3-Small + TFLite.

## Classes
`tiger`, `pig`, `flower`, `cloud`

## Dataset
[Google Quick Draw](https://github.com/googlecreativelab/quickdraw-dataset) — 5,000 samples per class (80/20 train/val split).

## Pipeline

```
download_data.py → train.py → export_tflite.py → inference.py
```

| Step | Script | Output |
|------|--------|--------|
| 1. Download & split data | `python download_data.py` | `data/train/`, `data/val/` |
| 2. Train model | `python train.py` | `outputs/checkpoints/best.pt` |
| 3. Export to TFLite | `python export_tflite.py` | `outputs/tflite/drawing_classifier.tflite` |
| 4. Test inference | `python inference.py <image>` | Predicted class + confidence |

## Setup

```bash
pip install -r requirements.txt
```

## Scope
This module handles **classification only**. Unity integration, 2.5D asset generation, and AR placement are separate modules.
