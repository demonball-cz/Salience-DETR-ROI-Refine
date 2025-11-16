# Salience-DETR-ROI-Refine

This repository provides training and inference scripts for a DETR-based object detection model with salience-guided ROI refinement. It builds on PyTorch and Hugging Face Accelerate for distributed training and supports COCO-style datasets.

## Project layout
- `main.py`: entry point for distributed training with gradient accumulation, mixed precision, and automatic checkpointing.
- `inference.py`: batch inference script that loads a trained checkpoint and optionally saves visualized predictions.
- `configs/`: configuration files for training and model definitions (see `configs/train_config.py` and subfolder `salience_detr/`).
- `datasets/`, `transforms/`, `models/`, `optimizer/`: dataset handling, data augmentation presets, model architectures, and optimizer parameter grouping helpers.
- `util/`: utilities for logging, checkpoint management, evaluation, and helper functions.
- `tools/` and `grad_cam/`: auxiliary tools and visualization utilities.

## Training
1. Edit `configs/train_config.py` to point to your COCO-format dataset and desired output directory.
2. Launch training (single or multi-GPU) with Accelerate:
   ```bash
   python main.py --config-file configs/train_config.py --mixed-precision fp16
   ```
   Useful flags include `--accumulate-steps` for gradient accumulation and `--dynamo-backend` to enable TorchDynamo compilation.

## Inference
Run inference on a directory of images with a trained checkpoint:
```bash
python inference.py \
  --image-dir /path/to/images \
  --model-config configs/salience_detr/salience_detr_resnet50_800_1333.py \
  --checkpoint checkpoints/salience_detr_resnet50_800_1333/train/<run>/best_ap.pth \
  --show-dir outputs/visualizations
```
Optional arguments let you control visualization details (font scale, box thickness, confidence threshold, and colors) or disable saving by omitting `--show-dir`.

## Notes
- Training leverages Accelerate's `Accelerator` for distributed execution and automatic TensorBoard logging.
- Checkpoints keep track of the best AP/AP50 metrics during evaluation and preserve label mappings for inference.
- Deterministic training is supported via the `--use-deterministic-algorithms` flag, though it may impact performance on older PyTorch versions.
