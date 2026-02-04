# D-FINE Simplified

This repository is a customized version of D-FINE to simplify the test and training processes. This repository is based on the original D-FINE repository: [D-FINE](https://github.com/Peterande/D-FINE)


---

## 🏋️ Training

```bash
# Set model size
export model=s  # n s m l x

# Training
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 \
    train.py -c configs/dfine/custom/dfine_hgnetv2_${model}_custom.yml \
    --use-amp --seed=0

# With W&B logging (set use_wandb: True in configs/runtime.yml)
```

---

## 🧪 Comprehensive Testing

Use `test.py` to run complete model evaluation with all metrics:

### Basic Usage
```bash
python test.py \
    -c configs/dfine/custom/dfine_hgnetv2_s_custom.yml \
    -r output/best.pth \
    -o ./test_results
```

### With TensorRT FPS Benchmark
```bash
python test.py \
    -c configs/dfine/custom/dfine_hgnetv2_s_custom.yml \
    -r output/best.pth \
    --trt-engine output/model.engine \
    -o ./test_results
```

### With W&B Logging
```bash
python test.py \
    -c configs/dfine/custom/dfine_hgnetv2_s_custom.yml \
    -r output/best.pth \
    --trt-engine output/model.engine \
    --use-wandb \
    -o ./test_results
```

### Test Output Metrics

| Category | Metrics |
|----------|---------|
| **Model Info** | FLOPs, MACs, Parameters |
| **PyTorch Speed** | Latency (ms), FPS |
| **TensorRT Speed** | Latency (ms), FPS |
| **COCO Metrics** | AP50:95, AP50, AP75, APsmall, APmedium, APlarge, AR1, AR10, AR100, ARsmall, ARmedium, ARlarge |
| **Validator Metrics** | F1, Precision, Recall, IoU, TPs, FPs, FNs |
| **Per-Class Metrics** | precision_{class}, recall_{class}, iou_{class} |

### Output Files
- `test_results.json` - All metrics in JSON format
- `test_summary.txt` - Human-readable summary report

---

## 🔄 TensorRT Conversion

```bash
# 1. Export to ONNX
python tools/deployment/export_onnx.py \
    -c configs/dfine/custom/dfine_hgnetv2_s_custom.yml \
    -r output/best.pth \
    -o output/model.onnx

# 2. Create TensorRT engine
trtexec --onnx=output/model.onnx \
    --saveEngine=output/model.engine \
    --fp16
```

---

## 📊 W&B Integration

Enable W&B logging by setting in `configs/runtime.yml`:

```yaml
use_wandb: True
project_name: "Your-Project-Name"
exp_name: "experiment-1"
```

### Metrics Logged to W&B

| Category | Metrics |
|----------|---------|
| **Training** | train/loss, lr, epoch |
| **COCO AP** | AP50:95, AP50, AP75, APsmall, APmedium, APlarge |
| **COCO AR** | AR1, AR10, AR100, ARsmall, ARmedium, ARlarge |
| **Validator** | f1, precision, recall, iou, TPs, FPs, FNs |
| **Per-Class** | per_class/precision_{id}, per_class/recall_{id}, per_class/iou_{id} |

---

## 🎨 Available Data Augmentations

Located in `src/data/transforms/`:

| Transform | Description | Key Parameters |
|-----------|-------------|----------------|
| `RandomPhotometricDistort` | Brightness, contrast, saturation, hue changes | `p: 0.5` |
| `RandomZoomOut` | Random zoom out with padding | `fill: 0` |
| `RandomIoUCrop` | IoU-based random cropping | `min_scale: 0.3`, `max_scale: 1`, `p: 0.8` |
| `RandomHorizontalFlip` | Horizontal flip | `p: 0.5` |
| `RandomCrop` | Random cropping | `size` |
| `Resize` | Image resizing | `size: [640, 640]` |
| `Mosaic` | 4-image mosaic augmentation | `size`, `max_size` |
| `PadToSize` | Pad image to specific size | `size`, `fill`, `padding_mode` |
| `Normalize` | Image normalization | `mean`, `std` |

### Augmentation Configuration

Edit `configs/dfine/include/dataloader.yml`:

```yaml
train_dataloader:
  dataset:
    transforms:
      ops:
        - {type: RandomPhotometricDistort, p: 0.5}
        - {type: RandomZoomOut, fill: 0}
        - {type: RandomIoUCrop, p: 0.8}
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: RandomHorizontalFlip}
        - {type: Resize, size: [640, 640]}
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: ConvertPILImage, dtype: 'float32', scale: True}
        - {type: ConvertBoxes, fmt: 'cxcywh', normalize: True}
      policy:
        name: stop_epoch
        epoch: 72  # Stop some augmentations after epoch 72
        ops: ['RandomPhotometricDistort', 'RandomZoomOut', 'RandomIoUCrop']
```

---

## 📁 Project Structure

```
D-FINE-Simplified/
├── train.py                 # Training script
├── test.py                  # Comprehensive testing script
├── configs/
│   ├── dfine/
│   │   ├── custom/          # Custom dataset configs
│   │   └── include/
│   │       ├── dataloader.yml    # Data augmentation settings
│   │       └── dfine_hgnetv2.yml # Model settings
│   ├── dataset/
│   │   └── custom_detection.yml  # Dataset paths
│   └── runtime.yml          # Runtime settings (W&B, EMA, etc.)
├── src/
│   ├── data/transforms/     # Augmentation implementations
│   └── solver/
│       ├── det_solver.py    # Training/validation logic
│       ├── det_engine.py    # Train/eval functions
│       └── validator.py     # Metrics calculation
└── tools/
    ├── benchmark/           # Speed benchmarking tools
    ├── deployment/          # ONNX export tools
    └── inference/           # Inference scripts
```

---

## 📖 References

- Original D-FINE: [https://github.com/Peterande/D-FINE](https://github.com/Peterande/D-FINE)
- Paper: [D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement](https://arxiv.org/abs/2410.13842)


