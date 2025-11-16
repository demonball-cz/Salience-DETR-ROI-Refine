# Salience-DETR with ROI Refine

[English](#english) | [中文](#中文)

---

<a name="english"></a>
## English

### Overview

**Salience-DETR-ROI-Refine** is an advanced object detection framework that extends DETR (Detection Transformer) with salience-based attention mechanisms and ROI (Region of Interest) refinement. This implementation enhances detection performance by focusing on salient regions and refining detected boxes through a query-conditioned ROI transformer decoder.

### Key Features

- **Salience-based Attention**: Focus on salient regions in multi-scale feature maps
- **ROI Refinement**: Query-conditioned ROI transformer for precise box regression
- **Multi-Scale Detection**: Support for 4-level feature pyramid networks
- **Multiple Backbone Support**: ResNet50, ConvNeXt, FocalNet, Swin Transformer, etc.
- **Two-Stage Detection**: Encoder proposals + decoder refinement
- **Denoising Training**: Contrastive denoising for robust training
- **Flexible Configuration**: Easy-to-customize model and training configurations

### Architecture Components

- **Backbone**: ResNet, ConvNeXt, FocalNet, Swin Transformer, ViT
- **Neck**: Channel Mapper with RepVGG-style blocks
- **Encoder**: Multi-scale deformable attention encoder with salience filtering
- **Decoder**: Salience-aware decoder with ROI refinement module
- **Loss Functions**: Classification, BBox regression, GIoU, Salience loss

### Installation

#### Requirements

- Python >= 3.7
- PyTorch >= 1.10
- CUDA (recommended for training)

#### Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
```
accelerate
torch
torchvision
albumentations
pycocotools
tensorboard
fvcore
omegaconf
```

### Dataset Preparation

This implementation uses COCO format datasets. Organize your dataset as follows:

```
/path/to/dataset/
├── train2017/
│   ├── 000000000001.jpg
│   ├── 000000000002.jpg
│   └── ...
├── val2017/
│   ├── 000000000001.jpg
│   ├── 000000000002.jpg
│   └── ...
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

### Training

#### 1. Configure Training

Edit `configs/train_config.py` to set:
- Dataset paths (`coco_path`)
- Model configuration (`model_path`)
- Training hyperparameters (batch size, learning rate, epochs)
- Output directory (`output_dir`)

Example:
```python
coco_path = "/path/to/your/dataset"
model_path = "configs/salience_detr/salience_detr_resnet50_800_1333.py"
batch_size = 2
num_epochs = 12
learning_rate = 1e-4
```

#### 2. Start Training

Basic training:
```bash
python main.py --config-file configs/train_config.py
```

With mixed precision (recommended):
```bash
python main.py \
    --config-file configs/train_config.py \
    --mixed-precision fp16
```

Multi-GPU training:
```bash
accelerate launch --multi_gpu --num_processes 4 main.py \
    --config-file configs/train_config.py \
    --mixed-precision fp16
```

Resume from checkpoint:
```bash
python main.py \
    --config-file configs/train_config.py \
    --resume-from-checkpoint /path/to/checkpoint
```

### Testing & Evaluation

Evaluate a trained model on validation set:

```bash
python test.py \
    --coco-path /path/to/dataset \
    --subset val \
    --model-config configs/salience_detr/salience_detr_resnet50_800_1333.py \
    --checkpoint /path/to/checkpoint.pth
```

With visualization:
```bash
python test.py \
    --coco-path /path/to/dataset \
    --model-config configs/salience_detr/salience_detr_resnet50_800_1333.py \
    --checkpoint /path/to/checkpoint.pth \
    --show-dir ./visualizations \
    --show-conf 0.5
```

### Inference

Run inference on custom images:

```bash
python inference.py \
    --image-dir /path/to/images \
    --model-config configs/salience_detr/salience_detr_resnet50_800_1333.py \
    --checkpoint /path/to/checkpoint.pth \
    --show-dir ./results \
    --show-conf 0.5
```

For Jupyter notebook inference, see `inference.ipynb`.

### Model Configurations

Available pre-configured models in `configs/salience_detr/`:

| Model | Backbone | Input Size | Config File |
|-------|----------|------------|-------------|
| Salience-DETR | ResNet-50 | 800×1333 | `salience_detr_resnet50_800_1333.py` |
| Salience-DETR | ResNet-50 (5-scale) | 800×1333 | `salience_detr_resnet50_5scale_800_1333.py` |
| Salience-DETR | ConvNeXt-Large | 800×1333 | `salience_detr_convnext_l_800_1333.py` |
| Salience-DETR | FocalNet-Large | 800×1333 | `salience_detr_focalnet_large_lrf_800_1333.py` |
| Salience-DETR | Swin-Large | 800×1333 | `salience_detr_swin_l_800_1333.py` |

### Project Structure

```
.
├── configs/              # Model and training configurations
│   ├── salience_detr/   # Model configs for different backbones
│   └── train_config.py  # Training configuration
├── datasets/            # Dataset loading and preprocessing
├── models/              # Model architecture
│   ├── backbones/      # Backbone networks (ResNet, Swin, etc.)
│   ├── bricks/         # Model components (transformer, losses, etc.)
│   ├── detectors/      # Main detector classes
│   ├── matcher/        # Hungarian matcher for assignment
│   └── necks/          # Feature pyramid networks
├── transforms/          # Data augmentation and transforms
├── util/               # Utility functions
├── tools/              # Additional tools (ONNX export, benchmark, etc.)
├── main.py             # Training script
├── test.py             # Evaluation script
├── inference.py        # Inference script
└── requirements.txt    # Python dependencies
```

### Key Model Parameters

Edit model config files to customize:

```python
embed_dim = 256              # Transformer embedding dimension
num_classes = 91             # Number of object classes (COCO: 91)
num_queries = 900            # Number of object queries
num_feature_levels = 4       # Number of FPN levels
transformer_enc_layers = 6   # Encoder layers
transformer_dec_layers = 6   # Decoder layers
num_heads = 8                # Attention heads
dim_feedforward = 2048       # FFN dimension
```

### Tools

#### ONNX Export
```bash
python tools/pytorch2onnx.py \
    --model-config configs/salience_detr/salience_detr_resnet50_800_1333.py \
    --checkpoint /path/to/checkpoint.pth \
    --output model.onnx
```

#### Dataset Visualization
```bash
python tools/visualize_datasets.py \
    --coco-path /path/to/dataset \
    --subset train \
    --show-dir ./visualizations
```

#### Model Benchmark
```bash
python tools/benchmark_model.py \
    --model-config configs/salience_detr/salience_detr_resnet50_800_1333.py
```

### Advanced Features

#### Gradient-based Visualization

Use notebooks in `grad_cam/` for:
- Feature map visualization
- Gradient-weighted class activation maps
- Query attention visualization

#### Custom Dataset

To use your own dataset:
1. Convert annotations to COCO format
2. Update `coco_path` in `configs/train_config.py`
3. Adjust `num_classes` in model config

### Tips & Tricks

- **Mixed Precision**: Use `--mixed-precision fp16` for faster training
- **Gradient Accumulation**: Use `--accumulate-steps N` for larger effective batch size
- **Learning Rate**: Start with `1e-4` for fine-tuning, `2e-4` for training from scratch
- **Warmup**: Enable learning rate warmup for stable training
- **Multi-scale Training**: Edit transform presets for better generalization

### Troubleshooting

**Out of Memory**: Reduce batch size or use gradient accumulation
```bash
python main.py --config-file configs/train_config.py --accumulate-steps 4
```

**Slow Training**: Enable mixed precision and increase num_workers
```python
# In train_config.py
num_workers = 8
pin_memory = True
```

---

<a name="中文"></a>
## 中文

### 概述

**Salience-DETR-ROI-Refine** 是一个先进的目标检测框架，它在 DETR（检测变换器）的基础上扩展了基于显著性的注意力机制和 ROI（感兴趣区域）细化。该实现通过聚焦于显著区域并通过查询条件化的 ROI 变换器解码器来细化检测框，从而提高检测性能。

### 主要特性

- **显著性注意力**: 关注多尺度特征图中的显著区域
- **ROI 细化**: 查询条件化的 ROI 变换器用于精确的边界框回归
- **多尺度检测**: 支持 4 层特征金字塔网络
- **多种骨干网络**: ResNet50、ConvNeXt、FocalNet、Swin Transformer 等
- **两阶段检测**: 编码器提议 + 解码器细化
- **去噪训练**: 对比去噪实现鲁棒训练
- **灵活配置**: 易于自定义的模型和训练配置

### 架构组件

- **骨干网络**: ResNet、ConvNeXt、FocalNet、Swin Transformer、ViT
- **颈部网络**: 带有 RepVGG 风格块的通道映射器
- **编码器**: 带显著性过滤的多尺度可变形注意力编码器
- **解码器**: 带 ROI 细化模块的显著性感知解码器
- **损失函数**: 分类、边界框回归、GIoU、显著性损失

### 安装

#### 环境要求

- Python >= 3.7
- PyTorch >= 1.10
- CUDA（推荐用于训练）

#### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
```
accelerate
torch
torchvision
albumentations
pycocotools
tensorboard
fvcore
omegaconf
```

### 数据集准备

此实现使用 COCO 格式数据集。按以下方式组织数据集：

```
/path/to/dataset/
├── train2017/
│   ├── 000000000001.jpg
│   ├── 000000000002.jpg
│   └── ...
├── val2017/
│   ├── 000000000001.jpg
│   ├── 000000000002.jpg
│   └── ...
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

### 训练

#### 1. 配置训练

编辑 `configs/train_config.py` 设置：
- 数据集路径 (`coco_path`)
- 模型配置 (`model_path`)
- 训练超参数（批量大小、学习率、轮数）
- 输出目录 (`output_dir`)

示例：
```python
coco_path = "/path/to/your/dataset"
model_path = "configs/salience_detr/salience_detr_resnet50_800_1333.py"
batch_size = 2
num_epochs = 12
learning_rate = 1e-4
```

#### 2. 开始训练

基础训练：
```bash
python main.py --config-file configs/train_config.py
```

使用混合精度（推荐）：
```bash
python main.py \
    --config-file configs/train_config.py \
    --mixed-precision fp16
```

多 GPU 训练：
```bash
accelerate launch --multi_gpu --num_processes 4 main.py \
    --config-file configs/train_config.py \
    --mixed-precision fp16
```

从检查点恢复：
```bash
python main.py \
    --config-file configs/train_config.py \
    --resume-from-checkpoint /path/to/checkpoint
```

### 测试与评估

在验证集上评估训练好的模型：

```bash
python test.py \
    --coco-path /path/to/dataset \
    --subset val \
    --model-config configs/salience_detr/salience_detr_resnet50_800_1333.py \
    --checkpoint /path/to/checkpoint.pth
```

带可视化：
```bash
python test.py \
    --coco-path /path/to/dataset \
    --model-config configs/salience_detr/salience_detr_resnet50_800_1333.py \
    --checkpoint /path/to/checkpoint.pth \
    --show-dir ./visualizations \
    --show-conf 0.5
```

### 推理

对自定义图像运行推理：

```bash
python inference.py \
    --image-dir /path/to/images \
    --model-config configs/salience_detr/salience_detr_resnet50_800_1333.py \
    --checkpoint /path/to/checkpoint.pth \
    --show-dir ./results \
    --show-conf 0.5
```

对于 Jupyter notebook 推理，请参阅 `inference.ipynb`。

### 模型配置

`configs/salience_detr/` 中可用的预配置模型：

| 模型 | 骨干网络 | 输入尺寸 | 配置文件 |
|------|----------|----------|----------|
| Salience-DETR | ResNet-50 | 800×1333 | `salience_detr_resnet50_800_1333.py` |
| Salience-DETR | ResNet-50 (5尺度) | 800×1333 | `salience_detr_resnet50_5scale_800_1333.py` |
| Salience-DETR | ConvNeXt-Large | 800×1333 | `salience_detr_convnext_l_800_1333.py` |
| Salience-DETR | FocalNet-Large | 800×1333 | `salience_detr_focalnet_large_lrf_800_1333.py` |
| Salience-DETR | Swin-Large | 800×1333 | `salience_detr_swin_l_800_1333.py` |

### 项目结构

```
.
├── configs/              # 模型和训练配置
│   ├── salience_detr/   # 不同骨干网络的模型配置
│   └── train_config.py  # 训练配置
├── datasets/            # 数据集加载和预处理
├── models/              # 模型架构
│   ├── backbones/      # 骨干网络（ResNet、Swin 等）
│   ├── bricks/         # 模型组件（transformer、损失等）
│   ├── detectors/      # 主检测器类
│   ├── matcher/        # 用于分配的匈牙利匹配器
│   └── necks/          # 特征金字塔网络
├── transforms/          # 数据增强和变换
├── util/               # 实用函数
├── tools/              # 附加工具（ONNX 导出、基准测试等）
├── main.py             # 训练脚本
├── test.py             # 评估脚本
├── inference.py        # 推理脚本
└── requirements.txt    # Python 依赖
```

### 关键模型参数

编辑模型配置文件以自定义：

```python
embed_dim = 256              # Transformer 嵌入维度
num_classes = 91             # 目标类别数（COCO: 91）
num_queries = 900            # 目标查询数量
num_feature_levels = 4       # FPN 层数
transformer_enc_layers = 6   # 编码器层数
transformer_dec_layers = 6   # 解码器层数
num_heads = 8                # 注意力头数
dim_feedforward = 2048       # FFN 维度
```

### 工具

#### ONNX 导出
```bash
python tools/pytorch2onnx.py \
    --model-config configs/salience_detr/salience_detr_resnet50_800_1333.py \
    --checkpoint /path/to/checkpoint.pth \
    --output model.onnx
```

#### 数据集可视化
```bash
python tools/visualize_datasets.py \
    --coco-path /path/to/dataset \
    --subset train \
    --show-dir ./visualizations
```

#### 模型基准测试
```bash
python tools/benchmark_model.py \
    --model-config configs/salience_detr/salience_detr_resnet50_800_1333.py
```

### 高级功能

#### 基于梯度的可视化

使用 `grad_cam/` 中的 notebook：
- 特征图可视化
- 梯度加权类激活图
- 查询注意力可视化

#### 自定义数据集

使用自己的数据集：
1. 将标注转换为 COCO 格式
2. 更新 `configs/train_config.py` 中的 `coco_path`
3. 调整模型配置中的 `num_classes`

### 技巧与提示

- **混合精度**: 使用 `--mixed-precision fp16` 加速训练
- **梯度累积**: 使用 `--accumulate-steps N` 获得更大的有效批量大小
- **学习率**: 微调时从 `1e-4` 开始，从头训练时用 `2e-4`
- **预热**: 启用学习率预热以稳定训练
- **多尺度训练**: 编辑变换预设以获得更好的泛化

### 故障排除

**内存不足**: 减少批量大小或使用梯度累积
```bash
python main.py --config-file configs/train_config.py --accumulate-steps 4
```

**训练缓慢**: 启用混合精度并增加 num_workers
```python
# 在 train_config.py 中
num_workers = 8
pin_memory = True
```

### 许可证

请查看项目的许可证文件了解使用条款。

### 致谢

本项目基于 DETR 和相关目标检测研究。感谢开源社区的贡献。

---

## Citation

如果您在研究中使用此代码，请引用相关论文。

## Contact

如有问题或建议，请提交 issue 或 pull request。
