[English](README.md)

# LAU: Lite-Align Upsampler

LAU 是一个像素级 EEG 到图像重建模块，将 EEG 脑信号对齐到**降采样图像空间**（128x128）而非 VAE 潜空间。通过将图像重建视为从 EEG 嵌入直接进行像素级回归，LAU 避免了潜空间对齐带来的信息瓶颈，生成视觉上有意义的低分辨率图像，可由任意现成的超分辨率模型进一步精炼。

核心思路很简单：将 EEG 特征对齐到 128x128 降采样目标，比映射到压缩的 VAE 潜表示更直接、更易学习。轻量级卷积解码器随后将编码特征上采样为完整的 RGB 图像。

## 核心思想

1. **降采样图像空间对齐** -- 不将 EEG 信号投影到 VAE 潜空间（会压缩空间信息），而是将 EEG 嵌入对齐到 128x128 降采样图像的原始像素空间，保留更多空间细节，简化学习目标。

2. **轻量级上采样解码器** -- 紧凑的 `ATM_Adapter` 模块（全连接 + 转置卷积）将 1024 维 EEG 嵌入转换为 3x128x128 RGB 图像，单次前向传播完成，无需迭代采样。

## 安装

**前提条件**：Python 3.10+，CUDA GPU。

```bash
pip install -r requirements.txt
```

主要依赖：`torch`、`torchvision`、`einops`、`open_clip_torch`、`pytorch-msssim`、`transformers`、`numpy`。

## 数据准备

1. 从 [EEG Image Decode (HuggingFace)](https://huggingface.co/datasets/LidongYang/EEG_Image_decode/tree/main) 下载预处理后的 EEG 数据集。

2. 下载 `Preprocessed_data_250Hz/` 目录并放到 `data/processed/` 下：

```
data/processed/
└── Preprocessed_data_250Hz/
    ├── sub-01/
    │   ├── preprocessed_eeg_training.npy
    │   └── preprocessed_eeg_test.npy
    ├── sub-02/
    │   └── ...
    └── ...
```

3. 准备图像集。使用 `src/data/preprocess_images.py` 将图像降采样至 128x128：

```bash
python -m src.data.preprocess_images \
  --input_dir /path/to/raw/images \
  --output_dir data/processed/image_set \
  --size 128
```

> **提示：** 如果遇到相对路径的导入错误，可尝试使用 `--input_dir` 和 `--output_dir` 的绝对路径。

## 快速开始

### 训练

```bash
python -m src.train.eeg_encoders.train_atm_s \
  --config configs/eeg_encoders/atm_s_config.json
```

端到端训练 ATM_S 编码器：EEG 信号经过编码器主干生成 1024 维嵌入，适配器将其解码为 128x128 图像。模型使用 `LowLevelLoss` 对降采样后的真实图像进行训练。

检查点按照配置保存到 `ckpt/atm_s/`。

### 推理

```bash
python -m src.inference.eeg_encoders.inference_atm_s \
  --config configs/eeg_encoders/atm_s_config.json
```

重建图像保存到 `outputs/images/LR/`。

> **提示：** 训练、推理中如果遇到相对路径的导入错误，可尝试修改config中的路径为绝对路径。

## 结果

像素级重建性能对比（低级指标）：

| 方法 | PixCorr↑ | SSIM↑ |
|------|----------|-------|
| ATM | 0.160 | 0.345 |
| **LAU（本文）** | **0.292** | **0.553** |

## 模型架构

所有编码器变体均继承自 `BrainSignalEncoder`（抽象基类，位于 `src/models/eeg_encoders/brain_signal_encoder.py`），定义了两阶段流水线：

```
forward(x, subject_ids) = adapter(encode(x, subject_ids))
```

### ATM_Adapter（像素解码器）

共享的上采样模块，将 1024 维特征向量转换为 3x128x128 RGB 图像：

| 层 | 输出形状 | 说明 |
|----|---------|------|
| `Linear(1024, 4096)` | `[B, 4096]` | 256 * 4 * 4 |
| `ReLU` | `[B, 4096]` | |
| `Unflatten(1, (256, 4, 4))` | `[B, 256, 4, 4]` | |
| `ConvTranspose2d(256, 128, 4, stride=2, padding=1)` | `[B, 128, 8, 8]` | + BN + ReLU |
| `ConvTranspose2d(128, 64, 4, stride=2, padding=1)` | `[B, 64, 16, 16]` | + BN + ReLU |
| `ConvTranspose2d(64, 32, 4, stride=2, padding=1)` | `[B, 32, 32, 32]` | + BN + ReLU |
| `ConvTranspose2d(32, 16, 4, stride=2, padding=1)` | `[B, 16, 64, 64]` | + BN + ReLU |
| `ConvTranspose2d(16, 3, 4, stride=2, padding=1)` | `[B, 3, 128, 128]` | |
| `Tanh() * 127.5 + 127.5` | `[B, 3, 128, 128]` | 将 [-1,1] 映射到 [0,255] |

### ATM_S 编码器（推荐）

位于 `src/models/eeg_encoders/ATM_S/atm_s_encoder.py`。三组件流水线：

1. **iTransformer** -- 将每个 EEG 通道视为一个 token，反转标准 Transformer。使用 `DataEmbedding`（时频编码）+ 1 层 `FullAttention` 编码器（`d_model=250`、`n_heads=4`、`d_ff=256`）。输出形状 `[B, 63, 250]`。

2. **PatchEmbedding (Enc_eeg)** -- 应用 ShallowNet 风格时域卷积（`Conv2d(1,40,(1,25))` -> `AvgPool2d` -> `Conv2d(40,40,(63,1))`），然后线性投影到嵌入维度，展平为 `[B, 1440]`。

3. **Proj_eeg** -- `Linear(1440, 1024)` 加 GELU 残差块和 LayerNorm，生成最终的 1024 维 EEG 嵌入。

`ATMSEncoder` 类将 `ATMS`（上述主干）与 `ATM_Adapter` 组合，实现端到端 EEG 到图像重建。

### EEGProject 编码器

位于 `src/models/eeg_encoders/EEGProject/eeg_project_encoder.py`。更简单的方案：

1. **EEGProjectLayer** -- 将输入 `[B, 63, 250]` 展平为 `[B, 15750]`，然后应用 `Linear(15750, 1024)` 加 GELU 残差块和 LayerNorm。
2. **EEGProject_Adapter** -- 类似的转置卷积解码器，但从 `Unflatten(1, (256, 2, 2))` 开始，包含 6 层 `ConvTranspose2d`（2x2 -> 128x128）。

## 损失函数

默认训练损失为 `LowLevelLoss`：

```
L = alpha * SmoothL1(pred, target) + beta * SSIMLoss(pred, target)
```

默认权重 `alpha=1`、`beta=10`。SSIM 损失定义为 `1 - SSIM(pred, target)`，`data_range=255`。

训练使用混合精度（`torch.amp.autocast`）配合 `GradScaler` 实现稳定的 FP16 训练。

## 项目目录结构

```
LAU/
├── configs/
│   └── eeg_encoders/
│       ├── atm_s_config.json         # 训练与推理配置
│       └── eeg_project_config.json   # EEGProject 配置
├── src/
│   ├── data/
│   │   ├── eeg_dataset.py            # EEG 数据集加载
│   │   ├── preprocess_images.py      # 图像降采样与增强
│   │   └── extract_image_features.py # CLIP 特征提取
│   ├── models/
│   │   ├── eeg_encoders/
│   │   │   ├── brain_signal_encoder.py   # 抽象基类
│   │   │   ├── ATM_S/
│   │   │   │   ├── atm_s_encoder.py      # ATM_S 编码器 + ATM_Adapter
│   │   │   │   └── subject_layers/       # iTransformer 子模块
│   │   │   └── EEGProject/
│   │   │       └── eeg_project_encoder.py # 基线线性编码器
│   │   └── losses.py                     # 损失函数
│   ├── train/
│   │   └── eeg_encoders/
│   │       ├── train_atm_s.py        # ATM_S 训练脚本
│   │       └── train_eeg_project.py  # EEGProject 训练脚本
│   └── inference/
│       └── eeg_encoders/
│           ├── inference_atm_s.py    # ATM_S 推理脚本
│           └── inference_eeg_project.py  # EEGProject 推理脚本
├── ckpt/                             # 模型检查点
├── outputs/
│   └── images/
│       └── LR/                       # 重建的低分辨率图像
├── asset.jpg
├── requirements.txt
└── README.md
```

## 致谢

ATM_S 编码器基于 [Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2f2f44a0b1d4e6c4c4f6c0f7a8b9c0d1-Abstract-Conference.html) (Li et al., NeurIPS 2024)。预处理后的 EEG 数据集来自其 [HuggingFace 仓库](https://huggingface.co/datasets/LidongYang/EEG_Image_decode)。

## 许可证

MIT License
