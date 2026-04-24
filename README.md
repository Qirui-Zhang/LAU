[中文版](README_zh.md)

# LAU: Lite-Align Upsampler

LAU is a pixel-level EEG-to-image reconstruction module that aligns EEG brain signals to a **downsampled image space** (128x128) rather than a VAE latent space. By treating image reconstruction as direct pixel-level regression from EEG embeddings, LAU avoids the information bottleneck introduced by latent-space alignment and produces visually grounded low-resolution images that can be further refined by any off-the-shelf super-resolution model.

The key insight is simple: aligning EEG features to a 128x128 downsampling target is more direct and learnable than mapping to a compressed VAE latent representation. A lightweight convolutional decoder then upsamples the encoded features into full RGB images.

## Core Idea

1. **Downsampled-image-space alignment** -- Instead of projecting EEG signals into a VAE latent space (which compresses spatial information), LAU aligns EEG embeddings to the raw pixel space of 128x128 downsampled images. This preserves more spatial detail and simplifies the learning objective.

2. **Lightweight upsampling decoder** -- A compact `ATM_Adapter` module (fully connected + transposed convolutions) converts the 1024-dim EEG embedding into a 3x128x128 RGB image in a single forward pass, with no iterative sampling required.

## Installation

**Requirements:** Python 3.10+, CUDA-enabled GPU.

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `einops`, `open_clip_torch`, `pytorch-msssim`, `transformers`, `numpy`.

## Data Preparation

1. Download the preprocessed EEG dataset from [EEG Image Decode (HuggingFace)](https://huggingface.co/datasets/LidongYang/EEG_Image_decode/tree/main).

2. Download the `Preprocessed_data_250Hz/` directory and place it under `data/processed/`:

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

3. Prepare the image set. Place training and test images under `data/processed/image_set/`:

```
data/processed/image_set/
├── training_images/
│   ├── 00001_aardvark/
│   │   ├── aardvark_01b.jpg
│   │   └── ...
│   └── ...
└── test_images/
    └── ...
```

Use `src/data/preprocess_images.py` to downsample images to 128x128:

```bash
python -m src.data.preprocess_images \
  --input_dir /path/to/raw/images \
  --output_dir data/processed/image_set \
  --size 128
```

> **Tip:** If you encounter import errors with relative paths, try using absolute paths for `--input_dir` and `--output_dir`.

## Quick Start

### Training

```bash
python -m src.train.eeg_encoders.train_atm_s \
  --config configs/eeg_encoders/atm_s_config.json
```

This trains the ATM_S encoder end-to-end: EEG signals go through the encoder backbone to produce a 1024-dim embedding, which the adapter decodes into a 128x128 image. The model is trained against downsampled ground-truth images using the `LowLevelLoss`.

Checkpoints are saved to `ckpt/atm_s/` according to the config.

### Inference

```bash
python -m src.inference.eeg_encoders.inference_atm_s \
  --config configs/eeg_encoders/atm_s_config.json
```

Reconstructed images are saved to `outputs/images/LR/`.

## Model Architecture

All encoder variants inherit from `BrainSignalEncoder` (abstract base class in `src/models/eeg_encoders/brain_signal_encoder.py`), which defines a two-stage pipeline:

```
forward(x, subject_ids) = adapter(encode(x, subject_ids))
```

### ATM_Adapter (pixel decoder)

The shared upsampling module that converts a 1024-dim feature vector into a 3x128x128 RGB image:

| Layer | Output Shape | Details |
|-------|-------------|---------|
| `Linear(1024, 4096)` | `[B, 4096]` | 256 * 4 * 4 |
| `ReLU` | `[B, 4096]` | |
| `Unflatten(1, (256, 4, 4))` | `[B, 256, 4, 4]` | |
| `ConvTranspose2d(256, 128, 4, stride=2, padding=1)` | `[B, 128, 8, 8]` | + BN + ReLU |
| `ConvTranspose2d(128, 64, 4, stride=2, padding=1)` | `[B, 64, 16, 16]` | + BN + ReLU |
| `ConvTranspose2d(64, 32, 4, stride=2, padding=1)` | `[B, 32, 32, 32]` | + BN + ReLU |
| `ConvTranspose2d(32, 16, 4, stride=2, padding=1)` | `[B, 16, 64, 64]` | + BN + ReLU |
| `ConvTranspose2d(16, 3, 4, stride=2, padding=1)` | `[B, 3, 128, 128]` | |
| `Tanh() * 127.5 + 127.5` | `[B, 3, 128, 128]` | maps [-1,1] to [0,255] |

### ATM_S Encoder (recommended)

Located in `src/models/eeg_encoders/ATM_S/atm_s_encoder.py`. A three-component pipeline:

1. **iTransformer** -- Inverts the standard Transformer by treating each EEG channel as a token. Uses `DataEmbedding` (time-frequency encoding) + a 1-layer `FullAttention` encoder (`d_model=250`, `n_heads=4`, `d_ff=256`). Outputs shape `[B, 63, 250]`.

2. **PatchEmbedding (Enc_eeg)** -- Applies a ShallowNet-style temporal convolution (`Conv2d(1,40,(1,25))` -> `AvgPool2d` -> `Conv2d(40,40,(63,1))`) followed by a linear projection to embedding dim, then flattens to `[B, 1440]`.

3. **Proj_eeg** -- `Linear(1440, 1024)` with a GELU residual block and LayerNorm, producing the final 1024-dim EEG embedding.

The `ATMSEncoder` class combines `ATMS` (the backbone above) with `ATM_Adapter` for end-to-end EEG-to-image reconstruction.

### EEGProject Encoder 

Located in `src/models/eeg_encoders/EEGProject/eeg_project_encoder.py`. A simpler approach:

1. **EEGProjectLayer** -- Flattens the input `[B, 63, 250]` to `[B, 15750]`, then applies `Linear(15750, 1024)` with a GELU residual block and LayerNorm.
2. **EEGProject_Adapter** -- Similar transposed-convolution decoder but starts from `Unflatten(1, (256, 2, 2))` with 6 `ConvTranspose2d` layers (2x2 -> 128x128).

## Loss Function

The default training loss is `LowLevelLoss`:

```
L = alpha * SmoothL1(pred, target) + beta * SSIMLoss(pred, target)
```

With default weights `alpha=1`, `beta=10`. The SSIM loss is defined as `1 - SSIM(pred, target)` with `data_range=255`.

Training uses mixed precision (`torch.amp.autocast`) with `GradScaler` for stable FP16 training.

## Project Structure

```
LAU/
├── configs/
│   └── eeg_encoders/
│       └── atm_s_config.json         # Training & inference config
├── src/
│   ├── data/
│   │   ├── eeg_dataset.py            # EEG dataset loader
│   │   ├── preprocess_images.py      # Image downsampling & augmentation
│   │   └── extract_image_features.py # CLIP feature extraction
│   ├── models/
│   │   ├── eeg_encoders/
│   │   │   ├── brain_signal_encoder.py   # Abstract base class
│   │   │   ├── ATM_S/
│   │   │   │   ├── atm_s_encoder.py      # ATM_S encoder + ATM_Adapter
│   │   │   │   └── subject_layers/       # iTransformer sub-modules
│   │   │   └── EEGProject/
│   │   │       └── eeg_project_encoder.py # Baseline linear encoder
│   │   └── losses.py                     # Loss functions
│   ├── train/
│   │   └── eeg_encoders/
│   │       └── train_atm_s.py        # Training script
│   └── inference/
│       └── eeg_encoders/
│           └── inference_atm_s.py    # Inference script
├── ckpt/                             # Model checkpoints
├── outputs/
│   └── images/
│       └── LR/                       # Reconstructed low-res images
├── asset.jpg
├── requirements.txt
└── README.md
```

## License

MIT
