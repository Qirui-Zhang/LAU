import torch
import torch.nn as nn
import numpy as np
from ..brain_signal_encoder import BrainSignalEncoder

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class EEGProjectLayer(nn.Module):
    def __init__(self, z_dim=1024, c_num=63, timesteps=[0,250], drop_proj=0.3):
        super(EEGProjectLayer, self).__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps

        self.input_dim = self.c_num * (self.timesteps[1] - self.timesteps[0])
        proj_dim = z_dim

        self.model = nn.Sequential(nn.Linear(self.input_dim, proj_dim),
                                   ResidualAdd(nn.Sequential(
                                       nn.GELU(),
                                       nn.Linear(proj_dim, proj_dim),
                                       nn.Dropout(drop_proj),
                                   )),
                                   nn.LayerNorm(proj_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.view(x.shape[0], self.input_dim)
        x = self.model(x)
        return x

class EEGProject_Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            # 重塑为卷积输入格式
            nn.Unflatten(1, (256, 2, 2)),
            # 上采样模块
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 4x4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1),  # 128x128
            nn.Tanh()  # 输出归一化到[0,1]
        )

    def forward(self, x):
        # 将tanh输出[-1,1]线性映射到[0,255]
        return self.decoder(x) * 127.5 + 127.5

class EEGProject_Encoder(BrainSignalEncoder):
    def __init__(self, z_dim=1024, c_num=63, timesteps=[0,250], drop_proj=0.3):
        super(EEGProject_Encoder, self).__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps
        self.drop_proj = drop_proj

        self.eeg_project_layer = EEGProjectLayer(self.z_dim, self.c_num, self.timesteps, self.drop_proj)
        self.adapter = EEGProject_Adapter()

    def encode(self, x, subject_ids=None):
        return self.eeg_project_layer(x)

    def get_adapter(self):
        return self.adapter