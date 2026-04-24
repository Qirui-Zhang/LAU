import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
from torchvision.transforms import Normalize
import open_clip
from pytorch_msssim import SSIM, MS_SSIM


# ------------------- SSIM Loss -------------------
class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

# ------------------- SSIM Loss -------------------
class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIM(
            data_range=255,
            channel=3,
            size_average=True
        )

    def forward(self, pred, target):
        return 1 - self.ssim(pred, target)

# ------------------- MS-SSIM Loss -------------------
class MS_SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ms_ssim = MS_SSIM(
            data_range=255,
            channel=3,
            size_average=True
        )

    def forward(self, pred, target):
        return 1 - self.ms_ssim(pred, target)


# ------------------- Perceptual Loss (VGG19-based) -------------------
class PerceptualLoss(nn.Module):
    def __init__(self, device="cuda", layer_indices=None, use_normalization=True):
        super().__init__()
        vgg = models.vgg19(weights='IMAGENET1K_V1').features.eval()

        self.layer_indices = layer_indices or [4, 9, 18, 27]
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ) if use_normalization else None

        self.layers = nn.ModuleList()
        prev_idx = 0
        for idx in self.layer_indices:
            self.layers.append(vgg[prev_idx:idx])
            prev_idx = idx

        self.layers = self.layers.to(device)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_norm = input / 255.0
        target_norm = target / 255.0

        if self.normalize is not None:
            input_norm = self.normalize(input_norm)
            target_norm = self.normalize(target_norm)

        loss = 0.0
        x, y = input_norm, target_norm
        for layer in self.layers:
            x = layer(x)
            y = layer(y)
            loss += F.mse_loss(x, y)

        return loss / len(self.layers)

# ------------------- Edge-aware Loss (Sobel-based) -------------------
class EdgeAwareLoss(nn.Module):
    def __init__(self, epsilon=1e-6, device="cuda"):
        super().__init__()
        self.epsilon = epsilon
        self.device = device

        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)

        sobel_x_kernel = torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], dtype = torch.float32) / 8.0
        sobel_y_kernel = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], dtype = torch.float32) / 8.0

        self.sobel_x.weight.data = sobel_x_kernel
        self.sobel_y.weight.data = sobel_y_kernel

        for param in self.parameters():
            param.requires_grad = False

        self.to(device)

    def rgb_to_gray(self, x):
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def forward(self, input, target):
        input_norm = input / 255.0
        target_norm = target / 255.0

        input_gray = self.rgb_to_gray(input_norm)
        target_gray = self.rgb_to_gray(target_norm)

        grad_x = self.sobel_x(input_gray) - self.sobel_x(target_gray)
        grad_y = self.sobel_y(input_gray) - self.sobel_y(target_gray)

        edge_loss = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.epsilon)
        return torch.mean(edge_loss)

# LowLevelLoss:SmoothL1Loss, SSIMLoss
class LowLevelLoss(nn.Module):
    def __init__(self, alpha=1, beta=100, device="cuda"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.device = device

        self.smooth_l1_loss = nn.SmoothL1Loss().to(device)
        self.ssim_loss = SSIMLoss().to(device)

    def forward(self, pred, target):
        pred = pred.to(self.device)
        target = target.to(self.device)

        smooth_l1_loss = self.smooth_l1_loss(pred, target)
        ssim_loss = self.ssim_loss(pred, target)

        total_loss = self.alpha * smooth_l1_loss + self.beta * ssim_loss
        return total_loss, smooth_l1_loss, ssim_loss

class MultiObjectiveLoss(nn.Module):
    def __init__(self, active_components, alpha=1, beta=1, gamma=1, delta=1, device="cuda"):
        super().__init__()
        self.active_components = set(active_components)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.smooth_l1 = nn.SmoothL1Loss().to(device) if 'smooth_l1' in self.active_components else None
        self.ssim = SSIMLoss().to(device) if 'ssim' in self.active_components else None
        self.perceptual = PerceptualLoss(device=device).to(device) if 'perceptual' in self.active_components else None
        self.edge = EdgeAwareLoss(device=device).to(device) if 'edge' in self.active_components else None

    def forward(self, pred, target):
        losses = {}
        total_loss = 0

        if self.smooth_l1:
            losses['smooth_l1'] = self.smooth_l1(pred, target)
            total_loss += self.alpha * losses['smooth_l1']

        if self.ssim:
            losses['ssim'] = self.ssim(pred, target)
            total_loss += self.beta * losses['ssim']

        if self.perceptual:
            losses['perceptual'] = self.perceptual(pred, target)
            total_loss += self.gamma * losses['perceptual']

        if self.edge:
            losses['edge'] = self.edge(pred, target)
            total_loss += self.delta * losses['edge']

        return total_loss, losses
