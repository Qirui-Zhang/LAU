from abc import ABC, abstractmethod
import torch.nn as nn


class BrainSignalEncoder(ABC, nn.Module):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.adapter = None

    @abstractmethod
    def encode(self, x, subject_ids=None):
        """核心编码方法，返回原始编码结果"""
        pass

    @abstractmethod
    def get_adapter(self) -> nn.Module:
        """获取适配器模块"""
        pass

    def forward(self, x, subject_ids=None):
        """完整流程：编码+适配"""
        features = self.encode(x, subject_ids=subject_ids)
        return self.adapter(features)