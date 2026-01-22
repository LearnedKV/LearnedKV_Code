# policy/model.py

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor


class PolicyMLP(nn.Module):
    """
    共享的 head-level policy MLP:

    输入:
      - head_features: [B, L, H, d_feat]

    输出:
      - alpha: [B, L, H]  (per-head logits)
    """

    def __init__(self, d_feat: int, d_hid: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_feat, d_hid)
        self.fc2 = nn.Linear(d_hid, 1)
        self.act = nn.GELU()

    def forward(self, head_features: Tensor) -> Tensor:
        """
        head_features: [B, L, H, d_feat]
        return: alpha: [B, L, H]
        """
        B, L, H, D = head_features.shape
        x = head_features.view(B * L * H, D)
        x = self.act(self.fc1(x))
        alpha = self.fc2(x)  # [B*L*H, 1]
        alpha = alpha.view(B, L, H)
        return alpha