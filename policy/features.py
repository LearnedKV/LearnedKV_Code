# policy/features.py

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor


class HeadFeatureExtractor(nn.Module):
    """
    Phi(K, V) -> s^{(l,h)}

    输入:
      - K: [T_kv, d_head]
      - V: [T_kv, d_head]
      - layer_idx: int, 0-based
      - head_idx: int, 0-based
      - num_layers: int
      - num_heads: int

    输出:
      - feature: [d_feat]
    """

    def __init__(
        self,
        d_head: int,
        num_segments: int = 8,
        proj_dim: int = 8,
        var_proj_dim: int = 8,
    ) -> None:
        super().__init__()
        self.d_head = d_head
        self.num_segments = num_segments

        # 均值向量降维
        self.mu_K_proj = nn.Linear(d_head, proj_dim)
        self.mu_V_proj = nn.Linear(d_head, proj_dim)

        # 方差向量降维
        self.var_K_proj = nn.Linear(d_head, var_proj_dim)
        self.var_V_proj = nn.Linear(d_head, var_proj_dim)

        # d_feat = muK + muV + varK + varV + mK + mV + seg(2*num_segments) + struct(2)
        self._d_feat = (
            proj_dim
            + proj_dim
            + var_proj_dim
            + var_proj_dim
            + 1
            + 1
            + 2 * num_segments
            + 2
        )

    @property
    def feature_dim(self) -> int:
        return self._d_feat

    def forward(
        self,
        K: Tensor,
        V: Tensor,
        layer_idx: int,
        head_idx: int,
        num_layers: int,
        num_heads: int,
    ) -> Tensor:
        """
        K, V: [T_kv, d_head]
        返回: [d_feat]
        """
        # 基本统计
        # [d_head]
        mu_K = K.mean(dim=0)
        mu_V = V.mean(dim=0)

        # [d_head]
        sigma_K2 = ((K - mu_K) ** 2).mean(dim=0)
        sigma_V2 = ((V - mu_V) ** 2).mean(dim=0)

        # 标量范数统计
        # [T_kv] -> 标量
        m_K = K.norm(dim=-1).mean()
        m_V = V.norm(dim=-1).mean()

        # 均值 & 方差降维
        # [proj_dim]
        mu_K_proj = self.mu_K_proj(mu_K)
        mu_V_proj = self.mu_V_proj(mu_V)

        # [var_proj_dim]
        var_K_proj = self.var_K_proj(sigma_K2)
        var_V_proj = self.var_V_proj(sigma_V2)

        # 位置分段统计
        T_kv = K.size(0)
        seg_feats = []
        if self.num_segments > 0 and T_kv > 0:
            seg_len = max(1, T_kv // self.num_segments)
            for seg_idx in range(self.num_segments):
                start = seg_idx * seg_len
                if seg_idx == self.num_segments - 1:
                    end = T_kv
                else:
                    end = min(T_kv, (seg_idx + 1) * seg_len)
                if end <= start:
                    # 空段，用 0 代替
                    seg_feats.append(K.new_zeros(1)[0])
                    seg_feats.append(K.new_zeros(1)[0])
                    continue
                K_seg = K[start:end]  # [seg_len, d_head]
                V_seg = V[start:end]

                # 段内均值向量的 L2 范数
                seg_mu_K = K_seg.mean(dim=0).norm()
                seg_mu_V = V_seg.mean(dim=0).norm()
                seg_feats.append(seg_mu_K)
                seg_feats.append(seg_mu_V)
        seg_feats = torch.stack(seg_feats) if len(seg_feats) > 0 else K.new_zeros(2 * self.num_segments)

        # 结构特征 [2]
        l_norm = float(layer_idx) / max(1, (num_layers - 1))
        h_norm = float(head_idx) / max(1, (num_heads - 1))
        f_struct = K.new_tensor([l_norm, h_norm])

        # 拼接
        feat = torch.cat(
            [
                mu_K_proj,
                mu_V_proj,
                var_K_proj,
                var_V_proj,
                m_K.view(1),
                m_V.view(1),
                seg_feats,
                f_struct,
            ],
            dim=0,
        )
        assert feat.numel() == self._d_feat
        return feat