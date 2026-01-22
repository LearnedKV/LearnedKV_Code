# policy/loss.py

from __future__ import annotations
import torch
from torch import Tensor
import torch.nn.functional as F


def attention_mass_loss(
    alpha: Tensor,
    u: Tensor,
    c_target: Tensor | float,
    min_tokens_per_head: float = 0.0,
    eps: float = 1e-8,
) -> tuple[Tensor, dict]:
    """
    Attention-mass coverage loss with continuous interpolation.

    参数:
      - alpha: [B, L, H]  policy logits
      - u: [B, L, H, T_kv]  key-level importance (sum over queries)
      - c_target: float 或 [B] 张量, 全局压缩比 in (0, 1]
      - min_tokens_per_head: 每个 head 的最小 token 数 (实数, 连续容量, 训练时用于约束)
      - eps: 数值稳定

    返回:
      - loss: 标量
      - info: 一些中间统计 (可选用于 log)
    """
    assert alpha.dim() == 3 and u.dim() == 4, "shape mismatch: alpha[B,L,H], u[B,L,H,T_kv]"
    B, L, H, T_kv = u.shape
    N = L * H

    # softmax over heads
    alpha_flat = alpha.view(B, -1)  # [B, N]
    w = F.softmax(alpha_flat, dim=-1).view(B, L, H)  # [B,L,H]

    # 广播 c_target 到 batch
    if not torch.is_tensor(c_target):
        c_target = torch.full((B,), float(c_target), dtype=alpha.dtype, device=alpha.device)
    else:
        c_target = c_target.to(alpha.device).view(B)

    # 总 token 容量
    C_full = float(T_kv * N)
    # [B]
    B_total = c_target * C_full

    # per-head 最小子预算 (token 数)
    C_min = float(min_tokens_per_head)
    # C_pred: [B, L, H]
    C_pred = C_min + w * (B_total.view(B, 1, 1) - N * C_min)
    # 连续容量 k*: [0, T_kv]
    k_star = C_pred.clamp(min=0.0, max=float(T_kv) - 1e-6)

    # 归一化 u, 然后对每个 head 排序
    # u_sorted: [B,L,H,T_kv]
    u_reshaped = u.clamp(min=0.0)
    u_sum = u_reshaped.sum(dim=-1, keepdim=True)  # [B,L,H,1]
    u_sum = u_sum + eps
    u_norm = u_reshaped / u_sum

    u_sorted, _ = torch.sort(u_norm, dim=-1, descending=True)  # [B,L,H,T_kv]
    g = u_sorted.cumsum(dim=-1)  # [B,L,H,T_kv]
    # g_padded[j] = coverage when keeping top-j tokens, j∈[0..T_kv]
    zeros = torch.zeros_like(g[..., :1])
    g_padded = torch.cat([zeros, g], dim=-1)  # [B,L,H,T_kv+1]

    # 线性插值
    # k_star in [0, T_kv)
    j0 = torch.floor(k_star).long()  # [B,L,H]
    j0 = j0.clamp(min=0, max=T_kv - 1)
    delta = (k_star - j0.float()).clamp(min=0.0, max=1.0)

    # gather g(j0) and g(j0+1)
    idx_j0 = j0.unsqueeze(-1)  # [B,L,H,1]
    idx_j1 = (j0 + 1).clamp(max=T_kv).unsqueeze(-1)

    g_j0 = torch.gather(g_padded, dim=-1, index=idx_j0).squeeze(-1)  # [B,L,H]
    g_j1 = torch.gather(g_padded, dim=-1, index=idx_j1).squeeze(-1)  # [B,L,H]

    cov = (1.0 - delta) * g_j0 + delta * g_j1  # [B,L,H]
    loss_per_head = 1.0 - cov
    loss = loss_per_head.mean()

    info = {
        "mean_cov": cov.mean().item(),
        "mean_k_star": k_star.mean().item(),
        "c_target_mean": c_target.mean().item(),
    }
    return loss, info